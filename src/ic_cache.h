// ic_cache.h — vkblas dma-buf import 缓存核心表 (纯 host 可测: 不依赖 Vulkan/GPU, 销毁走回调)
//
// 背景与正确性模型 (2026-10-08 修复, 见 ANALYSIS-crash-20261008.md):
//
//   修复前 (根因 #1): 缓存满 (IC_CAP) 时 ic_add 固定 ic_drop(0), 而 swap-remove 会把
//   尾部条目换进槽 0 —— 单操作内连续 3 次 MISS 插入 (A,B,C 三连, 全网络每 op 必现)
//   会 "第 3 次插入销毁第 1 次刚插入的条目": 该条目的 VkBuffer 正被 (即将) 引用进尚未
//   提交的 command buffer → RADV use-after-free (长进程崩溃 / GPU VM fault)。
//
//   修复: 每条目加 holds 引用计数 —— import 返回句柄时 +1, release_ptr 归还时 -1;
//   逐出只销毁 holds==0 的条目, 在用条目 (同 op 早先导入 / 已被调用方持有) 永不被逐出。
//   全表在用 (病态场景) 时 ic_cache_add 拒绝插入: 调用方退回 "不缓存" 语义, 不破坏他者。
//
//   约束: 每个 ic_cache_add / ic_cache_hit 成功后, 调用方 (vkblas.c) 必须在句柄用完后
//   恰好一次 ic_cache_release; holds 平衡是上述保证的前提。
//
//   失效键 base 须为【真块基址】(vkblas.c 用 hsa_amd_pointer_info 解析) —— 否则
//   interior view 条目漏失效 (根因 #2)。
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <vulkan/vulkan.h>

#define IC_CAP 256

typedef struct {
    void* ptr;              // 缓存 key (GEMM 传入的用户指针)
    const void* base;       // 真块基址 (free hook 失效键; 解析失败 → 不缓存)
    size_t size;            // 创建时的 buffer 字节数
    VkBuffer b;
    VkDeviceMemory m;
    VkDeviceSize off;       // dma-buf 内偏移
    uint32_t holds;         // 未 release 的 import 次数 (0 = 可逐出)
} ic_entry_t;

typedef void (*ic_destroy_fn)(void* ctx, VkBuffer b, VkDeviceMemory m);

typedef struct {
    ic_entry_t tab[IC_CAP];
    uint32_t cnt;
    uint64_t hits, misses, evictions, refused, invalidated;
    ic_destroy_fn destroy;
    void* ctx;
} ic_cache_t;

static inline void ic_cache_init(ic_cache_t* c, ic_destroy_fn fn, void* ctx) {
    c->cnt = 0;
    c->hits = c->misses = c->evictions = c->refused = c->invalidated = 0;
    c->destroy = fn;
    c->ctx = ctx;
}

// swap-remove 第 i 条并销毁其句柄 (调用方须持锁; 在用性由调用点保证)
static inline void ic_cache_drop(ic_cache_t* c, uint32_t i) {
    c->destroy(c->ctx, c->tab[i].b, c->tab[i].m);
    c->tab[i] = c->tab[--c->cnt];
}

// 逐出一条 holds==0 的条目; 0=已逐出, -1=全在用 (拒绝)
static inline int ic_cache_evict_one(ic_cache_t* c) {
    for (uint32_t i = 0; i < c->cnt; i++) {
        if (c->tab[i].holds == 0) {
            ic_cache_drop(c, i);
            c->evictions++;
            return 0;
        }
    }
    c->refused++;
    return -1;
}

// 插入新条目 (holds=1)。满 → 先逐出可逐出者; 全在用 → 拒绝 (-1, 不插入不破坏)
static inline int ic_cache_add(ic_cache_t* c, const void* ptr, const void* base,
                               size_t size, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    if (c->cnt >= IC_CAP && ic_cache_evict_one(c) != 0) return -1;
    ic_entry_t* e = &c->tab[c->cnt++];
    e->ptr = (void*)ptr;
    e->base = base;
    e->size = size;
    e->b = b;
    e->m = m;
    e->off = off;
    e->holds = 1;
    return 0;
}

// 命中查找: ptr 匹配 && size 够 && base 一致 → holds++ 并回填句柄, 返回 0;
// 不匹配的陈旧条目 (holds==0) 顺手作废 (swap-remove 后重查该槽); 无命中 → -1
static inline int ic_cache_hit(ic_cache_t* c, const void* ptr, size_t size,
                               const void* base_now,
                               VkBuffer* b, VkDeviceMemory* m, VkDeviceSize* off) {
    for (uint32_t i = 0; i < c->cnt; ) {
        if (c->tab[i].ptr != ptr) {
            i++;
            continue;
        }
        if (size <= c->tab[i].size && c->tab[i].base == base_now) {
            c->tab[i].holds++;
            c->hits++;
            *b = c->tab[i].b;
            *m = c->tab[i].m;
            *off = c->tab[i].off;
            return 0;
        }
        if (c->tab[i].holds == 0) {
            ic_cache_drop(c, i);   // 作废, swap-remove 后重新检查该槽
            continue;
        }
        i++;   // 在用但 size/base 不符 → 跳过 (调用方将新建条目)
    }
    return -1;
}

// release: 按 (b,m) 找到 → holds-- (saturate) 并返回 1 (条目归缓存持有, 调用方勿销毁);
// 未找到 → 0 (非缓存对象, 调用方自行销毁)
static inline int ic_cache_release(ic_cache_t* c, VkBuffer b, VkDeviceMemory m) {
    for (uint32_t i = 0; i < c->cnt; i++) {
        if (c->tab[i].b == b && c->tab[i].m == m) {
            if (c->tab[i].holds > 0) c->tab[i].holds--;
            return 1;
        }
    }
    return 0;
}

// 外部失效 (自由指针解析出的真块基址; base==NULL → 全清)。
// 语义: hipFree hook 路径已被 g.lock 串行化, 理论上不存在在用条目; 若确命中在用条目
// (调用方 bug), 一并销毁并计入 invalidated —— 内存已释放, 保留反而危险。
static inline void ic_cache_invalidate_base(ic_cache_t* c, const void* base) {
    for (uint32_t i = 0; i < c->cnt; ) {
        if (base == NULL || c->tab[i].base == base) {
            ic_cache_drop(c, i);
            c->invalidated++;
        } else {
            i++;
        }
    }
}

// 测试/诊断: 返回第一条 ptr 匹配条目下标 (不改动状态), 无 → -1
static inline int ic_cache_probe(const ic_cache_t* c, const void* ptr) {
    for (uint32_t i = 0; i < c->cnt; i++)
        if (c->tab[i].ptr == ptr) return (int)i;
    return -1;
}
