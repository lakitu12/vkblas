// test/test_ic_cache.c — import 缓存表核心 (src/ic_cache.h) 纯 host 单测。
// 零 GPU / 驱动依赖, 不需要 LD_PRELOAD。构建: make test/test_ic_cache; 运行: ./test/test_ic_cache
//
// 覆盖 2026-10-08 崩溃修复 (ANALYSIS-crash-20261008.md 根因 #1/#2) 的表层行为:
//   [0] 修复前逻辑复刻: 满容量三连插入 → A 被销毁 (证明测试场景确实命中该类 bug)
//   [1] 修复后: 同场景 A/B/C 全存活, 逐出的只能是旧条目; holds / 计数语义正确
//   [2] holds 保护: 在用条目不被逐出; 全表在用 → 拒绝插入 (非破坏性降级)
//   [3] swap-remove 完整性: 逐出后其余条目全部完好可寻
//   [4] hit/release 计数平衡 + size/base 不符作废重导路径
//   [5] invalidate: 按真块基址精确清理 (含同块多条) / NULL 全清
#include "../src/ic_cache.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            fails++;                                       \
            printf("  FAIL: ");                            \
            printf(__VA_ARGS__);                           \
            printf("  [%s:%d]\n", __FILE__, __LINE__);     \
        }                                                  \
    } while (0)

// ---- 桩: 销毁记录 ----
#define MAXD 8192
static int destroyed_n = 0;
static VkBuffer destroyed_b[MAXD];
static void t_destroy(void* ctx, VkBuffer b, VkDeviceMemory m) {
    (void)ctx; (void)m;
    if (destroyed_n < MAXD) destroyed_b[destroyed_n] = b;
    destroyed_n++;
}
static int was_destroyed(VkBuffer b) {
    for (int i = 0; i < destroyed_n && i < MAXD; i++)
        if (destroyed_b[i] == b) return 1;
    return 0;
}

static ic_cache_t c;

static void* PTR(int id) { return (void*)(uintptr_t)(0x100000u + (unsigned)id); }
static const void* BASEP(int id) { return (const void*)(uintptr_t)(0x200000u + (unsigned)id); }
static VkBuffer HB(int id) { return (VkBuffer)(uintptr_t)(0x300000u + (unsigned)id); }
static VkDeviceMemory HM(int id) { return (VkDeviceMemory)(uintptr_t)(0x400000u + (unsigned)id); }

// 灌入 n 条 "旧代" 条目 (add 后立即 release → holds=0, 可逐出)
static void fill_released(int n, int id0) {
    for (int i = 0; i < n; i++) {
        CHECK(ic_cache_add(&c, PTR(id0 + i), BASEP(id0 + i), 64,
                           HB(id0 + i), HM(id0 + i), 0) == 0, "fill add #%d", i);
        CHECK(ic_cache_release(&c, HB(id0 + i), HM(id0 + i)) == 1, "fill release #%d", i);
    }
}

static uint32_t holds_of(int id) {
    int i = ic_cache_probe(&c, PTR(id));
    return i >= 0 ? c.tab[i].holds : 0xFFFFFFFFu;
}

int main(void) {
    printf("== ic_cache host unit test (zero GPU) ==\n");

    // [0] 修复前逻辑复刻 (历史行为演示): 固定毁槽 0 + swap-remove
    {
        printf("[0] pre-fix replica: 3 back-to-back inserts @full\n");
        VkBuffer tab[IC_CAP];
        int cnt = 0, dn = 0;
        VkBuffer dead[16];
        for (int i = 0; i < IC_CAP; i++) tab[cnt++] = HB(10000 + i);
        VkBuffer ins[3] = { HB(9001), HB(9002), HB(9003) };   // A, B, C
        for (int k = 0; k < 3; k++) {
            if (cnt >= IC_CAP) {          // 旧 ic_add: ic_drop(0)
                dead[dn++] = tab[0];      // 销毁槽 0
                tab[0] = tab[--cnt];      // swap-remove
            }
            tab[cnt++] = ins[k];          // 追加
        }
        int a_dead = 0;
        for (int i = 0; i < dn; i++)
            if (dead[i] == ins[0]) a_dead = 1;
        printf("  old logic: A destroyed = %s (expect yes -> harness catches this bug class)\n",
               a_dead ? "YES" : "no");
        CHECK(a_dead == 1, "replica should destroy A");
        CHECK(dn == 3, "3 evictions in replica (got %d)", dn);
    }

    // [1] 修复后: 满容量三连插入 A/B/C 全存活
    {
        printf("[1] fixed: 3 back-to-back inserts @full — A/B/C must survive\n");
        ic_cache_init(&c, t_destroy, NULL);
        destroyed_n = 0;
        fill_released(IC_CAP, 0);
        CHECK(c.cnt == IC_CAP, "full (cnt=%u)", c.cnt);
        CHECK(destroyed_n == 0, "no destruction while filling (got %d)", destroyed_n);
        int dn0 = destroyed_n;
        CHECK(ic_cache_add(&c, PTR(9001), BASEP(9001), 16, HB(9001), HM(9001), 0) == 0, "add A");
        CHECK(ic_cache_add(&c, PTR(9002), BASEP(9002), 16, HB(9002), HM(9002), 0) == 0, "add B");
        CHECK(ic_cache_add(&c, PTR(9003), BASEP(9003), 16, HB(9003), HM(9003), 0) == 0, "add C");
        CHECK(destroyed_n == dn0 + 3, "exactly 3 old entries evicted (got %d)", destroyed_n - dn0);
        CHECK(!was_destroyed(HB(9001)) && !was_destroyed(HB(9002)) && !was_destroyed(HB(9003)),
              "A/B/C all alive");
        CHECK(ic_cache_probe(&c, PTR(9001)) >= 0 && ic_cache_probe(&c, PTR(9002)) >= 0 &&
              ic_cache_probe(&c, PTR(9003)) >= 0, "A/B/C findable");
        CHECK(holds_of(9001) == 1 && holds_of(9002) == 1 && holds_of(9003) == 1,
              "A/B/C holds==1 (got %u/%u/%u)", holds_of(9001), holds_of(9002), holds_of(9003));
        CHECK(c.cnt == IC_CAP, "count stays at cap (cnt=%u)", c.cnt);
        CHECK(c.evictions == 3, "evictions counter == 3 (got %llu)", (unsigned long long)c.evictions);
        int dwork = destroyed_n;
        CHECK(ic_cache_release(&c, HB(9001), HM(9001)) == 1, "release A");
        CHECK(ic_cache_release(&c, HB(9002), HM(9002)) == 1, "release B");
        CHECK(ic_cache_release(&c, HB(9003), HM(9003)) == 1, "release C");
        CHECK(destroyed_n == dwork, "release must not destroy cached entries");
        CHECK(holds_of(9001) == 0, "A holds==0 after release (got %u)", holds_of(9001));
    }

    // [2] 全表在用 → 拒绝插入 (非破坏); 释放一条 → 下一个逐出victim就是它
    {
        printf("[2] all-held refusal + evict-after-release\n");
        ic_cache_init(&c, t_destroy, NULL);
        destroyed_n = 0;
        for (int i = 0; i < IC_CAP; i++)   // 全部 hold 住 (不 release)
            CHECK(ic_cache_add(&c, PTR(i), BASEP(i), 64, HB(i), HM(i), 0) == 0, "held add #%d", i);
        CHECK(c.cnt == IC_CAP && destroyed_n == 0, "all held, nothing destroyed");
        int r = ic_cache_add(&c, PTR(9999), BASEP(9999), 64, HB(9999), HM(9999), 0);
        CHECK(r == -1, "insert refused when all held (r=%d)", r);
        CHECK(destroyed_n == 0 && c.cnt == IC_CAP, "refusal is non-destructive");
        CHECK(c.refused == 1, "refused counter (got %llu)", (unsigned long long)c.refused);
        CHECK(ic_cache_release(&c, HB(5), HM(5)) == 1, "release #5");
        r = ic_cache_add(&c, PTR(9999), BASEP(9999), 64, HB(9999), HM(9999), 0);
        CHECK(r == 0, "insert ok after release");
        CHECK(was_destroyed(HB(5)), "victim == released entry");
        CHECK(!was_destroyed(HB(0)) && !was_destroyed(HB(6)), "other held entries untouched");
    }

    // [3] swap-remove 完整性: 逐出后其余条目全部可寻
    {
        printf("[3] swap-remove integrity\n");
        ic_cache_init(&c, t_destroy, NULL);
        destroyed_n = 0;
        fill_released(IC_CAP, 0);
        CHECK(ic_cache_add(&c, PTR(9001), BASEP(9001), 16, HB(9001), HM(9001), 0) == 0, "add");
        CHECK(was_destroyed(HB(0)), "victim entry-0 destroyed");
        int missing = 0;
        for (int i = 1; i < IC_CAP; i++)
            if (ic_cache_probe(&c, PTR(i)) < 0) missing++;
        CHECK(missing == 0, "all %d remaining old entries findable (missing=%d)", IC_CAP - 1, missing);
        CHECK(ic_cache_probe(&c, PTR(9001)) >= 0, "new entry findable");
    }

    // [4] hit/release 计数 + 作废重导
    {
        printf("[4] hit/release accounting + stale invalidation\n");
        ic_cache_init(&c, t_destroy, NULL);
        destroyed_n = 0;
        CHECK(ic_cache_add(&c, PTR(7), BASEP(7), 128, HB(7), HM(7), 0) == 0, "add");
        CHECK(ic_cache_release(&c, HB(7), HM(7)) == 1, "release");
        VkBuffer hb; VkDeviceMemory hm; VkDeviceSize ho;
        CHECK(ic_cache_hit(&c, PTR(7), 64, BASEP(7), &hb, &hm, &ho) == 0, "hit (size fits)");
        CHECK(hb == HB(7) && hm == HM(7) && ho == 0, "hit returns same handles");
        CHECK(holds_of(7) == 1, "holds++ on hit");
        CHECK(c.hits == 1, "hits counted");
        CHECK(ic_cache_release(&c, HB(7), HM(7)) == 1, "release after hit");
        CHECK(holds_of(7) == 0, "holds back to 0");
        CHECK(ic_cache_hit(&c, PTR(7), 256, BASEP(7), &hb, &hm, &ho) == -1, "size too big -> miss");
        CHECK(ic_cache_probe(&c, PTR(7)) < 0, "stale entry dropped");
        CHECK(was_destroyed(HB(7)), "stale destroyed");
        CHECK(ic_cache_add(&c, PTR(8), BASEP(8), 128, HB(8), HM(8), 0) == 0, "add 8");
        ic_cache_release(&c, HB(8), HM(8));
        CHECK(ic_cache_hit(&c, PTR(8), 64, BASEP(998), &hb, &hm, &ho) == -1, "base mismatch -> miss");
        CHECK(was_destroyed(HB(8)), "base-changed entry dropped");
        CHECK(ic_cache_release(&c, HB(12345), HM(12345)) == 0, "release not-found returns 0");
    }

    // [5] invalidate_base
    {
        printf("[5] invalidate\n");
        ic_cache_init(&c, t_destroy, NULL);
        destroyed_n = 0;
        for (int k = 0; k < 3; k++) {   // 同块三条 (模拟 interior views 同 base)
            CHECK(ic_cache_add(&c, PTR(100 + k), BASEP(100), 64, HB(100 + k), HM(100 + k), 0) == 0,
                  "add same-base %d", k);
            ic_cache_release(&c, HB(100 + k), HM(100 + k));
        }
        CHECK(ic_cache_add(&c, PTR(200), BASEP(200), 64, HB(200), HM(200), 0) == 0, "add other-base");
        ic_cache_release(&c, HB(200), HM(200));
        ic_cache_invalidate_base(&c, BASEP(100));
        CHECK(c.cnt == 1, "3 same-base dropped, 1 kept (cnt=%u)", c.cnt);
        CHECK(was_destroyed(HB(100)) && was_destroyed(HB(101)) && was_destroyed(HB(102)),
              "same-base all destroyed");
        CHECK(!was_destroyed(HB(200)), "other-base kept");
        CHECK(c.invalidated == 3, "invalidated counter (got %llu)", (unsigned long long)c.invalidated);
        ic_cache_invalidate_base(&c, NULL);
        CHECK(c.cnt == 0 && was_destroyed(HB(200)), "clear-all");
    }

    printf("== %s (destroyed=%d) ==\n", fails == 0 ? "ALL PASS" : "FAILURES", destroyed_n);
    if (fails) printf("%d check(s) failed\n", fails);
    return fails == 0 ? 0 : 1;
}
