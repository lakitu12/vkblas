# vkblas 整机集成崩溃 — 源码级分析报告
日期: 2026-10-08 (凌晨)    分析对象: ~/code/vkblas HEAD 81f391d (2026-08-26, 行为等同部署版 4298ef1a)
场景: torch 2.14 (r10test venv) + LD_PRELOAD vkblas_hipblas + VKBLAS_SHADER_DIR, 全 ComfyUI
      SD1.5 集成: ①f16 attention bmm 崩在 RADV (gdb 实抓); ②fp32 conv (force-fp32) 长进程内崩;
      ③伴随 10 个 GPU VM fault (读/写, 多 TC)。全部"长进程才崩、单测/短进程全过"。

---

## 根因 #1（主嫌，高置信）：全缓存时"单操作内≥3次 MISS 插入"自毁正在使用的缓冲

代码: src/vkblas.c
- `ic_add` (714):  `if (ic_cnt >= IC_CAP) ic_drop(0);`   // 满则"FIFO"逐出
- `ic_drop` (686):  destroy ic_tab[i].b/.m; `ic_tab[i] = ic_tab[--ic_cnt];`  // swap-remove 尾元素入槽

### 推演（缓存已满, ic_cnt == IC_CAP == 256, 槽位 0..255 全占）
同一操作内连续插入 A、B、C（如 attention 每个 batch 一次 import A/B/C；或任何 gemm 的 A/B/C 三连）：

| 步骤 | 动作 | 结果 |
|---|---|---|
| 插 A | drop(0): 毁旧槽0；槽0=旧槽255；追加 A 到槽255 | 缓存含 A |
| 插 B | drop(0): 毁(换进槽0的)旧条目；**槽0 = A**；追加 B | **A 被移到槽0** |
| 插 C | drop(0): **毁 A！**；槽0 = B；追加 C | **A 的 VkBuffer/VkDeviceMemory 已被销毁** |

要点：**第 3 次插入必定销毁第 1 次刚插入的条目**（不是"最旧"，是"第1个"——因 swap-remove + 固定毁槽0 的组合）。

### 为什么致命
- A 的 VkBuffer 句柄此刻仍被调用方持有（bA/mA），且即将（或已经）record 进 **尚未提交** 的
  command buffer（descriptor 指向它）。销毁后提交 → shader/驱动访问已释放的 BO →
  **use-after-free**。
- op 尾部 `release_ptr(mA,bA)` (833): 按 (buf,mem) 句柄值扫缓存表找不到 A（已被逐出）→
  **再次 vkDestroyBuffer+vkFreeMemory = 双重释放**（且句柄值可能已被新对象复用，扫描匹配本身不可靠）。

### 与所有现场证据的吻合
| 证据 | 吻合 |
|---|---|
| 崩溃前日志 `[vkblas] import cache MISS ptr=0x5a6c30200 ... (n=256)` | n=256 == IC_CAP，缓存刚满，MISS 正在插入 |
| gdb: #0 libvulkan_radeon ← #1 vkblas_gemm_f16 | RADV 内 UAF（提交/销毁已释放 BO 的 dma-buf 内存）|
| 10 条 GPU VM protection fault（多 TC 读/写）| shader 执行时 descriptor 指向已释放/重映射的 BO |
| 短测试/单算子全过 | 指针数 < 256，缓存不满，从不触发逐出 |
| 全网络（UNet/CLIP 一跑就有数百 unique 指针）| 缓存迅速打满 → 之后每个 ≥3 导入的操作必中招 |
| test_torch 全过 | 进程内仅少量张量，缓存未满 |
| VAE decode 单测过、整机崩 | 同上：单测指针少 |

### 触发条件总结（最小）
缓存满(≥256条) + 同一操作内 ≥3 次 ic_add(MISS)。第二个条件在全网络下"每 op 必现"
（每个 gemm 恰好 import A/B/C 或按 batch 反复 3 连）。

---

## 根因 #2（次嫌，确定性缺陷）：失效机制的两个前提都不成立

`ic_block_base()` (678) 注释称 "底层 HIP 块基址 (hipPointerGetAttributes().devicePointer)"。
**对 CLR 源码实锤这是错的**（~/code/rocm10-sys/projects/clr/hipamd/src/hip_memory.cpp:3936-3964，
即现役运行时自身的构建源码）:
```c
memObj = getMemoryObject(device, ptr, offset);       // 含内偏移的所属分配
attributes->devicePointer = devMem->virtualAddress() + offset;   // = 问询地址本身
```
- 它返回的是"问询指针在设备 VA 空间的表示"（== ptr），**不是分配基址**。
- 后果 1: 命中校验 `ic_block_base(ptr) == ic_tab[i].base` 恒真（双保险空转）。
- 后果 2: 失效键 `base` 实质==条目 ptr。hipFree(块基址) 只能失效"ptr 恰好==块基址"的条目；
  **interior view**（如 split 切片、offset 视图）的条目永远漏失效。
- 后果 3: torch 缓存分配器常态下回收块不调 hipFree（差不多只在 empty_cache/释放时），
  稳态复用块时旧条目留存——物理上同一 BO 尚可容忍，但一旦底层分配真的换 BO（VA 复用+新
  物理内存），陈旧 dma-buf 被复用 → UAF/错页。VMM/Async 分配路径（若有）完全绕过 hook。

---

## 次要发现
1. stride 截断: transpose 系列 push constant 把 int64 stride 强转 uint32（>4G 元素才炸，低危）。
2. `tc_tab`（转置缓存）同样以 (ptr,N,K,ldb,elem) 为键 + 同一 base 失效模型，同病同修。
3. "读域>物理大小"的 export 策略（覆盖 over-read 的备用 import）在分配尾部仍可能失败/截断
   （hsa_amd_portable_export_dmabuf 对超范围的行为未定义化处理）——注意保留，非本次主因。
4. IC_CAP=256 对全网络偏小；"FIFO"实为乱序（swap-remove 后顺序已乱）。

---

## 修复建议（按优先级）
1. **禁止操作内逐出**：条目记 op 代次(`gen`)，`ic_drop` 跳过 `gen==当前gen` 的条目（挑下一个
   候选），或把"已逐出"标记为 RETIRED、延迟到下次提交完成后统一销毁。单这一条即可消除 #1。
2. **release_ptr 与缓存统一生命周期**：不要按句柄值扫描；给每次 import 返回的条目加引用计数
   （op 级），op 结束后由缓存持有。已被逐出的句柄不得二次销毁。
3. **失效键改为真基址**：hook hipMalloc/hipMallocManaged/hipExtMallocWithFlags 维护
   (ptr,size)→记录表（或改用 hsa_amd_pointer_info 的 base 字段），free 时按"与释放区间重叠"
   失效所有条目（含 interior view）。
4. IC_CAP 提升 + 真正 LRU（或按 gen 批量清理），eviction 永不触碰当前 op。
5. （可选加固）对"过期条目 + size 不符"的命中路径，强制重导出而非静默复用。

## 最小验证方案（待用户批准后执行）
- 纯 host 侧：把 ic 表逻辑抽成单测（模拟 3 连插入 @满容量，断言第 1 条存活）——零 GPU 风险。
- 单进程 GPU 受控试验（一次操作 X，低风险）: 先循环做小 gemm 用 ≥300 个不同指针把缓存灌满，
  再跑一次 attention 形状 f16 gemm → 预期在旧代码崩（RADV/VM fault）、修复后干净。
  实验前须知会挂机风险；建议日志逐级 sync + 用户在场。

## 附: 现场文件
- gdb: /tmp/gdb_crash.log (已被重启清空; 摘要见 skill 18m)
- 崩溃前 trace: "[vk] f16 fused: import 194 cvtAB 0 bTsp 1 cvtC 0 gemm 1 f2h 226 us (total 422)"
- VM fault 段: boot -1 journal, Process python pid 541588, 0x2D0511~0x2D0A66 多 TC 读

---

# 【修复实施 + 验证记录】(2026-10-08 当天)

## 修复内容 (本地 commit 4cc07a5, 未 push)
1. **缓存表核心抽到 `src/ic_cache.h`** (纯 host 可测):
   - 每条目 `holds` 引用计数: import 返回句柄 +1, `release_ptr` 归还 -1;
   - **逐出只销毁 holds==0 的条目** (根因 #1 修复: 在用条目——同 op 早先导入 / 已被调用方持有——永不被逐出);
   - 全表在用 → `ic_cache_add` 拒绝插入 (非破坏性降级, 调用方退回"不缓存");
   - 单元测试含修复前逻辑复刻: 同场景 (满容 3 连插入) 旧逻辑必毁 A → 证明测试能捕获该类 bug。
2. **真块基址解析** 改 `hsa_amd_pointer_info().agentBaseAddress/hostBaseAddress` (根因 #2 修复):
   `vkblas_cache_invalidate_base(ptr)` 传入被释放指针, 内部解析真基址 → 同块全部条目
   (含 interior view) 失效; 解析失败 → 保守按 同 ptr/同 base 清理。tc 缓存同享新 resolver。
3. `release_ptr` 语义改为 holds 归还 (未命中才真销毁); tc 路径一处 `&&` 短路 hold 泄漏顺带修。
4. **构建代际兼容 (新发现, 必读)**: 源码原按 6.4 系 hipblas 头写成, 而现役 v10 树头 = hipblas 3.x
   (类型 `hipblasDatatype_t` 已删、`_v2` 函数不再声明、`HIPBLAS_R_*` 折叠为新值宏)。
   已在 vkblas_hipblas.c 加兼容块 (`hipblasVersionMajor>=3` 守卫): typedef 别名 + `_v2` 前向声明 +
   `is_*_ex` 老值字面量 (150/168, 反汇编部署版实锤) + `VKBLAS_GEMMEX_COMPUTE_T` 宏。
   构建: `make libvkblas_hipblas.so` (ROCM=/opt/rocm); 产物 **md5 6e25ce05**, 157440B,
   符号集与部署版 4298ef1a **逐符号一致**; RUNPATH=/opt/rocm/lib; NEEDED libamdhip64.so.7。

## 验证证据链 (本机实测, fault 全程 0)
| 项 | 结果 | 备注 |
|---|---|---|
| host 单测 `./test/test_ic_cache` | **ALL PASS** (6 场景) | 满容三连插入存活/holds 保护/拒绝/swap 完整性/失效 |
| pointer_info 语义探针 | 全绿 | interior(+256K/+1000) 同真基址; **对照组: hipPointerGetAttributes devicePointer==问询地址 (旧 resolver 缺陷实测实锤)**; host 路径 OK |
| 构建产物 | md5 6e25ce05 | 符号集=部署版; 警告均为既有风格 |
| test_gemm (烟测) | ALL PASS 95/0 | fp32 全形状 + merged + batch |
| test_cache 新 vs 老 A/B | **两版逐项一致** (ALL PASS, HIT=0/MISS=3) | HIT=0 系测试性质 (场景 A 在 hook 激活前; B 仅一次导入) |
| test_h 新×2 / 老×2 | 新 2/2 过; 老 1/2 复现同 case FAIL (6.25e-2) | 无种子随机 + bf16 1-ulp 边界抖动 → **非回归** |
| **满容复现 (受控, NEW)** | **DONE-CLEAN, faults 0/0** | **满容量插入 793 次 + HIT 593 + REFUSED 0** —— 旧版必崩条件反复执行零异常 (vkblas_repro_run.log) |
| 失效探针 (30 轮 free/reuse) | ALL-OK; hook 61 次; 180 import 全 MISS | reuse 0/30 (本进程未回池 — 弱覆盖, 记局限) |
| faults (boot 起) | 0 | — |

## 待办 (上机门)
- [x] **受控满容复现** (10-08 00:50): NEW DONE-CLEAN, faults 0; 满容量插入 793 次 / HIT 593 / REFUSED 0
- [x] **部署** (10-08): /opt/rocm/lib/libvkblas_hipblas.so md5 `47f07c4e` (install 版, 内嵌 shader 路径); 旧版备份 `~/rocm-gfx803-archive/vkblas/libvkblas_hipblas.so.bak-4298ef1a`; 部署版零环境冒烟通过 (shader-not-found 0 条)
- [x] **本地 commit** `4cc07a5` (未 push; 8 文件 +654/-106)
- [ ] 全服务端跑图 (r10_shim + vkblas 修复版) 与 OLD 对照 (含 CLIP 分叉复验)
- [ ] 遗留 (可选): IC_CAP 提升 + LRU; stride uint32 截断

## 产物索引
- 源码: `src/ic_cache.h`(新), `test/test_ic_cache.c`(新), `src/vkblas.c`(ic 段重写), `src/vkblas_hipblas.c`(兼容块), Makefile/README/vkblas.h
- 探针与脚本 (~/code/probes/): `vkblas_fix_smoke.sh`, `vkblas_ab_check.sh`, `vkblas_cache_overflow_repro.py`,
  `vkblas_invalidation_probe.py`, `hsa_pointer_info_probe.c`; 日志 `_smoke_*` / `_ab_*` / `_inval_probe.log` / `vkblas_fix_smoke.log` / `vkblas_ab.log`

---

# 【全服务端验证 — 门槛通过】(2026-10-08 上午, 实测)

夹具: `~/code/probes/run_vkblas_fullserver.py` (r10_shim/main.py 可换 + LD_PRELOAD + SD1.5 API workflow;
     逐级 fsync 落盘 / 显存采样 / 每轮查内核 fault / 命中即停 / 留存 journal+devcoredump)
工作流: `~/code/probes/workflows/sd15_dump.json` (512×512, 20 steps, euler, seed 1/2, ckpt v1-5 fp16)
日志: `~/code/probes/runlogs/vkfs_{base,base2,new,new2,ref28}_*` (driver.log / server.log / summary.json)

## 结果表 (全部 faults 全程 0, 无 VM fault / 无 ring timeout / 无挂机)
| 臂 | 配置 | 出图 | 显存峰值→出图后 | 缓存证据 |
|---|---|---|---|---|
| A | 2.14 不挂 vkblas, cpu-vae | 20/20 步完成但**全黑** 2001B (271.8s) | — | 无 |
| A2 | 同上 ×2 图 | 两图均黑 2002B, **无 OOM** | 4.9–5.3G → 3.3–3.7G | 无 |
| **B** | **2.14 + shim + vkblas 修复版 47f07c4e** | **正常图** 517879B mean113.584 std69.491 min0 max255 (277s) | — | MISS=21000 HIT=10821 REFUSED=0 hook=176 **n_max=256** |
| B2 | 同上复跑 ×2 图 | 像素与 B **完全相同** (相关1.000000 maxdiff=0); 第2图 OOM | **7.9–8.0G → 7817M(不回落)** | MISS=20996 HIT=10054 REFUSED=0 hook=173 n_max=256 |
| D | **2.8 轮子 (已知能出图)** 同种子 | 517788B mean113.427 std69.498 (278s) | 4.5G → 2927M | 无 |

**端到端像素比对**: vkblas修复版 vs 2.8轮子 = **相关 0.999397**, 平均|差| 1.05/255, 差>8 像素 2.05% ;
基线(黑) vs 2.8轮子 = **相关 0.000**。⇒ 修复版全服务端数值链与"已知能出图"路径等价 (CLIP 条件若坏不可能 0.9994 ⇒ 18p 的 CLIP 分叉线索随此收口)。

## 算子级定位 (CPU-RNG 输入消除污染; `nan_scan2.py` / `rng_case.py`)
| 算子 | 官方路径 | vkblas 修复版 |
|---|---|---|
| **fp16 einsum("b i d,b j d->b i j") 全量 b16/i1024/j4096/d40** | **相对误差 1.0 (输出全零 → NaN → 黑图)** | ✓ 7.6e-4 |
| fp16 einsum slice (非连续 q[:, :1024]) | ✓ 3.8e-4 | ✓ |
| fp32 matmul 512² / fp16 matmul 512² | ✓ 8e-7 / 2.8e-4 | ✓ 2.75e-7 / 5.6e-4 |
| silu / add / GroupNorm(fp32) | ✓ 精确 | ✓ 精确 |
| GPU randn (fp32/fp16) | ✓ 两臂逐值一致, 非零 | ✓ |
⇒ **官方路径上唯一坏的就是 fp16 `bmm`(strided-batched) 全量形状** —— 正是历史崩溃现场那条链
(`vkblas_gemm_f16 ← rocblas_gemm_strided_batched_ex ← torch bgemm→baddbmm→einsum`); vkblas 自实现它 ⇒ 出正常图。
(vkblas 修复版在全服务端顺带修掉"官方 rocBLAS 出不了图"这一现象。)

## 新发现 (非崩溃, 待修): vkblas 常驻显存 +~3GB
- vkblas 臂显存运行中即打满 ~7.9–8.0G 且**出图后不回落** (7817M); 基线/2.8 轮子出图后回落 (3.3G/2.9G)。
- 后果: 8G 卡上**第 2 张图必 OOM** (`Tried to allocate 16MiB`, 剩 100MiB, torch 仅占 1.50GiB) —— 是**容量问题非崩溃**。
- 时相: 占满发生在 t≈30s 后保持平坦 ⇒ 更像"固定多占 ~3G"(256 条 import 缓存 + Vulkan 侧开销)而非泄漏 (**推断, 未做归因实验**)。
- 候选修法: IC_CAP 降/LRU 化、prompt 结束后主动 flush 缓存、按 VRAM 压力降级不缓存。

## 未做 (需用户拍板)
- [ ] **OLD 对照臂** (旧 .so 4298ef1a 全服务端 或 受控满容复现): 预计必崩/可能挂机, 需用户手动硬重启 ⇒ 等指令
      (历史证据已足够: 同配置旧 .so 崩在 RADV, 10 条 VM fault, `MISS (n=256)`; 报告上半部分即该次分析)
- [ ] 显存 +3G 归因与修复

---

# 【显存 +3G / 第2图 OOM — 根因已定性】(10-08 上午, 全程零风险级: host + 小 shape 探针, faults 0)

**结论: 不是缓存设计问题, 而是 `HOOK_FREE` 把 free 转发到了错误的 HIP 运行时 ⇒ 带 vkblas 时
进程内每一次 hipFree 都【静默不释放】且返回成功。**

## 证据链 (全部本机实测)
1. `src/vkblas_hipblas.c:141 real_amdhip()` 只按名字 dlopen **libamdhip64.so.6**;
   本机该名字是兼容槽 symlink → `/opt/rocm/lib/libamdhip64.so.7.15.26333-bb6fb389f`,
   而 torch 实际链接/使用的是 `libamdhip64.so.6/.7` 之外那份 `.so.7 → ...0000000`
   ⇒ 进程内**同时映射两份 libamdhip64** (maps 实测: `...0000000` 4 段 + `...bb6fb389f` 5 段)。
2. 直调两份 runtime 的 hipFree 对**真 hipMalloc 分配**(torch.cuda.caching_allocator_alloc):
   | runtime | rc | 显存回降 |
   |---|---|---|
   | `.so.6` (bb6fb389f) | **0 (成功)** | **0–2M (静默 no-op)** |
   | `.so.7` (0000000) | 0 | **512M ✓ (真释放)** |
3. `hook_free_test`: 全局作用域 hipFree (有 preload 时 = vkblas hook) → **rc=0 但回降 0M**。
4. `vram_attr2` (300×5MiB + 300×1.25MiB, 全部被 import 过): 无 preload 释放后回降 1791M;
   有 vkblas **回降 0M**。⇒ 与"缓存条目钉住"无关 (条目自身仅 ~0.35MiB/条: 两臂 ops 后差 5M)。

## 机理与影响
- torch 的 free (empty_cache / 块回收 / 释放张量) → 被 hook 拦截 → 转发给 `.so.6` 实例 →
  该实例不拥有这些分配 → **返回成功但什么都没做** → 显存只增不减。
- 全服务端表现与此完全吻合: 第 1 张图运行中即打到 ~8.0G 并**出图后停在 7.8G 不回落**
  (基线出图后回落 3.3G) ⇒ 第 2 张图 OOM (容量问题, **非崩溃**, faults 全程 0)。
- 一般化: **任何 GPU 分配只要 vkblas 在场都不会真正归还** —— 长命进程会持续吃满显存。

## 修法 (建议, 未实施)
把 HOOK_FREE 的转发从"猜库文件名"改为**进程内正确的下一份定义**:
```c
#define _GNU_SOURCE            // 需要 RTLD_NEXT
#define HOOK_FREE(fname)                                                    \
    hipError_t fname(void* ptr) {                                           \
        static hipError_t (*real)(void*) = NULL;                            \
        if (!real) real = (hipError_t(*)(void*))dlsym(RTLD_NEXT, #fname);   \
        if (!real) { /* 显式报错, 别再静默 */ }                              \
        ...
```
并加**自检**: 若 `real == NULL` 或首次调用返回值异常要打日志 (现行实现静默返回 hipErrorRuntimeMemory)。
验证顺序: host 单测 → 冒烟 → 全服务端 3 张图 (期望: 显存出图后回落, 无 OOM, faults 0)。
回退: 保留现部署版 47f07c4e (archive 里另有 4298ef1a)。

---

# 【显存修复实施 + 验证】(10-08 中午) —— 已部署 md5 ae5d0958

## 改动 (src/vkblas_hipblas.c)
1. `#define _GNU_SOURCE`; `HOOK_FREE` 的转发目标改为 **`dlsym(RTLD_NEXT, #fname)`**
   (本 .so 由 LD_PRELOAD 插在搜索序最前 ⇒ RTLD_NEXT = 进程真正使用的那份 libamdhip64),
   并在解析失败/首次解析时打印真实地址 (不再静默返回 hipErrorRuntimeMemory)。
2. 兜底 `real_amdhip()` 由 `.so.6` 改为**首选 `.so.7`** (本机 .so.6 是兼容槽 symlink → 另一实例)。

## 证据 (修前 → 修后, 同探针)
| 检查 | 修前 (47f07c4e) | 修后 (ae5d0958) |
|---|---|---|
| hook 转发目标 | .so.6 实例 (地址 0x...7056a0) | **.so.7 实例 (地址 = .so.7 的 hipFree, 逐地址相等)** |
| `hook_free_test` 真分配 free | rc=0 但回降 **0M** | rc=0 回降 **514M** ✓ |
| `vram_attr2` 释放后 | 2698M (**回降 0M**) | **1115M (回降 1796M)** ≈ 无 preload 臂 1791M ✓ |
| host 单测 `test_ic_cache` | ALL PASS | ALL PASS (destroyed=4) |
| 烟测 (test_gemm/test_cache/test_h) | 95/0 + PASS + 0 失败 | **同** (faults 0) |
| **全服务端 3 张图** | 第 2 张 OOM | **3/3 完成, 无 OOM**; 出图后显存 3151/3223M (修前停 7817M); MISS=59435 HIT=28987 REFUSED=0 n_max=256; **faults 全程 0** |
| 数值回归 | — | 同种子图与修前 **逐像素完全一致 (max|diff|=0, 相关 1.000000)**; 与 2.8 轮子仍 0.999397 |

## 部署与回退
- 现部署: `/opt/rocm/lib/libvkblas_hipblas.so` = **ae5d0958** (install 版, 内嵌 shader 路径 `/opt/rocm/lib/vkblas-shaders`)
- repo 构建产物: `~/code/vkblas/libvkblas_hipblas.so` = c4f52303
- 回退: `~/rocm-gfx803-archive/vkblas/libvkblas_hipblas.so.bak-47f07c4e` (上一版, 无显存修复)
        或 `...bak-4298ef1a` (原始) → `sudo cp <bak> /opt/rocm-10.0.0/lib/libvkblas_hipblas.so`
- 日常: `LD_PRELOAD=/opt/rocm-10.0.0/lib/libvkblas_hipblas.so` (无需其它环境变量)

## 剩余
- [x] OLD 对照臂 (用户已同意: 修法验证后再跑; 受控复现版; 预计崩/可能挂机)

---

# 【OLD 对照臂 — 证伪闭环】(10-08 09:32, 同夹具同日 A/B)

配置: `~/code/probes/run_old_arm.sh` → 受控满容复现 (`vkblas_cache_overflow_repro.py`) × 旧 .so
      `~/rocm-gfx803-archive/vkblas/libvkblas_hipblas.so.bak-4298ef1a`, VKBLAS_TRACE=1, 日志逐行 sync

| | 旧 .so (4298ef1a) | 新 .so (ae5d0958) |
|---|---|---|
| 结果 | **rc=139 (SIGSEGV / core dump)** | DONE-CLEAN (本会话早先: 满容量插入 793 次) |
| 崩点 | 紧接 `import cache MISS ... (n=256)`(缓存刚打满) + **连续 3 连导入**之后 | — |
| 内核事件 | **+1: `GPU fault detected: 146 0x07e0480c`, `Process python pid 151373`, `VM_CONTEXT1_PROTECTION_FAULT_ADDR 0x001200FC`, `VM fault (0x0c, vmid 6, pasid 204) at page 1179900, read from 'TC4'`** | **0** |

⇒ 与根因 #1 的预测触发条件 (满缓存 + 单操作 ≥3 连插入) 与历史签名族 (TC4 读, python 进程) **逐项吻合**;
同日同夹具 A/B ⇒ **修复的因果性闭合** (旧版必崩, 新版零异常)。

**机器处置**: 本次**未级联** —— 事后 `ring timeout = 0` / `GPU reset = 0`、桌面进程完好、
轻量 GPU 复测 (64² matmul) 通过 ⇒ **无需硬重启** (与 10-07 两次级联挂机不同)。
日志: `~/code/probes/runlogs/oldarm_20261008_093221.log`

**全部上机门至此闭合**: 单测 → 烟测 → 受控满容复现 (新旧 A/B) → 全服务端 3 图 → 显存修复验证 → OLD 证伪。

