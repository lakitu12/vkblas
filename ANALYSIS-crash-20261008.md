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

## 修复内容 (repo 工作树, 未提交)
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
| 失效探针 (30 轮 free/reuse) | ALL-OK; hook 61 次; 180 import 全 MISS | reuse 0/30 (本进程未回池 — 弱覆盖, 记局限) |
| faults (boot 起) | 0 | — |

## 待办 (上机门)
- [ ] **受控满容复现** `~/code/probes/vkblas_cache_overflow_repro.py` (需用户在场; 旧版在该脚本阶段 1 即应崩, 新版预期 DONE-CLEAN)
- [ ] 全服务端跑图 (r10_shim + vkblas 修复版) 与 OLD 对照 (含 CLIP 分叉复验)
- [ ] 部署 `sudo make install` (→ /opt/rocm/lib + vkblas-shaders; 覆盖前备份 4298ef1a)
- [ ] 遗留 (可选): IC_CAP 提升 + LRU; stride uint32 截断

## 产物索引
- 源码: `src/ic_cache.h`(新), `test/test_ic_cache.c`(新), `src/vkblas.c`(ic 段重写), `src/vkblas_hipblas.c`(兼容块), Makefile/README/vkblas.h
- 探针与脚本 (~/code/probes/): `vkblas_fix_smoke.sh`, `vkblas_ab_check.sh`, `vkblas_cache_overflow_repro.py`,
  `vkblas_invalidation_probe.py`, `hsa_pointer_info_probe.c`; 日志 `_smoke_*` / `_ab_*` / `_inval_probe.log` / `vkblas_fix_smoke.log` / `vkblas_ab.log`
