# test_torch.py — PyTorch 端到端: 正确性 + 性能 (LD_PRELOAD 下运行)
import torch
import time

torch.manual_seed(42)
dev = 'cuda'

# ---- 正确性: 各种形状 vs CPU ----
print("=== 正确性 ===")
shapes = [
    (128, 128, 128), (512, 512, 512), (4096, 4096, 4096),
    (4096, 1024, 4096), (1024, 4096, 4096), (257, 129, 64),
    (33, 65, 17), (1, 64, 64), (64, 1, 64), (3, 5, 7),
]
bad = 0
for (m, n, k) in shapes:
    a = torch.randn(m, k, device=dev)
    b = torch.randn(k, n, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    ref = (a.cpu() @ b.cpu())
    d = (c.cpu() - ref).abs().max().item()
    ok = d < 0.1
    bad += 0 if ok else 1
    print(f"  {m}x{n}x{k}: maxdiff={d:.2e} {'OK' if ok else 'FAIL'}")

# batched
a = torch.randn(4, 64, 32, device=dev)
b = torch.randn(4, 32, 128, device=dev)
c = torch.bmm(a, b)
torch.cuda.synchronize()
d = (c.cpu() - torch.bmm(a.cpu(), b.cpu())).abs().max().item()
bad += 0 if d < 0.1 else 1
print(f"  bmm 4x64x32x128: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# linear 层 (PyTorch 常用: x @ W.t())
x = torch.randn(128, 256, device=dev)
w = torch.randn(512, 256, device=dev)
c = x @ w.t()
torch.cuda.synchronize()
d = (c.cpu() - x.cpu() @ w.cpu().t()).abs().max().item()
bad += 0 if d < 0.1 else 1
print(f"  linear x@W.t 128x256x512: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# 坏项回归: fp32 非连续 x.t()@w (真库 Ailk_Bjlk->802 坏, vkblas 已根治)
# x 连续 [K,M], x.t() 是 [M,K] 转置视图 (stride=(1,K)) → hipblas transA=T
for (M, K, N) in [(33, 97, 65), (128, 256, 128), (512, 768, 256)]:
    x = torch.randn(K, M, device=dev)
    w = torch.randn(K, N, device=dev)
    c = x.t() @ w
    torch.cuda.synchronize()
    d = (c.cpu() - x.t().cpu() @ w.cpu()).abs().max().item()
    bad += 0 if d < 0.1 else 1
    print(f"  x.t()@w {M}x{K}x{N}: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# batched 转置 A: (B,K,M).transpose(1,2) @ (B,K,N)
x = torch.randn(3, 256, 320, device=dev)
w = torch.randn(3, 256, 128, device=dev)
c = x.transpose(1, 2) @ w
torch.cuda.synchronize()
d = (c.cpu() - x.transpose(1, 2).cpu() @ w.cpu()).abs().max().item()
bad += 0 if d < 0.1 else 1
print(f"  bmm xt@w 3x320x256x128: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# alpha/beta: addmm(bias, x.t(), w) — beta 乘 bias (无 C 参数)
x = torch.randn(256, 320, device=dev)
w = torch.randn(256, 128, device=dev)
bias = torch.randn(128, device=dev)
out = torch.addmm(bias, x.t(), w, beta=0.5, alpha=2.5)
torch.cuda.synchronize()
d = (out.cpu() - (0.5 * bias.cpu() + 2.5 * (x.t().cpu() @ w.cpu()))).abs().max().item()
bad += 0 if d < 0.1 else 1
print(f"  addmm xt@w alpha=2.5 beta=0.5: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# alpha/beta (addmm)
a = torch.randn(256, 256, device=dev)
b = torch.randn(256, 256, device=dev)
c = torch.randn(256, 256, device=dev)
out = torch.addmm(c, a, b, beta=0.7, alpha=0.5)
torch.cuda.synchronize()
ref = 0.7 * c.cpu() + 0.5 * (a.cpu() @ b.cpu())
d = (out.cpu() - ref).abs().max().item()
bad += 0 if d < 0.1 else 1
print(f"  addmm 256: maxdiff={d:.2e} {'OK' if d < 0.1 else 'FAIL'}")

# fp16 (vkblas 接管: PyTorch 走 hipblaslt_ext → gfx803 fallback 直连 rocblas_gemm_ex, 由 rocblas 层拦截)
a16 = torch.randn(128, 128, device=dev, dtype=torch.float16)
b16 = torch.randn(128, 128, device=dev, dtype=torch.float16)
c16 = a16 @ b16
torch.cuda.synchronize()
d = (c16.cpu().float() - (a16.cpu().float() @ b16.cpu().float())).abs().max().item()
print(f"  fp16 128: maxdiff={d:.2e} (vkblas)")

# complex128 (Zgemm 回退: 拆 4×fp64 GEMM; 注意 complex linear = x@w^T 不共轭!)
for (m, n, k) in [(16, 16, 16), (33, 65, 17), (128, 256, 320), (512, 256, 768)]:
    a = torch.randn(m, k, device=dev, dtype=torch.complex128)
    b = torch.randn(k, n, device=dev, dtype=torch.complex128)
    c = a @ b
    torch.cuda.synchronize()
    d = (c.cpu() - a.cpu() @ b.cpu()).abs().max().item()
    bad += 0 if d < 1e-8 else 1
    print(f"  z64 matmul {m}x{n}x{k}: maxdiff={d:.2e} {'OK' if d < 1e-8 else 'FAIL'}")

# complex128 batched (bmm)
a = torch.randn(3, 64, 32, device=dev, dtype=torch.complex128)
b = torch.randn(3, 32, 48, device=dev, dtype=torch.complex128)
c = torch.bmm(a, b)
torch.cuda.synchronize()
d = (c.cpu() - torch.bmm(a.cpu(), b.cpu())).abs().max().item()
bad += 0 if d < 1e-8 else 1
print(f"  z64 bmm 3x64x32x48: maxdiff={d:.2e} {'OK' if d < 1e-8 else 'FAIL'}")

# complex128 linear: x @ W.t() (不共轭)
x = torch.randn(64, 128, device=dev, dtype=torch.complex128)
w = torch.randn(256, 128, device=dev, dtype=torch.complex128)
c = x @ w.t()
torch.cuda.synchronize()
d = (c.cpu() - x.cpu() @ w.cpu().t()).abs().max().item()
bad += 0 if d < 1e-8 else 1
print(f"  z64 linear x@W.t 64x128x256: maxdiff={d:.2e} {'OK' if d < 1e-8 else 'FAIL'}")

# complex128 非连续 x.t()@w
x = torch.randn(97, 33, device=dev, dtype=torch.complex128)
w = torch.randn(97, 65, device=dev, dtype=torch.complex128)
c = x.t() @ w
torch.cuda.synchronize()
d = (c.cpu() - x.t().cpu() @ w.cpu()).abs().max().item()
bad += 0 if d < 1e-8 else 1
print(f"  z64 x.t()@w 33x97x65: maxdiff={d:.2e} {'OK' if d < 1e-8 else 'FAIL'}")

# complex128 alpha/beta: addmm (复数 alpha/beta)
a = torch.randn(64, 128, device=dev, dtype=torch.complex128)
b = torch.randn(128, 96, device=dev, dtype=torch.complex128)
c0 = torch.randn(64, 96, device=dev, dtype=torch.complex128)
alpha = torch.complex(torch.tensor(0.5), torch.tensor(-0.25)).to(dev)
beta = torch.complex(torch.tensor(1.5), torch.tensor(0.75)).to(dev)
out = torch.addmm(c0, a, b, beta=beta, alpha=alpha)
torch.cuda.synchronize()
ref = beta.cpu() * c0.cpu() + alpha.cpu() * (a.cpu() @ b.cpu())
d = (out.cpu() - ref).abs().max().item()
bad += 0 if d < 1e-8 else 1
print(f"  z64 addmm alpha/beta complex: maxdiff={d:.2e} {'OK' if d < 1e-8 else 'FAIL'}")

print(f"\n=== {'ALL PASS' if bad == 0 else f'{bad} FAILURES'} ===\n")

# ---- 性能 ----
print("=== 性能 (4096³ TT) ===")
a = torch.randn(4096, 4096, device=dev)
b = torch.randn(4096, 4096, device=dev)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
best = 1e9
for _ in range(5):
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    best = min(best, time.time() - t0)
print(f"  4096³ matmul: {best*1000:.1f} ms | {2*4096**3/best/1e12:.2f} TFLOPS")

# 小矩阵吞吐 (PyTorch 常见)
print("\n=== 小矩阵延迟 ===")
for (m, n, k) in [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]:
    a = torch.randn(m, k, device=dev)
    b = torch.randn(k, n, device=dev)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(10):
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    print(f"  {m}³: {best*1000:.2f} ms | {2*m**3/best/1e12:.2f} TFLOPS")
