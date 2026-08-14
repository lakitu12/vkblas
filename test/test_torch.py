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

# fp16 (应 fallback 到真 rocBLAS)
a16 = torch.randn(128, 128, device=dev, dtype=torch.float16)
b16 = torch.randn(128, 128, device=dev, dtype=torch.float16)
c16 = a16 @ b16
torch.cuda.synchronize()
d = (c16.cpu().float() - (a16.cpu().float() @ b16.cpu().float())).abs().max().item()
print(f"  fp16 128: maxdiff={d:.2e} (fallback)")

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
