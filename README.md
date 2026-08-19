# vkblas — Vulkan compute BLAS with a hipBLAS ABI compatibility layer

Vulkan compute implementation of BLAS (currently fp32 GEMM) for AMD GPUs, with a
hipBLAS ABI-compatible shim that lets PyTorch (and any ROCm software) run GEMM on
the Vulkan compute pipeline **without source changes**, via `LD_PRELOAD`.

Tuned and verified on **gfx803 (Polaris / RX 470-class, 36 CU)** with RADV.

## Performance (4096³ fp32 GEMM, gfx803)

| Variant | Time | Notes |
|---|---|---|
| TT | ~54 ms | TB=1 native fast path — **2.54 TFLOPS** (62% of FMA peak) |
| TN | ~58 ms | host transpose of B → TB=1 fast path |
| NT | ~67 ms | TB=1 native fast path |
| NN | ~71 ms | host transpose of B → TB=1 fast path (PyTorch `a @ b` main path) |
| **PyTorch `a @ b`** | **74.6 ms / 1.84 TFLOPS** | via LD_PRELOAD, zero code change |

For comparison, rocBLAS on the same GPU takes ~270 ms (0.51 TFLOPS) for 4096³.

Correctness: full test suite passes — 4 transpose combos × 13 shapes (incl. 3×5×7,
64×1×64, padded leading dimensions), alpha/beta scaling, and PyTorch
matmul/bmm/linear/addmm end-to-end.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ PyTorch / any ROCm app (calls hipblas* symbols)          │
└──────────────────────────┬───────────────────────────────┘
                    LD_PRELOAD (no rebuild)
┌──────────────────────────▼───────────────────────────────┐
│ vkblas_hipblas.so — hipBLAS ABI shim                     │
│   fp32 GEMM → vkblas engine; everything else → dlsym     │
│   forward to the real hipblas library                    │
└──────────────────────────┬───────────────────────────────┘
┌──────────────────────────▼───────────────────────────────┐
│ vkblas.c — Vulkan compute engine                         │
│   • zero-copy HIP ↔ Vulkan: hsa_amd_portable_export_dmabuf│
│   • 4-variant GEMM shader (TA/TB transposition matrix)   │
│   • transpose shader (row-major B → TB=1 fast path)      │
│   • buffer pool, global serialized execution             │
└──────────────────────────────────────────────────────────┘
```

## Requirements

- ROCm 6.4.3 (headers + libamdhip64 + libhsa-runtime64), or adjust `ROCM=` in the Makefile
- `glslangValidator` (glslang) for shader compilation
- Vulkan loader + a Vulkan driver (RADV; the engine needs
  `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`)
- gfx803: run with `HSA_OVERRIDE_GFX_VERSION=8.0.3` (any gfx8/9 card works with the right override)

## Build & test

```sh
make                 # libvkblas_hipblas.so + test/test_gemm (+ shaders via glslang)
make clean           # remove build artifacts

# C-level correctness (CPU reference, column-major semantics)
HSA_OVERRIDE_GFX_VERSION=8.0.3 LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib ./test/test_gemm
# expect: ALL PASS (0 failures)

# PyTorch end-to-end (uses the venv's python; adjust path)
env -u PYTHONPATH -u PYTHONHOME \
  LD_PRELOAD=$PWD/libvkblas_hipblas.so \
  LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib \
  HSA_OVERRIDE_GFX_VERSION=8.0.3 \
  <your-python> test/test_torch.py
```

## How the LD_PRELOAD shim works

1. `hipblasCreate` must return a real handle (PyTorch calls the real library's
   `hipblasSetWorkspace` through `RTLD_NEXT`); our own stream/mode state lives in
   globals.
2. fp32 GEMM entry points are intercepted and mapped to the Vulkan engine
   (`transA=N/transB=N` → the NN variant, etc.). Everything else is forwarded to
   the real hipblas via `dlsym`.
3. HIP device pointers are exported as dma-buf fds
   (`hsa_amd_portable_export_dmabuf`) and imported into Vulkan as external
   memory — zero copy. Exports are cached per pointer (imports stay alive until
   the HIP block is freed): the shim hooks `hipFree`/`hipHostFree`/`hipFreeManaged`
   and invalidates cache entries by HIP block base, so a freed address that is
   re-malloc'd always gets a fresh export. The cache self-enables only once a
   free hook call proves all frees in the process route through us (LD_PRELOAD);
   plain-`dlopen` processes keep the old per-call export/import behaviour.
4. Row-major B (op_b=N) is transposed host-side by a dedicated transpose shader
   into column-major before the GEMM — a row-major direct read is ~9× slower on
   gfx803/RADV.

## GEMM kernel notes (what made it fast on gfx803)

- tile 64×64, 16 accumulators/thread, local 16×16 (256 threads = 4 waves), single
  buffered LDS 16.9 KB → 3 workgroups/CU = 12 waves
- LDS layout `[k][m]`/`[k][n]` with row stride 66 floats (≡2 mod 32 → bank groups
  of 16, 2-4 way write conflicts). Row strides that are ≡0 mod 32 collapse banks
  (16-way conflicts); strides ≡4 mod 16 break 16B alignment (b128 silently
  corrupts — 33-float strides produce wrong data!)
- inner loop: 4×ds_read_b64 + 16 FMA per k-step (0.5 LDS bank-cycles/FMA)
- 64-thread workgroups under-occupy gfx803 (6 waves/CU); 256-thread wgs with 3/CU
  is the sweet spot
- write-C pattern: wave covers 16 rows × 256B — write bandwidth on gfx803 is
  governed by the number of rows a wave touches (see `src/shaders/gemm_tmpl.comp`)
- transpose shader: vec4 reads + 4 scattered scalar writes + chunked dispatch
  (8 MB per dispatch; single dispatches ≥16384 workgroups collapse to 6 GB/s)

## Layout

```
src/vkblas.c            Vulkan engine (init, dma-buf import, GEMM/transpose)
src/vkblas_hipblas.c    hipBLAS ABI shim (LD_PRELOAD)
src/vkblas.h            public API
src/shaders/gemm_tmpl.comp   GEMM shader template (TA/TB → 4 variants)
src/shaders/transpose.comp   row-major → column-major transpose
test/test_gemm.c        C-level correctness (CPU reference)
test/test_torch.py      PyTorch end-to-end + perf
```
