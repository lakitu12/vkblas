// test_v9h.c — v9h 2B 直通正确性验证 (经 hipblasGemmEx 公开入口, 与 PyTorch 同路)
// 对比 CPU fp32 参考 (从 2B 位模式精确展开); bf16 容差用相对误差 (固有 0.4%)
// 用法: ./test_v9h   (v9h 为 128-tile 直通默认实现, 无需 env)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

typedef hipblasStatus_t (*gemmex_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                     int, int, int, const void*, const void*, hipblasDatatype_t, int,
                                     const void*, hipblasDatatype_t, int, const void*, void*,
                                     hipblasDatatype_t, int, hipblasDatatype_t, hipblasGemmAlgo_t);
typedef hipblasStatus_t (*create_fn)(hipblasHandle_t*);

static unsigned short f_to_bf16(float f) {
    unsigned u; memcpy(&u, &f, 4);
    return (unsigned short)(((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) & 0xFFFFu);
}
static unsigned short f_to_fp16(float f) {
    // 精确覆盖测试值域 [-2,2) 的 RNE 半精度转换
    unsigned u; memcpy(&u, &f, 4);
    unsigned sign = (u >> 16) & 0x8000u;
    int exp = ((u >> 23) & 0xFF) - 127;
    unsigned man = u & 0x7FFFFF;
    if (((u >> 23) & 0xFF) == 0xFF) return (unsigned short)(sign | 0x7C00u | (man ? 0x200u : 0));
    if (exp > 15) return (unsigned short)(sign | 0x7C00u);
    if (exp < -14) return (unsigned short)sign; // 测试值不会到 denormal
    unsigned half_man = man >> 13;
    unsigned rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half_man & 1))) half_man++;
    if (half_man & 0x400u) { half_man = 0; exp++; }
    return (unsigned short)(sign | (unsigned)((exp + 15) << 10) | half_man);
}
static float bf16_to_f(unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; }
static float fp16_to_f(unsigned short h) {
    unsigned sign = (h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned man = h & 0x3FF;
    unsigned f32;
    if (exp == 0) f32 = sign;
    else if (exp == 31) f32 = sign | 0x7F800000u | (man << 13);
    else f32 = sign | ((unsigned)(exp - 15 + 127) << 23) | (man << 13);
    float f; memcpy(&f, &f32, 4); return f;
}

int main(void) {
    void* so = dlopen("./libvkblas_hipblas.so", RTLD_NOW | RTLD_GLOBAL);
    gemmex_fn vk = (gemmex_fn)dlsym(so, "hipblasGemmEx");
    create_fn cr = (create_fn)dlsym(so, "hipblasCreate");
    if (!vk || !cr) { fprintf(stderr, "dlsym failed\n"); return 2; }
    hipInit(0); hipSetDevice(0);
    hipblasHandle_t h; cr(&h);

    int fails = 0;
    // dtype: 0=bf16(HIPBLAS_R_16B=168), 1=fp16(HIPBLAS_R_16F=150)
    struct { int dtype; hipblasDatatype_t dt; const char* nm; } dts[] = {
        {0, HIPBLAS_R_16B, "bf16"}, {1, HIPBLAS_R_16F, "f16"} };
    // (M,N,K,lda,ldb,ldc, ta,tb): 全部满足 N 偶/ldc 偶 (直通条件);
    // 前两个 pick_tile128=1 走 128 tile (v9h 生效路径), 后两个小 shape 走 v6h 对照
    struct { int m, n, k, lda, ldb, ldc, ta, tb; } cases[] = {
        {256, 256, 256, 256, 256, 256, 0, 1},   // NT (PyTorch linear 形态), 方形 tile
        {130, 258, 70,  74,  72,  260, 0, 1},   // NT 边界: 128+2 行/列跨 tile
        {256, 256, 256, 256, 256, 256, 1, 0},   // TN
        {130, 258, 70,  130, 258, 260, 1, 0},   // TN 边界 (ta=1: lda≥M; tb=0: ldb≥N)
        {64,  64,  48,  48,  64,  64,  0, 1},   // 小 shape (v6h 对照)
    };
    int nc = sizeof(cases) / sizeof(cases[0]);
    for (int d = 0; d < 2; d++) {
        for (int ci = 0; ci < nc; ci++) {
            int M = cases[ci].m, N = cases[ci].n, K = cases[ci].k;
            int lda = cases[ci].lda, ldb = cases[ci].ldb, ldc = cases[ci].ldc;
            int ta = cases[ci].ta, tb = cases[ci].tb;
            size_t sa = (size_t)(ta ? K : M) * lda, sb = (size_t)(tb ? N : K) * ldb;
            unsigned short *hA = malloc(sa * 2), *hB = malloc(sb * 2), *hC = calloc((size_t)M * ldc, 2);
            srand(777 + d * 100 + ci);
            // 值域 [0.5,1) 带符号: fp16 输出累加上限 256×1 = 256 < 65504 不溢出
            // (bf16 版本曾用 [0x3800,0x7000) 值域至 32768 → f16 累加溢出假 FAIL)
            for (size_t i = 0; i < sa; i++) {
                unsigned short v = (unsigned short)(rand() % 0x0400 + 0x3800);
                if (rand() & 1) v |= 0x8000;
                hA[i] = v;
            }
            for (size_t i = 0; i < sb; i++) {
                unsigned short v = (unsigned short)(rand() % 0x0400 + 0x3800);
                if (rand() & 1) v |= 0x8000;
                hB[i] = v;
            }
            void *dA, *dB, *dC;
            hipMalloc(&dA, sa * 2); hipMalloc(&dB, sb * 2);
            hipMalloc(&dC, (size_t)M * ldc * 2);
            hipMemset(dC, 0, (size_t)M * ldc * 2);
            hipMemcpy(dA, hA, sa * 2, hipMemcpyHostToDevice);
            hipMemcpy(dB, hB, sb * 2, hipMemcpyHostToDevice);

            float al = 1.0f, be = 0.0f;
            // 列主序: C(N×M) = op(B)(N×K) · op(A)(K×M) → 行主 C(M×N)=A·B
            vk(h, tb ? HIPBLAS_OP_T : HIPBLAS_OP_N, ta ? HIPBLAS_OP_T : HIPBLAS_OP_N,
               N, M, K, &al, dB, dts[d].dt, ldb, dA, dts[d].dt, lda,
               &be, dC, dts[d].dt, ldc, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
            hipDeviceSynchronize();
            hipMemcpy(hC, dC, (size_t)M * ldc * 2, hipMemcpyDeviceToHost);

            // CPU 参考 (行主): C[i][j] = sum_k A_eff[i][k] * B_eff[k][j], 从位模式精确展开
            double max_rel = 0; int nbad = 0;
            for (int i = 0; i < M && nbad < 5; i++) {
                for (int j = 0; j < N; j++) {
                    float acc = 0;
                    for (int k = 0; k < K; k++) {
                        float a = dts[d].dtype == 0 ? bf16_to_f(hA[(size_t)(ta ? k * lda : i * lda) + (ta ? i : k)])
                                                    : fp16_to_f(hA[(size_t)(ta ? k * lda : i * lda) + (ta ? i : k)]);
                        float b = dts[d].dtype == 0 ? bf16_to_f(hB[(size_t)(tb ? j * ldb : k * ldb) + (tb ? k : j)])
                                                    : fp16_to_f(hB[(size_t)(tb ? j * ldb : k * ldb) + (tb ? k : j)]);
                        acc += a * b;
                    }
                    unsigned short got = hC[(size_t)i * ldc + j];
                    float cf = dts[d].dtype == 0 ? bf16_to_f(got) : fp16_to_f(got);
                    double rel = fabs((double)cf - acc) / (1.0 + fabs(acc));
                    if (rel > max_rel) max_rel = rel;
                    if (fabs(cf - acc) > 0.02 * (1.0 + fabs(acc))) nbad++;
                }
            }
            int ok = nbad == 0 && max_rel < 0.02;
            printf("%s %-4s %dx%dx%d %s%s: max_rel=%.4f nbad=%d\n",
                   ok ? "PASS" : "FAIL", dts[d].nm, M, N, K,
                   ta ? "T" : "N", tb ? "T" : "N", max_rel, nbad);
            if (!ok) fails++;
            hipFree(dA); hipFree(dB); hipFree(dC);
            free(hA); free(hB); free(hC);
        }
    }
    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);

    // ---- batched 冒烟 (合并快路径: dispatch x = Mt*batch, b=wg/Mt 分割) ----
    typedef hipblasStatus_t (*sbex_fn)(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
                                       int, int, int, const void*, const void*, hipblasDatatype_t, int, long long,
                                       const void*, hipblasDatatype_t, int, long long, const void*, void*,
                                       hipblasDatatype_t, int, long long, int, hipblasDatatype_t, hipblasGemmAlgo_t);
    sbex_fn sb = (sbex_fn)dlsym(so, "hipblasGemmStridedBatchedEx_v2");
    if (sb) {
        int B_ = 4, M2 = 130, N2 = 258, K2 = 70, ldc2 = 260;
        long long sa2 = (long long)M2*K2, sb2 = (long long)K2*N2;
        unsigned short *hA2 = malloc(sa2*B_*2), *hB2 = malloc(sb2*2), *hC2 = calloc((size_t)M2*ldc2*B_, 2);
        srand(555);
        for (long i = 0; i < sa2*B_; i++) { unsigned short v = rand()%0x400+0x3800; if (rand()&1) v|=0x8000; hA2[i]=v; }
        for (long i = 0; i < sb2; i++)   { unsigned short v = rand()%0x400+0x3800; if (rand()&1) v|=0x8000; hB2[i]=v; }
        void *dA2,*dB2,*dC2;
        hipMalloc(&dA2,(size_t)sa2*B_*2); hipMalloc(&dB2,(size_t)sb2*2); hipMalloc(&dC2,(size_t)M2*ldc2*B_*2);
        hipMemset(dC2,0,(size_t)M2*ldc2*B_*2);
        hipMemcpy(dA2,hA2,(size_t)sa2*B_*2,hipMemcpyHostToDevice);
        hipMemcpy(dB2,hB2,(size_t)sb2*2,hipMemcpyHostToDevice);
        float al=1.0f, be=0.0f;
        // 行主 C_b(M×N)=A_b·B²: 列主翻译 NT
        sb(h, HIPBLAS_OP_N, HIPBLAS_OP_T, N2, M2, K2, &al, dB2, HIPBLAS_R_16B, N2, sb2,
           dA2, HIPBLAS_R_16B, K2, sa2, &be, dC2, HIPBLAS_R_16B, ldc2, (long long)M2*ldc2, B_,
           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        hipDeviceSynchronize();
        unsigned short *hC3 = malloc((size_t)M2*ldc2*B_*2);
        hipMemcpy(hC3,dC2,(size_t)M2*ldc2*B_*2,hipMemcpyDeviceToHost);
        int nbad = 0;
        for (int bt2 = 0; bt2 < B_ && nbad < 5; bt2++)
            for (int i = 0; i < M2 && nbad < 5; i++)
                for (int j = 0; j < N2 && nbad < 5; j++) {
                    float acc = 0;
                    for (int k = 0; k < K2; k++)
                        acc += bf16_to_f(hA2[(size_t)bt2*sa2+i*K2+k]) * bf16_to_f(hB2[(size_t)k*N2+j]);
                    float cf = bf16_to_f(hC3[((size_t)bt2*M2+i)*ldc2+j]);
                    if (fabs(cf-acc) > 0.02*(1.0+fabs(acc))) nbad++;
                }
        printf("%s bf16 batched NT b=%d %dx%dx%d: nbad=%d\n", nbad?"FAIL":"PASS", B_, M2,N2,K2, nbad);
        if (nbad) fails++;
        hipFree(dA2); hipFree(dB2); hipFree(dC2);
        free(hA2); free(hB2); free(hC2); free(hC3);
    }
    return fails == 0 ? 0 : 1;
}
