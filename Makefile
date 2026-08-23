# vkblas — Vulkan 实现的 BLAS (hipBLAS ABI 兼容层)
ROCM ?= /opt/rocm-6.4.3
CC ?= gcc
CFLAGS = -O2 -Wall -Wextra -fPIC -D__HIP_PLATFORM_AMD__ -I$(ROCM)/include -I src
LDFLAGS = -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib
SHADER_DIR = src/shaders

SHADERS = $(SHADER_DIR)/gemm_nn.spv $(SHADER_DIR)/gemm_tn.spv \
          $(SHADER_DIR)/gemm_nt.spv $(SHADER_DIR)/gemm_tt.spv \
          $(SHADER_DIR)/matvec_n.spv $(SHADER_DIR)/matvec_t.spv \
          $(SHADER_DIR)/matvec_sk_n.spv $(SHADER_DIR)/matvec_sk_t.spv \
          $(SHADER_DIR)/matvec_sk_h16.spv $(SHADER_DIR)/matvec_sk_bf16.spv \
          $(SHADER_DIR)/split_k_reduce_h16.spv $(SHADER_DIR)/split_k_reduce_bf16.spv \
          $(SHADER_DIR)/gemm128_nn.spv $(SHADER_DIR)/gemm128_tn.spv \
          $(SHADER_DIR)/gemm128_nt.spv $(SHADER_DIR)/gemm128_tt.spv \
          $(SHADER_DIR)/gemm_h16_nn.spv $(SHADER_DIR)/gemm_h16_tn.spv \
          $(SHADER_DIR)/gemm_h16_nt.spv $(SHADER_DIR)/gemm_h16_tt.spv \
          $(SHADER_DIR)/gemm_b16_nn.spv $(SHADER_DIR)/gemm_b16_tn.spv \
          $(SHADER_DIR)/gemm_b16_nt.spv $(SHADER_DIR)/gemm_b16_tt.spv \
          $(SHADER_DIR)/gemm128_h16_nn.spv $(SHADER_DIR)/gemm128_h16_tn.spv \
          $(SHADER_DIR)/gemm128_h16_nt.spv $(SHADER_DIR)/gemm128_h16_tt.spv \
          $(SHADER_DIR)/gemm128_b16_nn.spv $(SHADER_DIR)/gemm128_b16_tn.spv \
          $(SHADER_DIR)/gemm128_b16_nt.spv $(SHADER_DIR)/gemm128_b16_tt.spv \
          $(SHADER_DIR)/transpose_h16.spv $(SHADER_DIR)/transpose_b16.spv \
          $(SHADER_DIR)/gemm_sk_nn.spv $(SHADER_DIR)/gemm_sk_tn.spv \
          $(SHADER_DIR)/gemm_sk_nt.spv $(SHADER_DIR)/gemm_sk_tt.spv \
          $(SHADER_DIR)/gemm_sk128_nn.spv $(SHADER_DIR)/gemm_sk128_tn.spv \
          $(SHADER_DIR)/gemm_sk128_nt.spv $(SHADER_DIR)/gemm_sk128_tt.spv \
          $(SHADER_DIR)/split_k_reduce.spv \
          $(SHADER_DIR)/transpose.spv \
          $(SHADER_DIR)/cvt_b2f.spv $(SHADER_DIR)/cvt_b2f_tsp.spv \
          $(SHADER_DIR)/cvt_f2b.spv $(SHADER_DIR)/cvt_f2b_atomic.spv \
          $(SHADER_DIR)/cvt_h2f.spv \
          $(SHADER_DIR)/cvt_f2h.spv $(SHADER_DIR)/cvt_f2h_atomic.spv \
          $(SHADER_DIR)/cvt_cx_planar.spv $(SHADER_DIR)/cvt_cx_inter.spv \
          $(SHADER_DIR)/cx_combine.spv \
          $(SHADER_DIR)/gemm_d64_nn.spv $(SHADER_DIR)/gemm_d64_tn.spv \
          $(SHADER_DIR)/gemm_d64_nt.spv $(SHADER_DIR)/gemm_d64_tt.spv \
          $(SHADER_DIR)/transpose_d64.spv \
          $(SHADER_DIR)/cvt_cz_planar.spv $(SHADER_DIR)/cx_combine_d64.spv \

all: libvkblas_hipblas.so test/test_gemm test/test_h

# --- shader 4 变体 (TA/TB = A/B 因子是否转置读) ---
$(SHADER_DIR)/gemm_nn.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_tn.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_nt.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_tt.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- matvec (M==1 decode, 免 B 转置) ---
$(SHADER_DIR)/matvec_n.spv: $(SHADER_DIR)/matvec_tmpl.comp
	glslangValidator -V -DTB=0 $< -o $@
$(SHADER_DIR)/matvec_t.spv: $(SHADER_DIR)/matvec_tmpl.comp
	glslangValidator -V -DTB=1 $< -o $@
$(SHADER_DIR)/matvec_sk_n.spv: $(SHADER_DIR)/matvec_sk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=0 $< -o $@
$(SHADER_DIR)/matvec_sk_t.spv: $(SHADER_DIR)/matvec_sk_tmpl.comp
	glslangValidator -V -DTB=1 -DHALF=0 $< -o $@
$(SHADER_DIR)/matvec_sk_h16.spv: $(SHADER_DIR)/matvec_sk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=1 $< -o $@
$(SHADER_DIR)/matvec_sk_bf16.spv: $(SHADER_DIR)/matvec_sk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=2 $< -o $@
$(SHADER_DIR)/split_k_reduce_h16.spv: $(SHADER_DIR)/split_k_reduce_h.comp
	glslangValidator -V -DHALF=1 $< -o $@
$(SHADER_DIR)/split_k_reduce_bf16.spv: $(SHADER_DIR)/split_k_reduce_h.comp
	glslangValidator -V -DHALF=2 $< -o $@
$(SHADER_DIR)/transpose.spv: $(SHADER_DIR)/transpose.comp
	glslangValidator -V $< -o $@
# --- v7-128 tile (llama.cpp l_warptile 移植: 128×128, BK16, 32 acc/线程) ---
$(SHADER_DIR)/gemm128_nn.spv: $(SHADER_DIR)/gemm_tmpl_128.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_tn.spv: $(SHADER_DIR)/gemm_tmpl_128.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_nt.spv: $(SHADER_DIR)/gemm_tmpl_128.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm128_tt.spv: $(SHADER_DIR)/gemm_tmpl_128.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- split-k (llama.cpp 借鉴: K 分段并行 + reduce 归约) ---
$(SHADER_DIR)/gemm_sk_nn.spv: $(SHADER_DIR)/gemm_sk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_sk_tn.spv: $(SHADER_DIR)/gemm_sk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_sk_nt.spv: $(SHADER_DIR)/gemm_sk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_sk_tt.spv: $(SHADER_DIR)/gemm_sk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_sk128_nn.spv: $(SHADER_DIR)/gemm_sk_128_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_sk128_tn.spv: $(SHADER_DIR)/gemm_sk_128_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_sk128_nt.spv: $(SHADER_DIR)/gemm_sk_128_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_sk128_tt.spv: $(SHADER_DIR)/gemm_sk_128_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/split_k_reduce.spv: $(SHADER_DIR)/split_k_reduce.comp
	glslangValidator -V $< -o $@
# --- f16/bf16 直通 (HALF_TYPE=1 fp16 / 2 bf16): v6 与 v7-128, 4 变体 + 转置 ---
$(SHADER_DIR)/gemm_h16_nn.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_h16_tn.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_h16_nt.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_h16_tt.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_b16_nn.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_b16_tn.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_b16_nt.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_b16_tt.spv: $(SHADER_DIR)/gemm_tmpl_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm128_h16_nn.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_h16_tn.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_h16_nt.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm128_h16_tt.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm128_b16_nn.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_b16_tn.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm128_b16_nt.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm128_b16_tt.spv: $(SHADER_DIR)/gemm_tmpl_128_h.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/transpose_h16.spv: $(SHADER_DIR)/transpose_h.comp
	glslangValidator -V -DHALF_TYPE=1 $< -o $@
$(SHADER_DIR)/transpose_b16.spv: $(SHADER_DIR)/transpose_h.comp
	glslangValidator -V -DHALF_TYPE=2 $< -o $@
$(SHADER_DIR)/cvt_b2f.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=1 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_b2f_tsp.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=1 -DCVT_TSP=1 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2b.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=0 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2b_atomic.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=0 -DCVT_TSP=0 -DCVT_ATOMIC=1 $< -o $@
$(SHADER_DIR)/cvt_h2f.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_H2F=1 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2h.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_H2F=0 -DCVT_F2H=1 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2h_atomic.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_H2F=0 -DCVT_F2H=1 -DCVT_TSP=0 -DCVT_ATOMIC=1 $< -o $@
$(SHADER_DIR)/cvt_cx_planar.spv: $(SHADER_DIR)/cvt_cx.comp
	glslangValidator -V -DCX_PLANAR=1 $< -o $@
$(SHADER_DIR)/cvt_cx_inter.spv: $(SHADER_DIR)/cvt_cx.comp
	glslangValidator -V -DCX_PLANAR=0 $< -o $@
$(SHADER_DIR)/cx_combine.spv: $(SHADER_DIR)/cx_combine.comp
	glslangValidator -V $< -o $@
$(SHADER_DIR)/cvt_cz_planar.spv: $(SHADER_DIR)/cvt_cz.comp
	glslangValidator -V $< -o $@
$(SHADER_DIR)/cx_combine_d64.spv: $(SHADER_DIR)/cx_combine_d64.comp
	glslangValidator -V $< -o $@
$(SHADER_DIR)/gemm_d64_nn.spv: $(SHADER_DIR)/gemm_d64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_d64_tn.spv: $(SHADER_DIR)/gemm_d64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_d64_nt.spv: $(SHADER_DIR)/gemm_d64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_d64_tt.spv: $(SHADER_DIR)/gemm_d64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/transpose_d64.spv: $(SHADER_DIR)/transpose_d64.comp
	glslangValidator -V $< -o $@
	glslangValidator -V $< -o $@
	glslangValidator -V $< -o $@

# --- LD_PRELOAD 兼容层 ---
libvkblas_hipblas.so: src/vkblas.c src/vkblas_hipblas.c src/vkblas.h $(SHADERS)
	$(CC) $(CFLAGS) -shared -o $@ src/vkblas.c src/vkblas_hipblas.c \
	    -ldl -lpthread -lvulkan -lamdhip64 $(LDFLAGS)

# --- 正确性/性能测试 (dlopen 我们的 .so + 直链真 hipblas) ---
test/test_gemm: test/test_gemm.c src/vkblas.h libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/test_gemm.c -lhipblas -lamdhip64 $(LDFLAGS)

# f16/bf16 直通回归 (引擎层直连, 4 变体 × 7 shape × 2 dtype + padding ld)
test/test_h: test/test_h.c src/vkblas.h libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/test_h.c -lamdhip64 $(LDFLAGS)

# v6/v7 tile 对比扫描 (同一进程内切 VKBLAS_TILE128, CPU 参考校验 + best-of-3)
test/bench_shapes: test/bench_shapes.c libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/bench_shapes.c -lhipblas -lamdhip64 $(LDFLAGS)

# import 缓存 (hit 路径) 回归 — 必须 LD_PRELOAD 运行 (hook 自证实才启用缓存):
#   LD_PRELOAD=$PWD/libvkblas_hipblas.so ./test/test_cache
test/test_cache: test/test_cache.c src/vkblas.h libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/test_cache.c -lhipblas -lamdhip64 $(LDFLAGS)

clean:
	rm -f libvkblas_hipblas.so test/test_gemm test/test_h test/test_cache $(SHADERS)

# --- 部署到本机 ROCm (/opt/rocm-6.4.3/lib) ---
# 安装版 .so 用 -DVKBLAS_SHADER_DIR 指向固定 shader 目录 (自包含, 不依赖源码树);
# shaders 目录可被 VKBLAS_SHADER_DIR 环境变量覆盖
SHADER_INSTALL ?= /opt/rocm-6.4.3/lib/vkblas-shaders
install: libvkblas_hipblas.so
	mkdir -p $(SHADER_INSTALL)
	cp src/shaders/*.spv $(SHADER_INSTALL)/
	$(CC) $(CFLAGS) -DVKBLAS_SHADER_DIR=\"$(SHADER_INSTALL)\" -shared -o /opt/rocm-6.4.3/lib/libvkblas_hipblas.so \
	    src/vkblas.c src/vkblas_hipblas.c -ldl -lpthread -lvulkan -lamdhip64 $(LDFLAGS)
	@echo "installed: /opt/rocm-6.4.3/lib/libvkblas_hipblas.so (shaders -> $(SHADER_INSTALL))"

.PHONY: all clean install
