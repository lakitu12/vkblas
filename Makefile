# vkblas — Vulkan 实现的 BLAS (hipBLAS ABI 兼容层)
ROCM ?= /opt/rocm-6.4.3
CC ?= gcc
CFLAGS = -O2 -Wall -Wextra -fPIC -D__HIP_PLATFORM_AMD__ -I$(ROCM)/include -I src
LDFLAGS = -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib
SHADER_DIR = src/shaders

SHADERS = $(SHADER_DIR)/gemm_nn.spv $(SHADER_DIR)/gemm_tn.spv \
          $(SHADER_DIR)/gemm_nt.spv $(SHADER_DIR)/gemm_tt.spv \
          $(SHADER_DIR)/transpose.spv \
          $(SHADER_DIR)/cvt_b2f.spv $(SHADER_DIR)/cvt_b2f_tsp.spv \
          $(SHADER_DIR)/cvt_f2b.spv $(SHADER_DIR)/cvt_f2b_atomic.spv \
          $(SHADER_DIR)/cvt_cx_planar.spv $(SHADER_DIR)/cvt_cx_inter.spv \
          $(SHADER_DIR)/cx_combine.spv \
          $(SHADER_DIR)/gemm_d64_nn.spv $(SHADER_DIR)/gemm_d64_tn.spv \
          $(SHADER_DIR)/gemm_d64_nt.spv $(SHADER_DIR)/gemm_d64_tt.spv \
          $(SHADER_DIR)/transpose_d64.spv \
          $(SHADER_DIR)/cvt_cz_planar.spv $(SHADER_DIR)/cx_combine_d64.spv \

all: libvkblas_hipblas.so test/test_gemm

# --- shader 4 变体 (TA/TB = A/B 因子是否转置读) ---
$(SHADER_DIR)/gemm_nn.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_tn.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_nt.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_tt.spv: $(SHADER_DIR)/gemm_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/transpose.spv: $(SHADER_DIR)/transpose.comp
	glslangValidator -V $< -o $@
$(SHADER_DIR)/cvt_b2f.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=1 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_b2f_tsp.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=1 -DCVT_TSP=1 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2b.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=0 -DCVT_TSP=0 -DCVT_ATOMIC=0 $< -o $@
$(SHADER_DIR)/cvt_f2b_atomic.spv: $(SHADER_DIR)/cvt_tmpl.comp
	glslangValidator -V -DCVT_B2F=0 -DCVT_TSP=0 -DCVT_ATOMIC=1 $< -o $@
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

clean:
	rm -f libvkblas_hipblas.so test/test_gemm $(SHADERS)

.PHONY: all clean
