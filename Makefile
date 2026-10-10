# vkblas — Vulkan 实现的 BLAS (hipBLAS ABI 兼容层)
ROCM ?= /opt/rocm
CC ?= gcc
CFLAGS = -O2 -Wall -Wextra -fPIC -D__HIP_PLATFORM_AMD__ -I$(ROCM)/include -I src
LDFLAGS = -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib
SHADER_DIR = src/shaders

SHADERS = $(SHADER_DIR)/gemm_f32_64x64_nn.spv $(SHADER_DIR)/gemm_f32_64x64_tn.spv \
          $(SHADER_DIR)/gemm_f32_64x64_nt.spv $(SHADER_DIR)/gemm_f32_64x64_tt.spv \
          $(SHADER_DIR)/matvec_f32_n.spv $(SHADER_DIR)/matvec_f32_t.spv \
          $(SHADER_DIR)/matvec_f32_splitk_n.spv $(SHADER_DIR)/matvec_f32_splitk_t.spv \
          $(SHADER_DIR)/matvec_f16_splitk.spv $(SHADER_DIR)/matvec_bf16_splitk.spv \
          $(SHADER_DIR)/matvec_f16_splitk_m.spv $(SHADER_DIR)/matvec_bf16_splitk_m.spv \
          $(SHADER_DIR)/splitk_reduce_m_f16.spv $(SHADER_DIR)/splitk_reduce_m_bf16.spv \
          $(SHADER_DIR)/splitk_reduce_f16.spv $(SHADER_DIR)/splitk_reduce_bf16.spv \
          $(SHADER_DIR)/gemm_f32_128x128_bankconflict_nn.spv $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tn.spv \
          $(SHADER_DIR)/gemm_f32_128x128_bankconflict_nt.spv $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tt.spv \
          $(SHADER_DIR)/gemm_f32_128x64_nn.spv $(SHADER_DIR)/gemm_f32_128x64_tn.spv \
          $(SHADER_DIR)/gemm_f32_128x64_nt.spv $(SHADER_DIR)/gemm_f32_128x64_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x64_nn.spv $(SHADER_DIR)/gemm_f16_128x64_tn.spv \
          $(SHADER_DIR)/gemm_f16_128x64_nt.spv $(SHADER_DIR)/gemm_f16_128x64_tt.spv \
          $(SHADER_DIR)/gemm_f32_128x128_bankfree_nn.spv $(SHADER_DIR)/gemm_f32_128x128_bankfree_tn.spv \
          $(SHADER_DIR)/gemm_f32_128x128_bankfree_nt.spv $(SHADER_DIR)/gemm_f32_128x128_bankfree_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_nn.spv $(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_tn.spv \
          $(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_nt.spv $(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_nn.spv $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_tn.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_nt.spv $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x64_nn.spv $(SHADER_DIR)/gemm_bf16_128x64_tn.spv \
          $(SHADER_DIR)/gemm_bf16_128x64_nt.spv $(SHADER_DIR)/gemm_bf16_128x64_tt.spv \
          $(SHADER_DIR)/gemm_f16_64x64_nn.spv $(SHADER_DIR)/gemm_f16_64x64_tn.spv \
          $(SHADER_DIR)/gemm_f16_64x64_nt.spv $(SHADER_DIR)/gemm_f16_64x64_tt.spv \
          $(SHADER_DIR)/gemm_bf16_64x64_nn.spv $(SHADER_DIR)/gemm_bf16_64x64_tn.spv \
          $(SHADER_DIR)/gemm_bf16_64x64_nt.spv $(SHADER_DIR)/gemm_bf16_64x64_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x128_bankconflict_nn.spv $(SHADER_DIR)/gemm_f16_128x128_bankconflict_tn.spv \
          $(SHADER_DIR)/gemm_f16_128x128_bankconflict_nt.spv $(SHADER_DIR)/gemm_f16_128x128_bankconflict_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_bankconflict_nn.spv $(SHADER_DIR)/gemm_bf16_128x128_bankconflict_tn.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_bankconflict_nt.spv $(SHADER_DIR)/gemm_bf16_128x128_bankconflict_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x128_bankfree_nn.spv $(SHADER_DIR)/gemm_f16_128x128_bankfree_tn.spv \
          $(SHADER_DIR)/gemm_f16_128x128_bankfree_nt.spv $(SHADER_DIR)/gemm_f16_128x128_bankfree_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_bankfree_nn.spv $(SHADER_DIR)/gemm_bf16_128x128_bankfree_tn.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_bankfree_nt.spv $(SHADER_DIR)/gemm_bf16_128x128_bankfree_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x256_bankfree_nn.spv $(SHADER_DIR)/gemm_f16_128x256_bankfree_nt.spv \
          $(SHADER_DIR)/gemm_f16_128x256_bankfree_tn.spv $(SHADER_DIR)/gemm_f16_128x256_bankfree_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x256_bankfree_nn.spv $(SHADER_DIR)/gemm_bf16_128x256_bankfree_nt.spv \
          $(SHADER_DIR)/gemm_bf16_128x256_bankfree_tn.spv $(SHADER_DIR)/gemm_bf16_128x256_bankfree_tt.spv \
          $(SHADER_DIR)/gemm_f16_128x128_splitk_nn.spv $(SHADER_DIR)/gemm_f16_128x128_splitk_nt.spv \
          $(SHADER_DIR)/gemm_f16_128x128_splitk_tn.spv $(SHADER_DIR)/gemm_f16_128x128_splitk_tt.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_splitk_nn.spv $(SHADER_DIR)/gemm_bf16_128x128_splitk_nt.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_splitk_tn.spv $(SHADER_DIR)/gemm_bf16_128x128_splitk_tt.spv \
          $(SHADER_DIR)/splitk_reduce_hgemm_f16.spv $(SHADER_DIR)/splitk_reduce_hgemm_bf16.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_nn.spv $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_tn.spv \
          $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_nt.spv $(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_tt.spv \
          $(SHADER_DIR)/transpose_f16.spv $(SHADER_DIR)/transpose_bf16.spv \
          $(SHADER_DIR)/gemm_f32_64x64_splitk_nn.spv $(SHADER_DIR)/gemm_f32_64x64_splitk_tn.spv \
          $(SHADER_DIR)/gemm_f32_64x64_splitk_nt.spv $(SHADER_DIR)/gemm_f32_64x64_splitk_tt.spv \
          $(SHADER_DIR)/gemm_f32_128x128_splitk_nn.spv $(SHADER_DIR)/gemm_f32_128x128_splitk_tn.spv \
          $(SHADER_DIR)/gemm_f32_128x128_splitk_nt.spv $(SHADER_DIR)/gemm_f32_128x128_splitk_tt.spv \
          $(SHADER_DIR)/splitk_reduce_f32.spv \
          $(SHADER_DIR)/transpose_f32.spv \
          $(SHADER_DIR)/cvt_b2f.spv $(SHADER_DIR)/cvt_b2f_tsp.spv \
          $(SHADER_DIR)/cvt_f2b.spv $(SHADER_DIR)/cvt_f2b_atomic.spv \
          $(SHADER_DIR)/cvt_h2f.spv \
          $(SHADER_DIR)/cvt_f2h.spv $(SHADER_DIR)/cvt_f2h_atomic.spv \
          $(SHADER_DIR)/cvt_cx_planar.spv $(SHADER_DIR)/cvt_cx_inter.spv \
          $(SHADER_DIR)/cx_combine.spv \
          $(SHADER_DIR)/gemm_f64_32x32_nn.spv $(SHADER_DIR)/gemm_f64_32x32_tn.spv \
          $(SHADER_DIR)/gemm_f64_32x32_nt.spv $(SHADER_DIR)/gemm_f64_32x32_tt.spv \
          $(SHADER_DIR)/transpose_f64.spv \
          $(SHADER_DIR)/cvt_cz_planar.spv $(SHADER_DIR)/cx_combine_d64.spv \

all: libvkblas_hipblas.so test/test_gemm test/test_h test/test_ic_cache

# --- shader 4 变体 (TA/TB = A/B 因子是否转置读) ---
$(SHADER_DIR)/gemm_f32_64x64_nn.spv: $(SHADER_DIR)/gemm_f32_64x64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_tn.spv: $(SHADER_DIR)/gemm_f32_64x64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_nt.spv: $(SHADER_DIR)/gemm_f32_64x64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_tt.spv: $(SHADER_DIR)/gemm_f32_64x64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- matvec (M==1 decode, 免 B 转置) ---
$(SHADER_DIR)/matvec_f32_n.spv: $(SHADER_DIR)/matvec_f32_tmpl.comp
	glslangValidator -V -DTB=0 $< -o $@
$(SHADER_DIR)/matvec_f32_t.spv: $(SHADER_DIR)/matvec_f32_tmpl.comp
	glslangValidator -V -DTB=1 $< -o $@
$(SHADER_DIR)/matvec_f32_splitk_n.spv: $(SHADER_DIR)/matvec_splitk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=0 $< -o $@
$(SHADER_DIR)/matvec_f32_splitk_t.spv: $(SHADER_DIR)/matvec_splitk_tmpl.comp
	glslangValidator -V -DTB=1 -DHALF=0 $< -o $@
$(SHADER_DIR)/matvec_f16_splitk.spv: $(SHADER_DIR)/matvec_splitk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=1 $< -o $@
$(SHADER_DIR)/matvec_bf16_splitk.spv: $(SHADER_DIR)/matvec_splitk_tmpl.comp
	glslangValidator -V -DTB=0 -DHALF=2 $< -o $@
# --- matvec_sk_m (M 行 decode split-k) ---
$(SHADER_DIR)/matvec_f16_splitk_m.spv: $(SHADER_DIR)/matvec_splitk_m_tmpl.comp
	glslangValidator -V -DHALF=1 $< -o $@
$(SHADER_DIR)/matvec_bf16_splitk_m.spv: $(SHADER_DIR)/matvec_splitk_m_tmpl.comp
	glslangValidator -V -DHALF=2 $< -o $@
$(SHADER_DIR)/splitk_reduce_m_f16.spv: $(SHADER_DIR)/splitk_reduce_m_half.comp
	glslangValidator -V -DHALF=1 $< -o $@
$(SHADER_DIR)/splitk_reduce_m_bf16.spv: $(SHADER_DIR)/splitk_reduce_m_half.comp
	glslangValidator -V -DHALF=2 $< -o $@
$(SHADER_DIR)/splitk_reduce_f16.spv: $(SHADER_DIR)/splitk_reduce_half.comp
	glslangValidator -V -DHALF=1 $< -o $@
$(SHADER_DIR)/splitk_reduce_bf16.spv: $(SHADER_DIR)/splitk_reduce_half.comp
	glslangValidator -V -DHALF=2 $< -o $@
$(SHADER_DIR)/transpose_f32.spv: $(SHADER_DIR)/transpose_f32.comp
	glslangValidator -V $< -o $@
# --- f32 128x128 bankconflict (原 v7-128; llama.cpp l_warptile 移植: 128×128, BK16) ---
$(SHADER_DIR)/gemm_f32_128x128_bankconflict_nn.spv: $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankconflict_tn.spv: $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankconflict_nt.spv: $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankconflict_tt.spv: $(SHADER_DIR)/gemm_f32_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- gemm 128×64 tile (32 acc/线程; 实验保留, 默认不选) ---
$(SHADER_DIR)/gemm_f32_128x64_nn.spv: $(SHADER_DIR)/gemm_f32_128x64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x64_nt.spv: $(SHADER_DIR)/gemm_f32_128x64_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_128x64_tn.spv: $(SHADER_DIR)/gemm_f32_128x64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x64_tt.spv: $(SHADER_DIR)/gemm_f32_128x64_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- f32 128x128 bankfree (64 acc + 无冲突 LDS 读; fp32 128-tile 默认; 原 v9) ---
$(SHADER_DIR)/gemm_f32_128x128_bankfree_nn.spv: $(SHADER_DIR)/gemm_f32_128x128_bankfree_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankfree_nt.spv: $(SHADER_DIR)/gemm_f32_128x128_bankfree_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankfree_tn.spv: $(SHADER_DIR)/gemm_f32_128x128_bankfree_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_bankfree_tt.spv: $(SHADER_DIR)/gemm_f32_128x128_bankfree_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
# --- half 128x64 2B 直通 (HALF_TYPE=1 fp16 / 2 bf16; 原 v8h) ---
$(SHADER_DIR)/gemm_f16_128x64_nn.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x64_nt.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x64_tn.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x64_tt.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x64_nn.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x64_nt.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x64_tn.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x64_tt.spv: $(SHADER_DIR)/gemm_half_128x64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
# --- half 128x128 2B LDS 直通-真打包 (HALF_TYPE=1 fp16 / 2 bf16; 原 v10h) ---
$(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_nn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_nt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_tn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_lds_packed_b128_tt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_nn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_nt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_tn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b128_tt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b128_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
# --- split-k (llama.cpp 借鉴: K 分段并行 + reduce 归约) ---
$(SHADER_DIR)/gemm_f32_64x64_splitk_nn.spv: $(SHADER_DIR)/gemm_f32_64x64_splitk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_splitk_tn.spv: $(SHADER_DIR)/gemm_f32_64x64_splitk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_splitk_nt.spv: $(SHADER_DIR)/gemm_f32_64x64_splitk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_64x64_splitk_tt.spv: $(SHADER_DIR)/gemm_f32_64x64_splitk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_splitk_nn.spv: $(SHADER_DIR)/gemm_f32_128x128_splitk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_splitk_tn.spv: $(SHADER_DIR)/gemm_f32_128x128_splitk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_splitk_nt.spv: $(SHADER_DIR)/gemm_f32_128x128_splitk_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f32_128x128_splitk_tt.spv: $(SHADER_DIR)/gemm_f32_128x128_splitk_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/splitk_reduce_f32.spv: $(SHADER_DIR)/splitk_reduce_f32.comp
	glslangValidator -V $< -o $@
# --- f16/bf16 直通 (HALF_TYPE=1 fp16 / 2 bf16): 64x64 与 128x128 bankconflict, 4 变体 + 转置 ---
$(SHADER_DIR)/gemm_f16_64x64_nn.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_64x64_tn.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_64x64_nt.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_64x64_tt.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_64x64_nn.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_64x64_tn.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_64x64_nt.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_64x64_tt.spv: $(SHADER_DIR)/gemm_half_64x64_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankconflict_nn.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankconflict_tn.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankconflict_nt.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankconflict_tt.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankconflict_nn.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankconflict_tn.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankconflict_nt.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankconflict_tt.spv: $(SHADER_DIR)/gemm_half_128x128_bankconflict_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
# --- half 128x128 bankfree 2B 直通 (无冲突主循环; f16/bf16 128-tile 默认; 原 v9h) ---
$(SHADER_DIR)/gemm_f16_128x128_bankfree_nn.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankfree_nt.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankfree_tn.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_bankfree_tt.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankfree_nn.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankfree_nt.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankfree_tn.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_bankfree_tt.spv: $(SHADER_DIR)/gemm_half_128x128_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
# --- half 128x256 bankfree 高 acc 密度实验变体 (每线程 8x16=128 acc; VKBLAS_HT256 实验门) ---
$(SHADER_DIR)/gemm_f16_128x256_bankfree_nn.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x256_bankfree_nt.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x256_bankfree_tn.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x256_bankfree_tt.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x256_bankfree_nn.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x256_bankfree_nt.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x256_bankfree_tn.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x256_bankfree_tt.spv: $(SHADER_DIR)/gemm_half_128x256_bankfree_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_splitk_nn.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_splitk_nt.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_splitk_tn.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f16_128x128_splitk_tt.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=1 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_splitk_nn.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_splitk_nt.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_splitk_tn.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_splitk_tt.spv: $(SHADER_DIR)/gemm_half_128x128_splitk_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/splitk_reduce_hgemm_f16.spv: $(SHADER_DIR)/splitk_reduce_hgemm.comp
	glslangValidator -V -DHALF=1 $< -o $@
$(SHADER_DIR)/splitk_reduce_hgemm_bf16.spv: $(SHADER_DIR)/splitk_reduce_hgemm.comp
	glslangValidator -V -DHALF=2 $< -o $@
# --- half 128x128 lds_packed_b64 2B 打包 LDS 主循环 (原 v9hp; b64 无冲突 + 位模式解包; lds_packed_b128 的 b128 冲突版已证伪) ---
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_nn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_nt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_tn.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_bf16_128x128_lds_packed_b64_tt.spv: $(SHADER_DIR)/gemm_half_128x128_lds_packed_b64_tmpl.comp
	glslangValidator -V -DHALF_TYPE=2 -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/transpose_f16.spv: $(SHADER_DIR)/transpose_half.comp
	glslangValidator -V -DHALF_TYPE=1 $< -o $@
$(SHADER_DIR)/transpose_bf16.spv: $(SHADER_DIR)/transpose_half.comp
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
$(SHADER_DIR)/gemm_f64_32x32_nn.spv: $(SHADER_DIR)/gemm_f64_32x32_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f64_32x32_tn.spv: $(SHADER_DIR)/gemm_f64_32x32_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=0 $< -o $@
$(SHADER_DIR)/gemm_f64_32x32_nt.spv: $(SHADER_DIR)/gemm_f64_32x32_tmpl.comp
	glslangValidator -V -DTA=0 -DTB=1 $< -o $@
$(SHADER_DIR)/gemm_f64_32x32_tt.spv: $(SHADER_DIR)/gemm_f64_32x32_tmpl.comp
	glslangValidator -V -DTA=1 -DTB=1 $< -o $@
$(SHADER_DIR)/transpose_f64.spv: $(SHADER_DIR)/transpose_f64.comp
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

# 64x64/128x128 tile 对比扫描 (同一进程内切 VKBLAS_TILE128, CPU 参考校验 + best-of-3)
test/bench_shapes: test/bench_shapes.c libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/bench_shapes.c -lhipblas -lamdhip64 $(LDFLAGS)

# import 缓存 (hit 路径) 回归 — 必须 LD_PRELOAD 运行 (hook 自证实才启用缓存):
#   LD_PRELOAD=$PWD/libvkblas_hipblas.so ./test/test_cache
test/test_cache: test/test_cache.c src/vkblas.h libvkblas_hipblas.so
	$(CC) $(CFLAGS) -o $@ test/test_cache.c -lhipblas -lamdhip64 $(LDFLAGS)

# import 缓存表核心 host 单测 (零 GPU/驱动依赖, 不需 LD_PRELOAD): ./test/test_ic_cache
test/test_ic_cache: test/test_ic_cache.c src/ic_cache.h
	$(CC) $(CFLAGS) -o $@ test/test_ic_cache.c

clean:
	rm -f libvkblas_hipblas.so test/test_gemm test/test_h test/test_cache test/test_ic_cache $(SHADERS)

# --- 部署到本机 ROCm (/opt/rocm/lib) ---
# 安装版 .so 用 -DVKBLAS_SHADER_DIR 指向固定 shader 目录 (自包含, 不依赖源码树);
# shaders 目录可被 VKBLAS_SHADER_DIR 环境变量覆盖
SHADER_INSTALL ?= /opt/rocm/lib/vkblas-shaders
install: libvkblas_hipblas.so
	mkdir -p $(SHADER_INSTALL)
	cp src/shaders/*.spv $(SHADER_INSTALL)/
	$(CC) $(CFLAGS) -DVKBLAS_SHADER_DIR=\"$(SHADER_INSTALL)\" -shared -o /opt/rocm/lib/libvkblas_hipblas.so \
	    src/vkblas.c src/vkblas_hipblas.c -ldl -lpthread -lvulkan -lamdhip64 $(LDFLAGS)
	@echo "installed: /opt/rocm/lib/libvkblas_hipblas.so (shaders -> $(SHADER_INSTALL))"

.PHONY: all clean install
