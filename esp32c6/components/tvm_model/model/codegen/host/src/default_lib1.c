// tvm target: c -keys=cpu -model=esp32
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_layout_transform_reshape(float* p0, float* T_reshape, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 120; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
      T_reshape[cse_var_1] = p0[((((ax1_outer / 15) * 240) + ((cse_var_1 % 60) * 4)) + ((((ax1_outer % 15) * 4) + (ax1_inner >> 2)) / 15))];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_let = (&(global_const_workspace_4_var[3949360]));
  void* fused_constant_let = (&(global_const_workspace_4_var[3949232]));
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 484; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_5_var[9696]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_5_var[9680]));
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
      ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t kh = 0; kh < 2; ++kh) {
      for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
        ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (p0[(((ax0_ax1_fused_ax2_fused % 121) * 4) + (kh * 2))] * ((float*)fused_constant_let)[((((ax0_ax1_fused_ax2_fused / 121) * 8) + (kh * 4)) + oc_block_c)]));
      }
    }
    for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
      ((float*)conv2d_NCHWc_let)[oc_block] = ((float*)conv2d_NCHWc_global_let)[oc_block];
    }
    for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
      float v_ = ((float*)conv2d_NCHWc_let)[ax4] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_let)[(((ax0_ax1_fused_ax2_fused / 121) * 4) + ax4)];
      T_relu[((ax0_ax1_fused_ax2_fused * 4) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* fused_nn_contrib_conv2d_NCHWc_constant_1_let = (&(global_const_workspace_6_var[3949104]));
  void* fused_constant_1_let = (&(global_const_workspace_6_var[3940000]));
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 480; ++ax0_ax1_fused_ax2_fused) {
    void* conv2d_NCHWc_let = (&(global_workspace_7_var[15424]));
    void* conv2d_NCHWc_global_let = (&(global_workspace_7_var[15440]));
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 4; ++oc_block_c_init) {
      ((float*)conv2d_NCHWc_global_let)[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 4; ++ic_outer) {
      for (int32_t kh = 0; kh < 2; ++kh) {
        for (int32_t ic_inner = 0; ic_inner < 4; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 4; ++oc_block_c) {
            ((float*)conv2d_NCHWc_global_let)[oc_block_c] = (((float*)conv2d_NCHWc_global_let)[oc_block_c] + (p0[((((ic_outer * 484) + ((ax0_ax1_fused_ax2_fused % 60) * 8)) + (kh * 4)) + ic_inner)] * ((float*)fused_constant_1_let)[((((((ax0_ax1_fused_ax2_fused / 60) * 128) + (ic_outer * 32)) + (kh * 16)) + (ic_inner * 4)) + oc_block_c)]));
          }
        }
      }
    }
    for (int32_t oc_block = 0; oc_block < 4; ++oc_block) {
      ((float*)conv2d_NCHWc_let)[oc_block] = ((float*)conv2d_NCHWc_global_let)[oc_block];
    }
    for (int32_t ax4 = 0; ax4 < 4; ++ax4) {
      float v_ = ((float*)conv2d_NCHWc_let)[ax4] + ((float*)fused_nn_contrib_conv2d_NCHWc_constant_1_let)[(((ax0_ax1_fused_ax2_fused / 60) * 4) + ax4)];
      T_relu[((ax0_ax1_fused_ax2_fused * 4) + ax4)] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add(float* p0, float* T_add, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* fused_nn_contrib_dense_pack_constant_2_let = (&(global_const_workspace_14_var[3949424]));
  void* fused_constant_4_let = (&(global_const_workspace_14_var[3944096]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 2; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_15_var[272]));
    void* compute_global_let = (&(global_workspace_15_var[304]));
    for (int32_t x_c_init = 0; x_c_init < 7; ++x_c_init) {
      ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
    }
    for (int32_t k_outer = 0; k_outer < 50; ++k_outer) {
      for (int32_t x_c = 0; x_c < 7; ++x_c) {
        ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_4_let)[(((ax1_outer_ax0_outer_fused * 350) + (k_outer * 7)) + x_c)]));
      }
    }
    for (int32_t x_inner_inner = 0; x_inner_inner < 7; ++x_inner_inner) {
      ((float*)compute_let)[x_inner_inner] = ((float*)compute_global_let)[x_inner_inner];
    }
    for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 7; ++ax1_inner_inner) {
      int32_t cse_var_1 = ((ax1_outer_ax0_outer_fused * 7) + ax1_inner_inner);
      T_add[cse_var_1] = (((float*)compute_let)[ax1_inner_inner] + ((float*)fused_nn_contrib_dense_pack_constant_2_let)[cse_var_1]);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(float* p0, float* T_relu, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* fused_nn_contrib_dense_pack_constant_let = (&(global_const_workspace_10_var[3946896]));
  void* fused_constant_2_let = (&(global_const_workspace_10_var[0]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 25; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_11_var[9680]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 4; ++y_inner_outer_x_inner_outer_fused) {
      void* compute_global_let = (&(global_workspace_11_var[9760]));
      for (int32_t x_c_init = 0; x_c_init < 5; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 1920; ++k_outer) {
        for (int32_t x_c = 0; x_c < 5; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_2_let)[((((ax1_outer_ax0_outer_fused * 38400) + (y_inner_outer_x_inner_outer_fused * 9600)) + (k_outer * 5)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 5; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 5) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 4; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 5; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 5);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 20) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(float* p0, float* T_relu, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* fused_nn_contrib_dense_pack_constant_1_let = (&(global_const_workspace_12_var[3948896]));
  void* fused_constant_3_let = (&(global_const_workspace_12_var[3840000]));
  for (int32_t ax1_outer_ax0_outer_fused = 0; ax1_outer_ax0_outer_fused < 5; ++ax1_outer_ax0_outer_fused) {
    void* compute_let = (&(global_workspace_13_var[208]));
    for (int32_t y_inner_outer_x_inner_outer_fused = 0; y_inner_outer_x_inner_outer_fused < 2; ++y_inner_outer_x_inner_outer_fused) {
      void* compute_global_let = (&(global_workspace_13_var[256]));
      for (int32_t x_c_init = 0; x_c_init < 5; ++x_c_init) {
        ((float*)compute_global_let)[x_c_init] = 0.000000e+00f;
      }
      for (int32_t k_outer = 0; k_outer < 500; ++k_outer) {
        for (int32_t x_c = 0; x_c < 5; ++x_c) {
          ((float*)compute_global_let)[x_c] = (((float*)compute_global_let)[x_c] + (p0[k_outer] * ((float*)fused_constant_3_let)[((((ax1_outer_ax0_outer_fused * 5000) + (y_inner_outer_x_inner_outer_fused * 2500)) + (k_outer * 5)) + x_c)]));
        }
      }
      for (int32_t x_inner_inner = 0; x_inner_inner < 5; ++x_inner_inner) {
        ((float*)compute_let)[((y_inner_outer_x_inner_outer_fused * 5) + x_inner_inner)] = ((float*)compute_global_let)[x_inner_inner];
      }
    }
    for (int32_t ax1_inner_outer = 0; ax1_inner_outer < 2; ++ax1_inner_outer) {
      for (int32_t ax1_inner_inner = 0; ax1_inner_inner < 5; ++ax1_inner_inner) {
        int32_t cse_var_2 = (ax1_inner_outer * 5);
        int32_t cse_var_1 = (((ax1_outer_ax0_outer_fused * 10) + cse_var_2) + ax1_inner_inner);
        float v_ = ((float*)compute_let)[(cse_var_2 + ax1_inner_inner)] + ((float*)fused_nn_contrib_dense_pack_constant_1_let)[cse_var_1];
        T_relu[cse_var_1] = ((v_) > (0.000000e+00f) ? (v_) : (0.000000e+00f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_log(float* p0, float* T_log, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* T_softmax_maxelem_let = (&(global_workspace_17_var[64]));
  ((float*)T_softmax_maxelem_let)[0] = -3.402823e+38f;
  for (int32_t k = 0; k < 14; ++k) {
    float v_ = ((float*)T_softmax_maxelem_let)[0];
    float v__1 = p0[k];
    ((float*)T_softmax_maxelem_let)[0] = ((v_) > (v__1) ? (v_) : (v__1));
  }
  void* T_softmax_exp_let = (&(global_workspace_17_var[0]));
  for (int32_t i1 = 0; i1 < 14; ++i1) {
    ((float*)T_softmax_exp_let)[i1] = expf((p0[i1] - ((float*)T_softmax_maxelem_let)[0]));
  }
  void* T_softmax_expsum_let = (&(global_workspace_17_var[128]));
  ((float*)T_softmax_expsum_let)[0] = 0.000000e+00f;
  for (int32_t k_1 = 0; k_1 < 14; ++k_1) {
    ((float*)T_softmax_expsum_let)[0] = (((float*)T_softmax_expsum_let)[0] + ((float*)T_softmax_exp_let)[k_1]);
  }
  void* T_softmax_norm_let = (&(global_workspace_17_var[64]));
  for (int32_t i1_1 = 0; i1_1 < 14; ++i1_1) {
    ((float*)T_softmax_norm_let)[i1_1] = (((float*)T_softmax_exp_let)[i1_1] / ((float*)T_softmax_expsum_let)[0]);
  }
  for (int32_t ax1 = 0; ax1 < 14; ++ax1) {
    T_log[ax1] = logf(((float*)T_softmax_norm_let)[ax1]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_layout_transform(float* p0, float* T_layout_trans, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 242; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 2; ++ax3) {
      int32_t cse_var_1 = ((ax0_ax1_fused_ax2_fused * 2) + ax3);
      T_layout_trans[cse_var_1] = p0[cse_var_1];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_buffer_var, float* output_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_6_let = (&(global_workspace_1_var[0]));
  void* sid_5_let = (&(global_workspace_1_var[7680]));
  void* sid_4_let = (&(global_workspace_1_var[0]));
  void* sid_3_let = (&(global_workspace_1_var[7744]));
  void* sid_1_let = (&(global_workspace_1_var[7744]));
  void* sid_2_let = (&(global_workspace_1_var[0]));
  void* sid_7_let = (&(global_workspace_1_var[208]));
  if (tvmgen_default_fused_reshape_layout_transform(input_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu_1(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_layout_transform_reshape(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add_nn_relu_1(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_contrib_dense_pack_add(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax_log(sid_7_let, output_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

