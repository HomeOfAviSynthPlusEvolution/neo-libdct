#ifndef NEO_DCT_H
#define NEO_DCT_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum neo_dct_status {
  NEO_DCT_OK = 0,
  NEO_DCT_INVALID_ARGUMENT = 1,
  NEO_DCT_UNSUPPORTED = 2,
  NEO_DCT_OUT_OF_MEMORY = 3,
  NEO_DCT_INTERNAL_ERROR = 4
} neo_dct_status;
typedef enum neo_dct_backend { NEO_DCT_AUTO = 0, NEO_DCT_SCALAR = 1, NEO_DCT_HIGHWAY = 2 } neo_dct_backend;
typedef enum neo_dct_normalization {
  /* 2D DCT-II = 4*sum(input*cos*cos). Raw DCT-III is its inverse
     multiplied by 4*width*height (256 for 8x8). */
  NEO_DCT_FFTW = 0,
  /* Orthonormal coefficients; forward + inverse is identity. */
  NEO_DCT_ORTHO = 1,
  /* Legacy MV forward scaling: 2*sqrt(2)/(width*height)*sum.
     No pixel centering, quantization or special DC correction. */
  NEO_DCT_MV = 2
} neo_dct_normalization;

/* Immutable plan: may be shared by concurrent calls. No global thread pool. */
typedef struct neo_dct_plan neo_dct_plan;
/* Mutable scratch: one per concurrent execution; contains no pixel-format state. */
typedef struct neo_dct_workspace neo_dct_workspace;

/* All strides are in float elements, not bytes. Positive row strides >= width.
   Blocks may have disjoint storage envelopes, or form a horizontal strip:
   block_stride >= width and (count-1)*block_stride+width <= row_stride.
   Input/output may be exactly in-place with identical strides, or disjoint.
   Partial overlap is rejected. Buffer sizes/lifetimes are caller responsibilities.
   Padding is not read or written. count=0 is a no-op, with null data allowed. */
typedef struct neo_dct_batch {
  const float* input;
  float* output;
  size_t input_row_stride;
  size_t output_row_stride;
  size_t input_block_stride;
  size_t output_block_stride;
  size_t count;
} neo_dct_batch;

int neo_dct_axis_supported(int length);
neo_dct_status neo_dct_plan_create(int width, int height, neo_dct_backend backend, neo_dct_normalization normalization,
                                   neo_dct_plan** out);
void neo_dct_plan_destroy(neo_dct_plan* plan);
neo_dct_status neo_dct_workspace_create(const neo_dct_plan* plan, neo_dct_workspace** out);
void neo_dct_workspace_destroy(neo_dct_workspace* workspace);
/* Float scratch payload only; excludes small bookkeeping/allocation overhead. */
size_t neo_dct_workspace_bytes(const neo_dct_plan* plan);
neo_dct_status neo_dct_forward(const neo_dct_plan* plan, neo_dct_workspace* workspace, const neo_dct_batch* batch);
/* Currently 8x8 only; FFTW and ORTHO normalization supported. */
neo_dct_status neo_dct_inverse(const neo_dct_plan* plan, neo_dct_workspace* workspace, const neo_dct_batch* batch);
/* 8x8 only. Always a normalized round trip, independent of plan normalization.
   weights[v*8+u] multiplies coefficient (u,v). 64 finite values required;
   values need not be in [0,1]. Copy eight user factors' outer product here
   to implement DCTFilter. Input/coefficients are F32, including for F16 hosts.
   Weights are copied before execution; they may alias input/output. */
neo_dct_status neo_dct_filter8(const neo_dct_plan* plan, neo_dct_workspace* workspace, const neo_dct_batch* batch,
                               const float* weights);
const char* neo_dct_status_string(neo_dct_status status);

#ifdef __cplusplus
}
#endif
#endif
