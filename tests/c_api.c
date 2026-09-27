#include "neo_dct.h"
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif
int main(void) {
  neo_dct_plan* plan = 0;
  neo_dct_workspace* work = 0;
  float input[64], output[64], weights[64];
  int i;
  neo_dct_batch batch = {input, output, 8, 8, 64, 64, 1};
#ifdef _WIN32
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
  for (i = 0; i < 64; ++i) {
    input[i] = (float)i;
    weights[i] = 1.0f;
  }
  if (neo_dct_plan_create(8, 8, NEO_DCT_AUTO, NEO_DCT_FFTW, &plan) != NEO_DCT_OK)
    return 1;
  if (neo_dct_workspace_create(plan, &work) != NEO_DCT_OK)
    return 2;
  if (neo_dct_filter8(plan, work, &batch, weights) != NEO_DCT_OK)
    return 3;
  for (i = 0; i < 64; ++i)
    if (fabs(output[i] - input[i]) > 0.0001)
      return 4;
  neo_dct_workspace_destroy(work);
  neo_dct_plan_destroy(plan);
  return 0;
}
