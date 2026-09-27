#pragma once
#include "neo_dct.h"
#include "algorithms/constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct neo_dct_plan {
  int width, height;
  neo_dct_backend backend;
  neo_dct_normalization normalization;
};
struct neo_dct_workspace {
  int width, height;
  std::vector<float> storage;
};
namespace neo_dct::detail {
inline int padded_stride(int n) {
  return (n + 7) & ~7;
}
enum class Operation { forward, inverse, filter };
void scalar_execute(const neo_dct_plan&, neo_dct_workspace&, const neo_dct_batch&, Operation, const float*);
void highway_execute(const neo_dct_plan&, neo_dct_workspace&, const neo_dct_batch&, Operation, const float*);
} // namespace neo_dct::detail
