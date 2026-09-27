#include "internal.hpp"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "backends/highway.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
HWY_BEFORE_NAMESPACE();
namespace neo_dct::detail {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
#include "backends/vector_ops-inl.hpp"
#define NEO_DCT_INLINE HWY_INLINE
#include "algorithms/forward-inl.hpp"
#include "algorithms/block8-inl.hpp"
#include "algorithms/execute-inl.hpp"
#undef NEO_DCT_INLINE
void ExecuteNative(const neo_dct_plan& p, neo_dct_workspace& w, const neo_dct_batch& b, Operation o, const float* f) {
  Execute(VectorOps{}, p, w, b, o, f);
}
} // namespace HWY_NAMESPACE
} // namespace neo_dct::detail
HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace neo_dct::detail {
HWY_EXPORT(ExecuteNative);
void highway_execute(const neo_dct_plan& p, neo_dct_workspace& w, const neo_dct_batch& b, Operation o, const float* f) {
  HWY_DYNAMIC_DISPATCH(ExecuteNative)(p, w, b, o, f);
}
} // namespace neo_dct::detail
#endif
