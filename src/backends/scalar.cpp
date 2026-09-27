#include "internal.hpp"
#include "backends/scalar_ops.hpp"
namespace neo_dct::detail::scalar {
#define NEO_DCT_INLINE inline
#include "algorithms/forward-inl.hpp"
#include "algorithms/block8-inl.hpp"
#include "algorithms/execute-inl.hpp"
#undef NEO_DCT_INLINE
} // namespace neo_dct::detail::scalar
namespace neo_dct::detail {
void scalar_execute(const neo_dct_plan& p, neo_dct_workspace& w, const neo_dct_batch& b, Operation o, const float* f) {
  scalar::Execute(ScalarOps{}, p, w, b, o, f);
}
} // namespace neo_dct::detail
