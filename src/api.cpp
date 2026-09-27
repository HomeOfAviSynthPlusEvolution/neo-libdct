#include "internal.hpp"
#include <array>
#include <limits>
#include <memory>
#include <new>

namespace {
size_t scratch_count(const neo_dct_plan& p) {
  return std::max(size_t(1024),
                  size_t(3) * neo_dct::detail::padded_stride(p.width) * neo_dct::detail::padded_stride(p.height));
}
bool extent(size_t row, size_t block, size_t count, int w, int h, size_t& result) {
  const size_t max = static_cast<size_t>(PTRDIFF_MAX) / sizeof(float);
  if (row < size_t(w) || row > (max - size_t(w)) / size_t(h - 1))
    return false;
  const size_t span = row * size_t(h - 1) + size_t(w);
  if (count > 1) {
    if (block > (max - span) / (count - 1))
      return false;
    // Either disjoint block envelopes, or a horizontal strip of blocks whose
    // samples do not overlap even though their row-strided envelopes do.
    if (block < span && (block < size_t(w) || block > (row - size_t(w)) / (count - 1)))
      return false;
  }
  result = count > 1 ? (count - 1) * block + span : span;
  return true;
}
neo_dct_status run(const neo_dct_plan* p, neo_dct_workspace* w, const neo_dct_batch* b, neo_dct::detail::Operation op,
                   const float* weights) noexcept {
  using neo_dct::detail::Operation;
  if (!p || !w || !b || w->width != p->width || w->height != p->height)
    return NEO_DCT_INVALID_ARGUMENT;
  if (op != Operation::forward && (p->width != 8 || p->height != 8))
    return NEO_DCT_UNSUPPORTED;
  if (op == Operation::inverse && p->normalization == NEO_DCT_MV)
    return NEO_DCT_UNSUPPORTED;
  std::array<float, 64> factors{};
  if (op == Operation::filter) {
    if (!weights)
      return NEO_DCT_INVALID_ARGUMENT;
    for (size_t i = 0; i < 64; ++i) {
      if (!std::isfinite(weights[i]))
        return NEO_DCT_INVALID_ARGUMENT;
      factors[i] = weights[i];
    }
  }
  if (b->count == 0)
    return NEO_DCT_OK;
  size_t in_size = 0, out_size = 0;
  if (!b->input || !b->output ||
      !extent(b->input_row_stride, b->input_block_stride, b->count, p->width, p->height, in_size) ||
      !extent(b->output_row_stride, b->output_block_stride, b->count, p->width, p->height, out_size))
    return NEO_DCT_INVALID_ARGUMENT;
  const auto in = reinterpret_cast<uintptr_t>(b->input), out = reinterpret_cast<uintptr_t>(b->output);
  if (in > UINTPTR_MAX - in_size * sizeof(float) || out > UINTPTR_MAX - out_size * sizeof(float))
    return NEO_DCT_INVALID_ARGUMENT;
  const bool identical = in == out && b->input_row_stride == b->output_row_stride &&
                         (b->count == 1 || b->input_block_stride == b->output_block_stride);
  if (!identical && in < out + out_size * sizeof(float) && out < in + in_size * sizeof(float))
    return NEO_DCT_INVALID_ARGUMENT;
  try {
    if (p->backend == NEO_DCT_SCALAR)
      neo_dct::detail::scalar_execute(*p, *w, *b, op, factors.data());
    else
      neo_dct::detail::highway_execute(*p, *w, *b, op, factors.data());
    return NEO_DCT_OK;
  } catch (...) {
    return NEO_DCT_INTERNAL_ERROR;
  }
}
} // namespace
extern "C" {
int neo_dct_axis_supported(int n) {
  switch (n) {
    case 2:
    case 4:
    case 6:
    case 8:
    case 12:
    case 16:
    case 24:
    case 32:
    case 48:
    case 64:
    case 128:
      return 1;
    default:
      return 0;
  }
}
neo_dct_status neo_dct_plan_create(int width, int height, neo_dct_backend backend, neo_dct_normalization normalization,
                                   neo_dct_plan** out) {
  if (!out)
    return NEO_DCT_INVALID_ARGUMENT;
  *out = nullptr;
  if (!neo_dct_axis_supported(width) || !neo_dct_axis_supported(height))
    return NEO_DCT_UNSUPPORTED;
  if (backend < NEO_DCT_AUTO || backend > NEO_DCT_HIGHWAY || normalization < NEO_DCT_FFTW || normalization > NEO_DCT_MV)
    return NEO_DCT_INVALID_ARGUMENT;
  try {
    *out = new neo_dct_plan{width, height, backend, normalization};
    return NEO_DCT_OK;
  } catch (const std::bad_alloc&) {
    return NEO_DCT_OUT_OF_MEMORY;
  } catch (...) {
    return NEO_DCT_INTERNAL_ERROR;
  }
}
void neo_dct_plan_destroy(neo_dct_plan* p) {
  delete p;
}
neo_dct_status neo_dct_workspace_create(const neo_dct_plan* p, neo_dct_workspace** out) {
  if (!out)
    return NEO_DCT_INVALID_ARGUMENT;
  *out = nullptr;
  if (!p)
    return NEO_DCT_INVALID_ARGUMENT;
  try {
    auto w = std::make_unique<neo_dct_workspace>();
    w->width = p->width;
    w->height = p->height;
    w->storage.resize(scratch_count(*p));
    *out = w.release();
    return NEO_DCT_OK;
  } catch (const std::bad_alloc&) {
    return NEO_DCT_OUT_OF_MEMORY;
  } catch (...) {
    return NEO_DCT_INTERNAL_ERROR;
  }
}
void neo_dct_workspace_destroy(neo_dct_workspace* w) {
  delete w;
}
size_t neo_dct_workspace_bytes(const neo_dct_plan* p) {
  return p ? scratch_count(*p) * sizeof(float) : 0;
}
neo_dct_status neo_dct_forward(const neo_dct_plan* p, neo_dct_workspace* w, const neo_dct_batch* b) {
  return run(p, w, b, neo_dct::detail::Operation::forward, nullptr);
}
neo_dct_status neo_dct_inverse(const neo_dct_plan* p, neo_dct_workspace* w, const neo_dct_batch* b) {
  return run(p, w, b, neo_dct::detail::Operation::inverse, nullptr);
}
neo_dct_status neo_dct_filter8(const neo_dct_plan* p, neo_dct_workspace* w, const neo_dct_batch* b, const float* f) {
  return run(p, w, b, neo_dct::detail::Operation::filter, f);
}
const char* neo_dct_status_string(neo_dct_status s) {
  switch (s) {
    case NEO_DCT_OK:
      return "success";
    case NEO_DCT_INVALID_ARGUMENT:
      return "invalid argument";
    case NEO_DCT_UNSUPPORTED:
      return "unsupported operation or dimensions";
    case NEO_DCT_OUT_OF_MEMORY:
      return "out of memory";
    case NEO_DCT_INTERNAL_ERROR:
      return "internal error";
    default:
      return "unknown status";
  }
}
}
