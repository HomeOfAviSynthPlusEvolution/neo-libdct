#pragma once
namespace neo_dct::detail {
struct ScalarOps {
  static constexpr int lanes() { return 1; }
  float load(const float* p) const { return *p; }
  void store(float v, float* p) const { *p = v; }
  float set(float v) const { return v; }
  float zero() const { return 0; }
  float add(float a, float b) const { return a + b; }
  float sub(float a, float b) const { return a - b; }
  float mul(float a, float b) const { return a * b; }
  float neg(float a) const { return -a; }
  void transpose(const float* in, int w, int h, size_t stride, float* out, size_t out_stride,
                 bool zero_padding = true) const {
    for (int x = 0; x < w; ++x) {
      for (int y = 0; y < h; ++y)
        out[x * out_stride + y] = in[y * stride + x];
      if (zero_padding)
        std::fill(out + x * out_stride + h, out + (x + 1) * out_stride, 0.0f);
    }
  }
};
} // namespace neo_dct::detail
