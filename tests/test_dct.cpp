#include "neo_dct.h"
#include "hwy/targets.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {
void check(bool yes, const char* message) {
  if (!yes)
    throw std::runtime_error(message);
}
void ok(neo_dct_status s) {
  check(s == NEO_DCT_OK, neo_dct_status_string(s));
}
struct Context {
  neo_dct_plan* p = nullptr;
  neo_dct_workspace* w = nullptr;
  Context(int x, int y, neo_dct_backend b, neo_dct_normalization n) {
    ok(neo_dct_plan_create(x, y, b, n, &p));
    ok(neo_dct_workspace_create(p, &w));
  }
  ~Context() {
    neo_dct_workspace_destroy(w);
    neo_dct_plan_destroy(p);
  }
};
double cosine(int n, int k, int x) {
  return std::cos(std::acos(-1.0) * (x + 0.5) * k / n);
}
std::vector<double> forward(const float* in, int w, int h, size_t stride, neo_dct_normalization norm) {
  std::vector<double> rows(w * h), out(w * h);
  for (int y = 0; y < h; ++y)
    for (int k = 0; k < w; ++k)
      for (int x = 0; x < w; ++x)
        rows[y * w + k] += in[y * stride + x] * cosine(w, k, x);
  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) {
      double s = 0;
      for (int y = 0; y < h; ++y)
        s += rows[y * w + u] * cosine(h, v, y);
      double f = norm == NEO_DCT_FFTW ? 4.0 : 2 * std::sqrt(2.0) / (w * h);
      if (norm == NEO_DCT_ORTHO)
        f = std::sqrt((u ? 2.0 : 1.0) / w) * std::sqrt((v ? 2.0 : 1.0) / h);
      out[v * w + u] = s * f;
    }
  return out;
}
std::vector<double> inverse8(const std::vector<double>& in, neo_dct_normalization norm) {
  std::vector<double> out(64);
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      for (int v = 0; v < 8; ++v)
        for (int u = 0; u < 8; ++u) {
          double scale = norm == NEO_DCT_FFTW ? (u ? 2.0 : 1.0) * (v ? 2.0 : 1.0)
                                              : std::sqrt((u ? 2.0 : 1.0) / 8) * std::sqrt((v ? 2.0 : 1.0) / 8);
          out[y * 8 + x] += in[v * 8 + u] * cosine(8, u, x) * cosine(8, v, y) * scale;
        }
  return out;
}
void close(double actual, double expected, double absolute) {
  if (std::abs(actual - expected) > absolute + std::abs(expected) * 3e-6) {
    std::cerr << actual << " != " << expected << " tolerance " << absolute << '\n';
    throw std::runtime_error("numeric mismatch");
  }
}
void dimensions(neo_dct_backend backend) {
  const int sizes[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128};
  std::mt19937 random(892);
  std::uniform_real_distribution<float> dist(-1, 1);
  // Every supported rectangle, including narrow dimensions in wide SIMD targets.
  for (int h : sizes)
    for (int w : sizes) {
      const size_t stride = w + 3, span = stride * h;
      std::vector<float> in(span), out(span, -999);
      for (auto& v : in)
        v = dist(random);
      for (auto norm : {NEO_DCT_FFTW, NEO_DCT_ORTHO, NEO_DCT_MV}) {
        Context c(w, h, backend, norm);
        neo_dct_batch b{in.data(), out.data(), stride, stride, span, span, 1};
        ok(neo_dct_forward(c.p, c.w, &b));
        const auto expected = forward(in.data(), w, h, stride, norm);
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            close(out[y * stride + x], expected[y * w + x], norm == NEO_DCT_FFTW ? 0.001 : 0.00002);
        for (int y = 0; y < h; ++y)
          for (size_t x = w; x < stride; ++x)
            check(out[y * stride + x] == -999, "padding overwritten");
      }
    }
}
void blocks8(neo_dct_backend backend) {
  std::mt19937 random(741);
  std::uniform_real_distribution<float> dist(-0.5, 1);
  for (size_t count : {size_t(1), size_t(3), size_t(4), size_t(7), size_t(8), size_t(9), size_t(17)}) {
    const size_t row = 11, span = 91;
    std::vector<float> in(count * span), out(count * span, -999);
    for (auto& v : in)
      v = dist(random);
    for (auto norm : {NEO_DCT_FFTW, NEO_DCT_ORTHO, NEO_DCT_MV}) {
      Context c(8, 8, backend, norm);
      neo_dct_batch b{in.data(), out.data(), row, row, span, span, count};
      ok(neo_dct_forward(c.p, c.w, &b));
      for (size_t k = 0; k < count; ++k) {
        auto expected = forward(in.data() + k * span, 8, 8, row, norm);
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            close(out[k * span + y * row + x], expected[y * 8 + x], 0.00003);
      }
      auto inplace = in;
      neo_dct_batch inout{inplace.data(), inplace.data(), row, row, span, span, count};
      ok(neo_dct_forward(c.p, c.w, &inout));
      for (size_t k = 0; k < count; ++k)
        for (size_t i = 0; i < span; ++i) {
          if (i / row < 8 && i % row < 8)
            close(inplace[k * span + i], out[k * span + i], 0.00001);
          else
            check(inplace[k * span + i] == in[k * span + i], "forward inplace padding");
        }
      if (norm == NEO_DCT_MV)
        continue;
      // Independent inverse test, not only a round trip.
      b.input = in.data();
      ok(neo_dct_inverse(c.p, c.w, &b));
      for (size_t k = 0; k < count; ++k) {
        std::vector<double> coeff(64);
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            coeff[y * 8 + x] = in[k * span + y * row + x];
        auto expected = inverse8(coeff, norm);
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            close(out[k * span + y * row + x], expected[y * 8 + x], 0.00004);
      }
      ok(neo_dct_forward(c.p, c.w, &b));
      b.input = out.data();
      ok(neo_dct_inverse(c.p, c.w, &b));
      for (size_t k = 0; k < count; ++k)
        for (int y = 0; y < 8; ++y)
          for (int x = 0; x < 8; ++x)
            close(out[k * span + y * row + x], in[k * span + y * row + x] * (norm == NEO_DCT_FFTW ? 256 : 1), 0.00015);
    }
    for (auto norm : {NEO_DCT_FFTW, NEO_DCT_ORTHO, NEO_DCT_MV}) {
      Context c(8, 8, backend, norm);
      std::array<float, 64> weights;
      for (int pattern = 0; pattern < 4; ++pattern) {
        for (int i = 0; i < 64; ++i)
          weights[i] = pattern == 0   ? 1.0f
                       : pattern == 1 ? 0.0f
                       : pattern == 2 ? (i == 0 ? 1.0f : 0.0f)
                                      : float((i * 37) % 65) / 64;
        std::fill(out.begin(), out.end(), -999);
        neo_dct_batch b{in.data(), out.data(), row, row, span, span, count};
        ok(neo_dct_filter8(c.p, c.w, &b, weights.data()));
        for (size_t k = 0; k < count; ++k) {
          auto coeff = forward(in.data() + k * span, 8, 8, row, NEO_DCT_FFTW);
          for (int i = 0; i < 64; ++i)
            coeff[i] *= weights[i] / 256.0;
          auto expected = inverse8(coeff, NEO_DCT_FFTW);
          for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
              close(out[k * span + y * row + x], expected[y * 8 + x], 0.000003);
          for (size_t i = 0; i < span; ++i)
            if (i / row >= 8 || i % row >= 8)
              check(out[k * span + i] == -999, "batch padding overwritten");
        }
        auto inplace = in;
        b.input = inplace.data();
        b.output = inplace.data();
        ok(neo_dct_filter8(c.p, c.w, &b, weights.data()));
        for (size_t k = 0; k < count; ++k)
          for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
              close(inplace[k * span + y * row + x], out[k * span + y * row + x], 0.000001);
      }
    }
  }
}
void errors() {
  neo_dct_plan* p = nullptr;
  check(neo_dct_plan_create(3, 8, NEO_DCT_AUTO, NEO_DCT_FFTW, &p) == NEO_DCT_UNSUPPORTED && !p, "invalid size");
  check(neo_dct_plan_create(8, 8, NEO_DCT_AUTO, NEO_DCT_FFTW, nullptr) == NEO_DCT_INVALID_ARGUMENT, "null plan output");
  Context c(8, 8, NEO_DCT_AUTO, NEO_DCT_FFTW);
  float a[130] = {};
  std::array<float, 64> weights{};
  neo_dct_batch b{a, a + 64, 8, 8, 64, 64, 1};
  b.input_row_stride = 7;
  check(neo_dct_forward(c.p, c.w, &b) == NEO_DCT_INVALID_ARGUMENT, "short stride");
  b.input_row_stride = 8;
  b.output = a + 1;
  check(neo_dct_forward(c.p, c.w, &b) == NEO_DCT_INVALID_ARGUMENT, "partial alias");
  b.output = a + 64;
  b.input_row_stride = SIZE_MAX;
  check(neo_dct_forward(c.p, c.w, &b) == NEO_DCT_INVALID_ARGUMENT, "overflow");
  b.input_row_stride = 8;
  b.count = 2;
  b.input_block_stride = 1;
  check(neo_dct_forward(c.p, c.w, &b) == NEO_DCT_INVALID_ARGUMENT, "overlapping blocks");
  b.count = 0;
  b.input = nullptr;
  b.output = nullptr;
  ok(neo_dct_forward(c.p, c.w, &b));
  weights[0] = std::numeric_limits<float>::quiet_NaN();
  check(neo_dct_filter8(c.p, c.w, &b, weights.data()) == NEO_DCT_INVALID_ARGUMENT, "nan factor");
  Context other(4, 8, NEO_DCT_AUTO, NEO_DCT_MV);
  check(neo_dct_inverse(other.p, other.w, &b) == NEO_DCT_UNSUPPORTED, "inverse dimension");
  check(neo_dct_forward(other.p, c.w, &b) == NEO_DCT_INVALID_ARGUMENT, "wrong workspace");
}
void pixel_ranges(neo_dct_backend backend) {
  Context c(8, 8, backend, NEO_DCT_FFTW);
  std::array<float, 64> in{}, out{}, weights{};
  for (float peak : {1.0f, 255.0f, 1023.0f, 65535.0f})
    for (int pattern = 0; pattern < 3; ++pattern) {
      for (int i = 0; i < 64; ++i) {
        in[i] = pattern == 0 ? peak : pattern == 1 ? (i == 19 ? peak : 0) : ((i + (i / 8)) % 2 ? peak : 0);
        weights[i] = float((i * 13) % 65) / 64;
      }
      neo_dct_batch b{in.data(), out.data(), 8, 8, 64, 64, 1};
      ok(neo_dct_filter8(c.p, c.w, &b, weights.data()));
      auto coeff = forward(in.data(), 8, 8, 8, NEO_DCT_FFTW);
      for (int i = 0; i < 64; ++i)
        coeff[i] *= weights[i] / 256.0;
      auto expected = inverse8(coeff, NEO_DCT_FFTW);
      for (int i = 0; i < 64; ++i)
        close(out[i], expected[i], peak * 0.000002);
    }
}
void layouts_and_threads() {
  Context c(8, 8, NEO_DCT_AUTO, NEO_DCT_FFTW);
  auto task = [&c](float seed) {
    neo_dct_workspace* raw = nullptr;
    ok(neo_dct_workspace_create(c.p, &raw));
    std::unique_ptr<neo_dct_workspace, decltype(&neo_dct_workspace_destroy)> work(raw, neo_dct_workspace_destroy);
    constexpr size_t count = 9, input_row = 9, output_row = 13;
    constexpr size_t input_span = 7 * input_row + 8, output_span = 7 * output_row + 8;
    std::vector<float> input(count * input_span), output(count * output_span, -999);
    std::array<float, 64> weights;
    weights.fill(1);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = seed + float(i % 17);
    neo_dct_batch b{input.data(), output.data(), input_row, output_row, input_span, output_span, count};
    for (int repeat = 0; repeat < 8; ++repeat)
      ok(neo_dct_filter8(c.p, work.get(), &b, weights.data()));
    for (size_t k = 0; k < count; ++k)
      for (size_t y = 0; y < 8; ++y)
        for (size_t x = 0; x < 8; ++x)
          close(output[k * output_span + y * output_row + x], input[k * input_span + y * input_row + x], 0.00003);
  };
  auto first = std::async(std::launch::async, task, 0.0f);
  auto second = std::async(std::launch::async, task, 10.0f);
  first.get();
  second.get();
}
void horizontal_strips(neo_dct_backend backend) {
  Context c(8, 8, backend, NEO_DCT_FFTW);
  constexpr size_t count = 9, row = count * 8 + 3;
  std::vector<float> input(row * 7 + count * 8), output(input.size(), -999);
  std::array<float, 64> weights;
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = float(i % 29);
  for (size_t i = 0; i < 64; ++i)
    weights[i] = float(i % 7) / 6;
  neo_dct_batch b{input.data(), output.data(), row, row, 8, 8, count};
  ok(neo_dct_filter8(c.p, c.w, &b, weights.data()));
  for (size_t block = 0; block < count; ++block) {
    auto coeff = forward(input.data() + block * 8, 8, 8, row, NEO_DCT_FFTW);
    for (size_t i = 0; i < 64; ++i)
      coeff[i] *= weights[i] / 256.0;
    auto expected = inverse8(coeff, NEO_DCT_FFTW);
    for (size_t y = 0; y < 8; ++y)
      for (size_t x = 0; x < 8; ++x)
        close(output[y * row + block * 8 + x], expected[y * 8 + x], 0.0001);
  }
  auto inplace = input;
  b.input = inplace.data();
  b.output = inplace.data();
  ok(neo_dct_filter8(c.p, c.w, &b, weights.data()));
  for (size_t y = 0; y < 8; ++y)
    for (size_t x = 0; x < count * 8; ++x)
      close(inplace[y * row + x], output[y * row + x], 0.00001);
  for (size_t y = 0; y < 7; ++y)
    for (size_t x = count * 8; x < row; ++x) {
      check(output[y * row + x] == -999, "strip output padding");
      check(inplace[y * row + x] == input[y * row + x], "strip inplace padding");
    }
}
} // namespace
int main() {
#ifdef _WIN32
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
  try {
    errors();
    dimensions(NEO_DCT_SCALAR);
    blocks8(NEO_DCT_SCALAR);
    pixel_ranges(NEO_DCT_SCALAR);
    horizontal_strips(NEO_DCT_SCALAR);
    for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
      hwy::SetSupportedTargetsForTest(target);
      dimensions(NEO_DCT_HIGHWAY);
      blocks8(NEO_DCT_HIGHWAY);
      pixel_ranges(NEO_DCT_HIGHWAY);
      horizontal_strips(NEO_DCT_HIGHWAY);
      std::cout << "Passed " << hwy::TargetName(target) << '\n';
    }
    hwy::SetSupportedTargetsForTest(0);
    layouts_and_threads();
    std::cout << "All DCT tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
