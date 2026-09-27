struct VectorOps {
  const hn::CappedTag<float, 8> d;
  int lanes() const { return int(hn::Lanes(d)); }
  using V = hn::VFromD<decltype(d)>;
  V load(const float* p) const { return hn::LoadU(d, p); }
  void store(V v, float* p) const { hn::StoreU(v, d, p); }
  V set(float v) const { return hn::Set(d, v); }
  V zero() const { return hn::Zero(d); }
  V add(V a, V b) const { return hn::Add(a, b); }
  V sub(V a, V b) const { return hn::Sub(a, b); }
  V mul(V a, V b) const { return hn::Mul(a, b); }
  V neg(V a) const { return hn::Neg(a); }
  HWY_INLINE void transpose(const float* in, int w, int h, size_t stride, float* out, size_t out_stride,
                            bool zero_padding = true) const {
    int tile = 1;
// Fixed-width SVE targets still use sizeless vector types, which cannot be
// array elements. Keep this array-based tile on non-SVE fixed-width targets.
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE && !(HWY_TARGET & HWY_ALL_SVE)
    const hn::Repartition<std::uint64_t, decltype(d)> d64;
    for (int y = 0; y < h; y += 8) {
      for (int x = 0; x < w; x += 8) {
        V r[8], a[8], b[8];
        for (int i = 0; i < 8; ++i)
          r[i] = y + i < h ? load(in + (y + i) * stride + x) : zero();
        for (int i = 0; i < 4; ++i) {
          a[2 * i] = hn::InterleaveLower(d, r[2 * i], r[2 * i + 1]);
          a[2 * i + 1] = hn::InterleaveUpper(d, r[2 * i], r[2 * i + 1]);
        }
        for (int j = 0; j < 8; j += 4) {
          b[j] = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, a[j]), hn::BitCast(d64, a[j + 2])));
          b[j + 1] = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, a[j]), hn::BitCast(d64, a[j + 2])));
          b[j + 2] = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, a[j + 1]), hn::BitCast(d64, a[j + 3])));
          b[j + 3] = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, a[j + 1]), hn::BitCast(d64, a[j + 3])));
        }
        for (int i = 0; i < 4; ++i) {
          if (x + i < w)
            store(hn::ConcatLowerLower(d, b[i + 4], b[i]), out + (x + i) * out_stride + y);
          if (x + i + 4 < w)
            store(hn::ConcatUpperUpper(d, b[i + 4], b[i]), out + (x + i + 4) * out_stride + y);
        }
      }
    }
    return;
#elif HWY_TARGET != HWY_SCALAR
    const hn::CappedTag<float, 4> d4;
    const hn::Repartition<std::uint64_t, decltype(d4)> d64;
    if (hn::Lanes(d4) == 4) {
      tile = 4;
      for (int y = 0; y + 4 <= h; y += 4)
        for (int x = 0; x + 4 <= w; x += 4) {
          const auto r0 = hn::LoadU(d4, in + y * stride + x), r1 = hn::LoadU(d4, in + (y + 1) * stride + x);
          const auto r2 = hn::LoadU(d4, in + (y + 2) * stride + x), r3 = hn::LoadU(d4, in + (y + 3) * stride + x);
          const auto a0 = hn::InterleaveLower(d4, r0, r1), a1 = hn::InterleaveUpper(d4, r0, r1);
          const auto a2 = hn::InterleaveLower(d4, r2, r3), a3 = hn::InterleaveUpper(d4, r2, r3);
          hn::StoreU(hn::BitCast(d4, hn::InterleaveLower(d64, hn::BitCast(d64, a0), hn::BitCast(d64, a2))), d4,
                     out + x * out_stride + y);
          hn::StoreU(hn::BitCast(d4, hn::InterleaveUpper(d64, hn::BitCast(d64, a0), hn::BitCast(d64, a2))), d4,
                     out + (x + 1) * out_stride + y);
          hn::StoreU(hn::BitCast(d4, hn::InterleaveLower(d64, hn::BitCast(d64, a1), hn::BitCast(d64, a3))), d4,
                     out + (x + 2) * out_stride + y);
          hn::StoreU(hn::BitCast(d4, hn::InterleaveUpper(d64, hn::BitCast(d64, a1), hn::BitCast(d64, a3))), d4,
                     out + (x + 3) * out_stride + y);
        }
    }
#endif
    const int full_w = tile == 1 ? 0 : w / tile * tile;
    const int full_h = tile == 1 ? 0 : h / tile * tile;
    for (int y = 0; y < h; ++y)
      for (int x = (y < full_h ? full_w : 0); x < w; ++x)
        out[x * out_stride + y] = in[y * stride + x];
    for (int x = 0; x < w; ++x)
      if (zero_padding)
        std::fill(out + x * out_stride + h, out + (x + 1) * out_stride, 0.0f);
  }
};
