// Included in a backend namespace. SIMD lanes represent independent 8x8 blocks.
// All scratch arrays have 64*8 floats, including when the backend has fewer lanes.
inline float cosine8(int frequency, int position) {
  return position < 4 ? matrix8[frequency][position]
                      : ((frequency & 1) ? -matrix8[frequency][7 - position] : matrix8[frequency][7 - position]);
}
template <class Op>
NEO_DCT_INLINE void Forward8Axis(const Op& op, const float* in, float* out, int axis) {
  const int lanes = op.lanes();
  for (int line = 0; line < 8; ++line) {
    for (int k = 0; k < 8; ++k) {
      auto sum = op.zero();
      for (int i = 0; i < 4; ++i) {
        const int a = axis == 0 ? line * 8 + i : i * 8 + line;
        const int b = axis == 0 ? line * 8 + 7 - i : (7 - i) * 8 + line;
        auto av = op.load(in + a * lanes), bv = op.load(in + b * lanes);
        auto pair = (k & 1) ? op.sub(av, bv) : op.add(av, bv);
        sum = op.add(sum, op.mul(pair, op.set(matrix8[k][i])));
      }
      const int dst = axis == 0 ? line * 8 + k : k * 8 + line;
      op.store(op.mul(sum, op.set(2.0f)), out + dst * lanes);
    }
  }
}
template <class Op>
NEO_DCT_INLINE void Inverse8Axis(const Op& op, const float* in, float* out, int axis) {
  const int lanes = op.lanes();
  for (int line = 0; line < 8; ++line) {
    // Pair outputs at x and 7-x: odd frequencies change sign.
    for (int x = 0; x < 4; ++x) {
      const int dc = axis == 0 ? line * 8 : line;
      auto even = op.load(in + dc * lanes), odd = op.zero();
      for (int k = 1; k < 8; ++k) {
        const int pos = axis == 0 ? line * 8 + k : k * 8 + line;
        auto term = op.mul(op.load(in + pos * lanes), op.set(2.0f * cosine8(k, x)));
        if (k & 1)
          odd = op.add(odd, term);
        else
          even = op.add(even, term);
      }
      const int a = axis == 0 ? line * 8 + x : x * 8 + line;
      const int b = axis == 0 ? line * 8 + 7 - x : (7 - x) * 8 + line;
      op.store(op.add(even, odd), out + a * lanes);
      op.store(op.sub(even, odd), out + b * lanes);
    }
  }
}
inline float ortho8(int k) {
  return k == 0 ? 0.353553390593273762f : 0.5f;
}
// Columns occupy SIMD lanes for a small batch, so even one block uses the
// vector width. Matrix8 computes an unscaled cosine sum (not FFTW's 2*sum).
template <class Op>
NEO_DCT_INLINE void Inverse8Columns(const Op& op, const float* in, float* out) {
  for (int j = 0; j < 8; j += op.lanes()) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
    for (int x = 0; x < 4; ++x) {
      auto even = op.load(in + j), odd = op.zero();
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
      for (int k = 1; k < 8; ++k) {
        auto term = op.mul(op.load(in + k * 8 + j), op.set(2.0f * cosine8(k, x)));
        if (k & 1)
          odd = op.add(odd, term);
        else
          even = op.add(even, term);
      }
      op.store(op.add(even, odd), out + x * 8 + j);
      op.store(op.sub(even, odd), out + (7 - x) * 8 + j);
    }
  }
}

template <Operation operation, class Op>
void SmallBatch8(const Op& op, const neo_dct_plan& plan, neo_dct_workspace& work, const neo_dct_batch& batch,
                 size_t first, const float* weights) {
  float* a = work.storage.data();
  float* b = a + 64;
  for (size_t block = first; block < batch.count; ++block) {
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        float scale = operation == Operation::filter ? 1.0f / 64.0f : 1.0f;
        if (operation == Operation::inverse && plan.normalization == NEO_DCT_ORTHO)
          scale = 4.0f / (ortho8(y) * ortho8(x));
        a[y * 8 + x] = batch.input[block * batch.input_block_stride + y * batch.input_row_stride + x] * scale;
      }
    if (operation != Operation::inverse) {
      op.transpose(a, 8, 8, 8, b, 8);
      Matrix8<false>(op, b, a);
      op.transpose(a, 8, 8, 8, b, 8);
      if (plan.normalization == NEO_DCT_MV && operation == Operation::forward)
        Matrix8<true>(op, b, a);
      else
        Matrix8<false>(op, b, a);
    }
    if (operation == Operation::filter)
      for (int i = 0; i < 64; i += op.lanes())
        op.store(op.mul(op.load(a + i), op.load(weights + i)), a + i);
    if (operation != Operation::forward) {
      op.transpose(a, 8, 8, 8, b, 8);
      Inverse8Columns(op, b, a);
      op.transpose(a, 8, 8, 8, b, 8);
      Inverse8Columns(op, b, a);
    }
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        float scale = 1.0f;
        if (operation == Operation::forward)
          scale = plan.normalization == NEO_DCT_ORTHO ? ortho8(y) * ortho8(x)
                  : plan.normalization == NEO_DCT_MV  ? 1.0f
                                                      : 4.0f;
        if (operation == Operation::inverse && plan.normalization == NEO_DCT_ORTHO)
          scale = 1.0f / 256.0f;
        batch.output[block * batch.output_block_stride + y * batch.output_row_stride + x] = a[y * 8 + x] * scale;
      }
  }
}

template <Operation operation, class Op>
void Batch8(const Op& op, const neo_dct_plan& plan, neo_dct_workspace& work, const neo_dct_batch& batch,
            const float* weights) {
  const size_t lanes = static_cast<size_t>(op.lanes());
  float* a = work.storage.data();
  float* b = a + 512;
  for (size_t first = 0; first < batch.count;) {
    const size_t active = std::min(lanes, batch.count - first);
    // A nearly full group is cheaper to process across blocks than repeatedly
    // paying the two intra-block transposes. Keep only short tails here.
    if (active <= 3 && active < lanes) {
      SmallBatch8<operation>(op, plan, work, batch, first, weights);
      return;
    }
    // Full groups transpose contiguous pixels into independent SIMD lanes.
    const bool packed = lanes == 8 && active == 8;
    if (packed) {
      for (int y = 0; y < 8; ++y)
        op.transpose(batch.input + first * batch.input_block_stride + y * batch.input_row_stride, 8, 8,
                     batch.input_block_stride, a + y * 64, 8);
    }
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        float scale = operation == Operation::filter ? 1.0f / 256.0f : 1.0f;
        if (operation == Operation::inverse && plan.normalization == NEO_DCT_ORTHO)
          scale = 4.0f / (ortho8(y) * ortho8(x));
        float* dst = a + (y * 8 + x) * lanes;
        if (packed) {
          if (scale != 1.0f)
            op.store(op.mul(op.load(dst), op.set(scale)), dst);
        } else {
          for (size_t lane = 0; lane < lanes; ++lane)
            dst[lane] =
                lane < active
                    ? batch.input[(first + lane) * batch.input_block_stride + y * batch.input_row_stride + x] * scale
                    : 0.0f;
        }
      }
    if (operation != Operation::inverse) {
      Forward8Axis(op, a, b, 0);
      Forward8Axis(op, b, a, 1);
    }
    if (operation == Operation::filter)
      for (int i = 0; i < 64; ++i)
        op.store(op.mul(op.load(a + i * lanes), op.set(weights[i])), a + i * lanes);
    if (operation != Operation::forward) {
      Inverse8Axis(op, a, b, 0);
      Inverse8Axis(op, b, a, 1);
    }
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        float scale = 1.0f;
        if (plan.normalization == NEO_DCT_ORTHO) {
          if (operation == Operation::forward)
            scale = ortho8(y) * ortho8(x) / 4.0f;
          if (operation == Operation::inverse)
            scale = 1.0f / 256.0f;
        }
        if (operation == Operation::forward && plan.normalization == NEO_DCT_MV)
          scale = 0x1.6a09e6p+1f / 256.0f;
        float* src = a + (y * 8 + x) * lanes;
        if (packed) {
          if (scale != 1.0f)
            op.store(op.mul(op.load(src), op.set(scale)), src);
        } else {
          for (size_t lane = 0; lane < active; ++lane)
            batch.output[(first + lane) * batch.output_block_stride + y * batch.output_row_stride + x] =
                src[lane] * scale;
        }
      }
    if (packed)
      for (int y = 0; y < 8; ++y)
        op.transpose(a + y * 64, 8, 8, 8,
                     batch.output + first * batch.output_block_stride + y * batch.output_row_stride,
                     batch.output_block_stride, false);
    first += active;
  }
}
