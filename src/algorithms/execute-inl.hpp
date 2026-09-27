template <class Op>
void Execute(const Op& op, const neo_dct_plan& plan, neo_dct_workspace& work, const neo_dct_batch& batch,
             Operation operation, const float* weights) {
  if (plan.width == 8 && plan.height == 8 &&
      (plan.normalization != NEO_DCT_MV || operation == Operation::filter ||
       (op.lanes() >= 4 && batch.count >= static_cast<size_t>(op.lanes())))) {
    switch (operation) {
      case Operation::forward:
        Batch8<Operation::forward>(op, plan, work, batch, weights);
        break;
      case Operation::inverse:
        Batch8<Operation::inverse>(op, plan, work, batch, weights);
        break;
      case Operation::filter:
        Batch8<Operation::filter>(op, plan, work, batch, weights);
        break;
    }
    return;
  }
  const int w = plan.width, h = plan.height, stride = padded_stride(w);
  const int size = stride * padded_stride(h);
  float* input = work.storage.data();
  float* rows = input + size;
  float* output = rows + size;
  for (size_t block = 0; block < batch.count; ++block) {
    std::fill(input, input + size, 0.0f);
    for (int y = 0; y < h; ++y)
      std::copy_n(batch.input + block * batch.input_block_stride + y * batch.input_row_stride, w, input + y * stride);
    Transform(op, w, h, input, rows, output);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float value = output[y * stride + x];
        if (plan.normalization != NEO_DCT_MV) {
          const float sum_scale = float(w * h) / 0x1.6a09e6p+1f;
          if (plan.normalization == NEO_DCT_FFTW)
            value *= sum_scale * 4.0f;
          else
            value *= sum_scale * std::sqrt((x == 0 ? 1.0f : 2.0f) / w) * std::sqrt((y == 0 ? 1.0f : 2.0f) / h);
        }
        batch.output[block * batch.output_block_stride + y * batch.output_row_stride + x] = value;
      }
  }
}
