#ifndef CAFFE2_OPERATORS_REPEAT_OP_H_
#define CAFFE2_OPERATORS_REPEAT_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class RepeatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(RepeatOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<Tensor<Context>>(REPEATS));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& repeats = Input(REPEATS);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    CAFFE_ENFORCE(repeats.size() == 1);

    CPUContext cpuContext;
    repeats_host_.CopyFrom(repeats, &cpuContext);
    auto* repeats_data = repeats_host_.data<T>();

    auto shape = data.dims();
    auto len = shape[0];
    shape[0] = len * repeats_data[0];
    output->Resize(shape);

    if (OutputSize() > 1) {
      auto* lengths = Output(1);
      int shape_lengths[] = {len};
      int* shape_lengths_ptr = &shape_lengths[0];
      lengths->Resize(*shape_lengths_ptr);
      auto* lengths_data = lengths->template mutable_data<T>();
      math::Set<T, Context>(lengths->size(), repeats_data[0], lengths_data, &context_);
    }

    auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
    auto src = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (TIndex i = 0; i < len; ++i) {
      for (int32_t j = 0; j < repeats_data[0]; ++j) {
        context_.template CopyBytes<Context, Context>(block_bytesize, src, out);
        out += block_bytesize;
      }
      src += block_bytesize;
    }
    return true;
  }

  INPUT_TAGS(DATA, REPEATS);

  private:
    TensorCPU repeats_host_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REPEAT_OP_H_
