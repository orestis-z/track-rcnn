#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// RecordLengthOp records the size of the first dimension of the input tensor to a vector of int.
template <class Context>
class LengthOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor<Context>>(0);
    output->Resize(1);
    int* output_data = output->template mutable_data<int>();
    context_.template CopyBytes<Context, Context>(
        sizeof(int), input.dims().data(), output_data);
    return true;
  }
};

} // namespace caffe2
