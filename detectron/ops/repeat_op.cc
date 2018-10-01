#include "repeat_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Repeat, RepeatOp<CPUContext>);

OPERATOR_SCHEMA(Repeat)
    .NumInputs(2)
    .NumOutputs(1, 2)
    .Input(
      0,
      "DATA",
      "Input Tensor")
    .Input(
      1,
      "REPEATS",
      "Number of repeats")
    .Output(
      0,
      "OUTPUT",
      "Output Tensor")
    .Output(
      1,
      "LENGTHS",
      "Lengths Tensor");

class GetRepeatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    CAFFE_ENFORCE_EQ(def_.output_size(), 2);

    return SingleGradientDef(
        "LengthsSum",
        "",
        // input 1 is the lengths used to repeat
        // DATA in the forward pass
        vector<string>{GO(0), O(1)},
        // only concerned with the gradient on "DATA"
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Repeat, GetRepeatGradient);

} // namespace caffe2
