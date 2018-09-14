#include "repeat_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Repeat, RepeatOp<CPUContext>);

OPERATOR_SCHEMA(Repeat)
    .NumInputs(2)
    .NumOutputs(1)
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
      "Output Tensor");

} // namespace caffe2
