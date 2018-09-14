#include "length_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Length, LengthOp<CPUContext>);

OPERATOR_SCHEMA(Length)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims().size());
      out[0].set_data_type(TensorProto::INT32);
      return out;
    })
    .SetDoc("Size of the first dimension of the input tensor.");

SHOULD_NOT_DO_GRADIENT(Length);

} // namespace caffe2
