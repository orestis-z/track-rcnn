#include "caffe2/core/context_gpu.h"
#include "length_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Length, LengthOp<CUDAContext>);
}
