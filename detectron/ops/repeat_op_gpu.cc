#include "caffe2/core/context_gpu.h"
#include "repeat_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Repeat, RepeatOp<CUDAContext>);
} // namespace caffe2
