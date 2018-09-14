from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils


class RepeatOpTest(unittest.TestCase):

    def _run_repeat_op(self, X, repeats):
        op = core.CreateOperator('Repeat', ['X', 'repeats'], ['Y'])
        workspace.FeedBlob('X', X)
        workspace.FeedBlob('repeats', repeats)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def _run_repeat_op_gpu(self, X, repeats):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            op = core.CreateOperator('Repeat', ['X', 'repeats'], ['Y'])
            workspace.FeedBlob('X', X)
            workspace.FeedBlob('repeats', repeats)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        return Y

    def test_random_input(self):
        X = np.random.rand(3, 5)
        Y_exp = np.repeat(np.copy(X), 3, axis=0)
        Y_act = self._run_repeat_op(X, np.array(3, dtype=np.int32))
        np.testing.assert_allclose(Y_act, Y_exp)

    def test_gpu_random_input(self):
        X = np.random.rand(3, 5)
        Y_exp = np.repeat(np.copy(X), 3, axis=0)
        Y_act = self._run_repeat_op_gpu(X, np.array(3, dtype=np.int32))
        np.testing.assert_allclose(Y_act, Y_exp)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    c2_utils.import_custom_ops()
    assert 'Repeat' in workspace.RegisteredOperators()
    unittest.main()
