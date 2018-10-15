from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy import spatial
import unittest

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

import detectron.utils.c2 as c2_utils
from detectron.utils.math import cosine_similarity
from detectron.modeling.track_rcnn_heads import add_track_outputs
from detectron.modeling.detector import DetectionModelHelper

c2_utils.import_custom_ops()


class TrackOutputsTest(unittest.TestCase):

    def _add_track_outputs(self, X, track_n_rois):
        model = DetectionModelHelper(train=False, num_classes=1)
        X = model.GivenTensorFill([], 'X', values=X, shape=X.shape)
        add_track_outputs(model, X, None)
        workspace.FeedBlob('track_n_rois', track_n_rois)
        workspace.RunNetOnce(model.net)
        return workspace.FetchBlob('track_similarity')
 
    def _add_track_outputs_np(self, arr_in, track_n_rois):
        track_fc_one, track_fc_two, _ = np.split(arr_in, [track_n_rois[0], track_n_rois[0] + track_n_rois[1]])
        track_fc_one_len, track_fc_two_len = track_n_rois
        track_fc_one_tile = np.tile(track_fc_one, (track_fc_two_len, 1))
        track_fc_two_repeat = np.repeat(track_fc_two, track_fc_one_len, axis=0)
        track_cos_similarity = cosine_similarity(track_fc_one_tile, track_fc_two_repeat)
        track_similarity = np.expand_dims(track_cos_similarity, axis=0)
        return track_similarity.astype(np.float32)

    def test_gpu_random_input_gpu(self):
        X = np.random.rand(5, 4)
        track_n_rois = [2, 3]
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            Y_act = self._add_track_outputs(X, np.array(track_n_rois, dtype=np.int32))
        Y_exp = self._add_track_outputs_np(X.copy(), track_n_rois)
        np.testing.assert_allclose(Y_act, Y_exp, rtol=1e-06)

    def test_gpu_random_input(self):
        X = np.random.rand(5, 4)
        track_n_rois = [2, 3]
        Y_exp = self._add_track_outputs_np(X.copy(), track_n_rois)
        Y_act = self._add_track_outputs(X, np.array(track_n_rois, dtype=np.int32))
        np.testing.assert_allclose(Y_act, Y_exp, rtol=1e-06)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    unittest.main()