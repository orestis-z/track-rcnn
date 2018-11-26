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

from detectron.core.config import cfg
from detectron.core.config import assert_and_infer_cfg
import detectron.utils.c2 as c2_utils
from detectron.utils.math import cosine_similarity
from detectron.modeling.track_rcnn_heads import add_track_losses
from detectron.modeling.detector import DetectionModelHelper

c2_utils.import_custom_ops()


class TrackLossesTest(unittest.TestCase):

    def _add_track_losses(self, X, X_gt):
        model = DetectionModelHelper(train=False, num_classes=1)
        add_track_losses(model)
        workspace.FeedBlob('track_similarity', X)
        workspace.FeedBlob('track_int32', X_gt)
        workspace.RunNetOnce(model.net)
        return workspace.FetchBlob('loss_track')
 
    def _add_track_losses_np(self, arr_in, arr_gt):
        if cfg.TRCNN.LOSS == 'Cosine':
            track_cosine_similarity = cosine_similarity(arr_in, arr_gt)
            loss_track_raw = 1 - track_cosine_similarity
        elif cfg.TRCNN.LOSS == 'L2':
            track_l2_loss = 0.5 * np.sum(np.square(arr_in - arr_gt))
            loss_track_raw = track_l2_loss / arr_in.shape[1]
        elif cfg.TRCNN.LOSS == 'L2Balanced': 
            track_int32_non_matches = 1 - arr_gt
            track_delta_sq = np.square(arr_in - arr_gt)
            loss_track_matches_raw = np.matmul(track_delta_sq, arr_gt.T)[0]
            loss_track_non_matches_raw = np.matmul(track_delta_sq, track_int32_non_matches.T)[0]
            loss_track_matches = loss_track_matches_raw / np.sum(arr_gt)
            loss_track_non_matches = loss_track_non_matches_raw / np.sum(track_int32_non_matches)
            loss_track_raw = 0.5 * (loss_track_matches + loss_track_non_matches)
        else:
            # raise NotImplementedError,
            print('Test case for loss "{}" not implemented yet'.format(cfg.TRCNN.LOSS))

        return cfg.TRCNN.LOSS_WEIGHT * loss_track_raw

    def test_gpu_random_input_gpu(self):
        X = np.random.rand(1, 6).astype(np.float32)
        X_gt = np.random.randint(2, size=(1, 6)).astype(np.float32)
        for loss in ['Cosine', 'L2', 'L2Balanced', 'CrossEntropy', 'CrossEntropyBalanced', 'CrossEntropyWeighted']:
            cfg.immutable(False)
            cfg.TRCNN.LOSS = loss
            assert_and_infer_cfg(cache_urls=False)
            Y_exp = self._add_track_losses_np(X.copy(), X_gt.copy())
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                Y_act = self._add_track_losses(X.copy(), X_gt.copy())
            np.testing.assert_allclose(Y_act, Y_exp, rtol=1e-06)

    def test_gpu_random_input(self):
        X = np.random.rand(1, 6).astype(np.float32)
        X_gt = np.random.randint(2, size=(1, 6)).astype(np.float32)
        for loss in ['Cosine', 'L2', 'L2Balanced', 'CrossEntropy', 'CrossEntropyBalanced', 'CrossEntropyWeighted']:
            cfg.immutable(False)
            cfg.TRCNN.LOSS = loss
            assert_and_infer_cfg(cache_urls=False)
            Y_exp = self._add_track_losses_np(X.copy(), X_gt.copy())
            Y_act = self._add_track_losses(X.copy(), X_gt.copy())
            np.testing.assert_allclose(Y_act, Y_exp, rtol=1e-06)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    unittest.main()
