# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box track output -> track loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Tracking outputs and losses
# ---------------------------------------------------------------------------- #

def add_tracking_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer

    blob_out =  model.FC(
        blob_in,
        'track_logits',
        dim,
        1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0),
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('track_logits', 'track_prob', engine='CUDNN')

    return blob_out


def add_tracking_losses(model):
    track_prob, loss_track = model.net.SoftmaxWithLoss(
        ['track_score', 'track_int32'], ['track_prob', 'loss_track'],
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_track])
    model.AddLosses(['loss_track'])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Tracking heads
# ---------------------------------------------------------------------------- #

def add_tracking_head(model, blob_in, dim_in, spatial_scale):
    """Add a Mask R-CNN tracking head."""
    hidden_dim = cfg.TRCNN.MLP_HEAD_DIM
    roi_size = cfg.TRCNN.ROI_XFORM_RESOLUTION
    n_rois = cfg.TRAIN.RPN_POST_NMS_TOP_N if model.train else cfg.TEST.RPN_POST_NMS_TOP_N
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'track_roi_feat',
        blob_rois='tracking_rois',
        method=cfg.TRCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.TRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(
        roi_feat,
        "track_fc",
        dim_in * roi_size * roi_size,
        hidden_dim,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0),
    )
    model.Relu("track_fc", "track_fc")
    # _init_track_fc_prev_blob(model, n_rois, hidden_dim)

    model.Length("track_fc_prev", "track_fc_prev_len")
    model.Length("track_fc", "track_fc_len")
    
    model.Tile(["track_fc_prev", "track_fc_len"], "track_fc_prev_tile", axis=0)
    model.Repeat(["track_fc", "track_fc_prev_len"], "track_fc_repeat")

    blob_feat = model.Concat(["track_fc_repeat", "track_fc_prev_tile"], "track_feat")

    return blob_feat, hidden_dim * 2


def _init_track_fc_prev_blob(model, n_rois, hidden_dim):
    matrix = np.zeros((n_rois, hidden_dim))
    return model.GivenTensorFill([], "track_fc_prev", values=matrix, shape=[n_rois, hidden_dim])
