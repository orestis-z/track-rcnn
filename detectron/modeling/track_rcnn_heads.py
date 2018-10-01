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

def add_track_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer

    # blob_out =  model.FC(
    #     blob_in,
    #     'track_logits',
    #     dim,
    #     1,
    #     weight_init=gauss_fill(0.01),
    #     bias_init=const_fill(0.0),
    # )
    model.EnsureCPUOutput("track_n_rois", "track_n_rois_cpu")
    
    if model.train:
        model.Split(["track_ids_int32", "track_n_rois_cpu"], ["track_ids_one_int32", "track_ids_two_int32"], axis=0)
        model.GenerateTrackingLabels(["track_ids_one_int32", "track_ids_two_int32"], "track_int32")

    model.Split([blob_in, "track_n_rois_cpu"], ["track_fc_one", "track_fc_two"], axis=0)
    model.Length("track_fc_one", "track_fc_one_len")
    model.Length("track_fc_two", "track_fc_two_len")
    
    model.Tile(["track_fc_one", "track_fc_two_len"], "track_fc_one_tile", axis=0)

    repeat_outputs = ["track_fc_two_repeat"]
    if model.train:
        repeat_outputs.append("track_fc_two_repeat_lengths")
    model.Repeat(["track_fc_two", "track_fc_one_len"], repeat_outputs)


    # track_concat = model.Concat(["track_fc_repeat", "track_fc_two_tile"], "track_concat")
    model.CosineSimilarity(["track_fc_one_tile", "track_fc_two_repeat"], "track_similarity")

    # if model.train:
    track_similarity = model.ExpandDims("track_similarity", "track_similarity_", dims=[0])
    # model.Sigmoid('track_similarity', 'track_prob')
    # track_prob = model.ExpandDims("_track_prob", "track_prob", dims=[0])
    # else:
    #     model.Softmax('track_similarity', 'track_prob', engine='CUDNN')

    # model.ExpandDims("track_similarity", "track_similarity", dims=[1])


    # model.track_rec_net.Copy("track_fc", "track_fc_two")

    # return track_concat, hidden_dim * 2

    #     # Only add softmax when testing; during training the softmax is combined
    #     # with the label cross entropy loss for numerical stability
    #     model.Softmax('track_similarity', 'track_prob', engine='CUDNN')

    return track_similarity


def add_track_losses(model):
    model.StopGradient("track_int32", "track_int32")
    if cfg.TRCNN.LOSS == 'Cosine':
        model.CosineSimilarity(["track_similarity_", "track_int32"], "track_cosine_similarity")
        model.Negative("track_cosine_similarity", "track_cosine_similarity_neg")
        model.ConstantFill([], "ONE", value=1.0)
        model.Add(["track_cosine_similarity_neg", "ONE"], "loss_track_raw")
    elif cfg.TRCNN.LOSS == 'L2':
        model.SquaredL2Distance(["track_similarity_", "track_int32"], "track_l2_loss")
        model.Length("track_similarity", "track_n_pairs")
        model.Cast("track_n_pairs", "track_n_pairs_float", to=1) # FLOAT=1
        model.StopGradient("track_n_pairs_float", "track_n_pairs_float")
        model.Div(["track_l2_loss", "track_n_pairs_float"], "loss_track_raw")
    else:
        raise

    loss_track = model.Scale("loss_track_raw", "loss_track", scale=cfg.TRCNN.LOSS_WEIGHT)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_track])
    model.AddLosses(['loss_track'])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Tracking heads
# ---------------------------------------------------------------------------- #

def add_track_head(model, blob_in, dim_in, spatial_scale):
    """Add a Mask R-CNN track head."""
    hidden_dim = cfg.TRCNN.MLP_HEAD_DIM
    roi_size = cfg.TRCNN.ROI_XFORM_RESOLUTION
    n_rois = cfg.TRAIN.RPN_POST_NMS_TOP_N if model.train else cfg.TEST.RPN_POST_NMS_TOP_N
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'track_roi_feat',
        blob_rois='track_rois',
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
    track_fc = model.Relu("track_fc", "track_fc")

    return track_fc, hidden_dim





# def _init_track_fc_two_blob(model, n_rois, hidden_dim):
#     matrix = np.zeros((n_rois, hidden_dim))
#     return model.GivenTensorFill([], "track_fc_two", values=matrix, shape=[n_rois, hidden_dim])
