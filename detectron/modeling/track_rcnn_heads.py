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
from caffe2.proto import caffe2_pb2

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Tracking outputs and losses
# ---------------------------------------------------------------------------- #

def add_track_outputs(model, blob_in, dim):    
    model.EnsureCPUOutput("track_n_rois", "track_n_rois_cpu")
    
    if model.train:
        model.Split(["track_ids_int32", "track_n_rois_cpu"], ["track_ids_one_int32", "track_ids_two_int32"], axis=0)
        model.GenerateTrackingLabels(["track_ids_one_int32", "track_ids_two_int32"], "track_int32")

    model.Split([blob_in, "track_n_rois_cpu"], ["track_fc_one", "track_fc_two"], axis=0)
    
    repeat_outputs = ["track_fc_one_repeat"]
    if model.train:
        repeat_outputs.append("track_fc_one_repeat_lengths")
    model.Repeat(["track_fc_one", "track_n_rois_two"], repeat_outputs) # (n_pairs, mlp_dim)

    model.Tile(["track_fc_two", "track_n_rois_one"], "track_fc_two_tile", axis=0) # (n_pairs, mlp_dim)

    if cfg.TRCNN.OUTPUTS == 'Cosine':
        model.CosineSimilarity(["track_fc_one_repeat", "track_fc_two_tile"], "track_cos_similarity") # (n_pairs,)
     
        blob_out = model.ExpandDims("track_cos_similarity", "track_similarity", dims=[0]) # (1, n_pairs)

    elif cfg.TRCNN.OUTPUTS == 'MatchNet':
        hidden_dim = cfg.TRCNN.MLP_HIDDEN_DIM
        model.Concat(["track_fc_one_repeat", "track_fc_two_tile"], "track_pairs")
        model.FC(
            "track_pairs",
            "track_pairs_fc1",
            2 * dim,
            hidden_dim,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )
        model.Relu("track_pairs_fc1", "track_pairs_fc1")
        model.FC(
            "track_pairs_fc1",
            "track_pairs_fc2",
            hidden_dim,
            hidden_dim,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )
        model.Relu("track_pairs_fc2", "track_pairs_fc2")
        blob_out = model.FC(
            "track_pairs_fc2",
            "track_score",
            hidden_dim,
            2,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax("track_score", "track_prob", axis=1, engine='CUDNN')
            model.Slice("track_prob", "track_similarity_", starts=[0, 1], ends=[-1, -1])
            blob_out = model.Transpose("track_similarity_", "track_similarity")

    return blob_out



def add_track_losses(model):
    model.ConstantFill([], "ONE_int32", value=1, dtype=caffe2_pb2.TensorProto.INT32, shape=(1,))
    model.ConstantFill([], "ZERO_int32", value=0, dtype=caffe2_pb2.TensorProto.INT32, shape=(1,))
    model.ConstantFill([], "ONE_float32", value=1.0)

    model.StopGradient("track_int32", "track_int32")

    if cfg.TRCNN.LOSS == 'Cosine': # L_cos
        model.CosineSimilarity(["track_similarity", "track_int32"], "track_cosine_similarity")
        model.Negative("track_cosine_similarity", "track_cosine_similarity_neg")
        model.Add(["track_cosine_similarity_neg", "ONE"], "loss_track_raw")
    elif cfg.TRCNN.LOSS == 'L2': # L2sq / n_pairs
        model.SquaredL2Distance(["track_similarity", "track_int32"], "track_l2_loss")
        model.Shape("track_similarity", "track_n_pairs", axes=[1])
        model.Cast("track_n_pairs", "track_n_pairs_float", to=1) # FLOAT=1
        model.StopGradient("track_n_pairs_float", "track_n_pairs_float")
        model.Div(["track_l2_loss", "track_n_pairs_float"], "loss_track_raw")
    elif cfg.TRCNN.LOSS == 'L2Balanced': # 0.5 * (L2sq_match / n_match_pairs + L2sq_nomatch / n_nomatch_pairs)
        model.ExpandDims("track_int32", "track_int32_", dims=[0])
        b = model.Cast("track_int32_", "track_float32", to=1)
        model.StopGradient(b, b)
        model.Negative("track_float32", "track_float32_neg")
        b = model.Add(["track_float32_neg", "ONE"], "track_float32_nomatch")
        model.StopGradient(b, b)
        model.Sub(["track_similarity", "track_float32"], "track_delta")
        model.Sqr("track_delta", "track_delta_sq")
        model.DotProduct(["track_delta_sq", "track_int32"], "loss_track_match_raw")
        model.DotProduct(["track_delta_sq", "track_float32_nomatch"], "loss_track_nomatch_raw")
        b = model.SumElements("track_float32", "track_n_match")
        model.StopGradient(b, b)
        b = model.SumElements("track_float32_nomatch", "track_n_nomatch")
        model.StopGradient(b, b)
        model.Div(["loss_track_match_raw", "track_n_match"], "loss_track_match")
        model.Div(["loss_track_nomatch_raw", "track_n_nomatch"], "loss_track_nomatch")
        model.Add(["loss_track_match", "loss_track_nomatch"], "loss_track_sum")
        model.Scale("loss_track_sum", "loss_track_raw", scale=0.5)
    elif cfg.TRCNN.LOSS == 'CrossEntropy':
        # model.Transpose("track_int32", "track_match_int32")
        # model.Sub(["ONE", "track_match_int32"], "track_nomatch_int32")
        # model.Concat(["track_match_int32", "track_nomatch_int32"], "track_labels_int32")
        model.SoftmaxWithLoss(['track_score', 'track_int32'], ['track_prob', 'loss_track_raw'])
    elif cfg.TRCNN.LOSS == 'CrossEntropyBalanced':
        # model.Transpose("track_int32", "track_match_int32")
        # model.Sub(["ONE", "track_match_int32"], "track_nomatch_int32")
        # model.Concat(["track_match_int32", "track_nomatch_int32"], "track_labels_int32")
        model.Sub(["ONE_int32", "track_int32"], "track_nomatch_int32")
        model.LengthsTile(["track_score", "track_int32"], "track_score_match")
        model.LengthsTile(["track_score", "track_nomatch_int32"], "track_score_nomatch")
        model.SumElementsInt("track_int32", "track_n_match_")
        model.SumElementsInt("track_nomatch_int32", "track_n_nomatch_")
        model.ExpandDims("track_n_match_", "track_n_match", dims=[0])
        model.ExpandDims("track_n_nomatch_", "track_n_nomatch", dims=[0])
        model.Tile(['ONE_int32', "track_n_match"], "ONEs_int32", axis=0)
        model.Tile(['ZERO_int32', "track_n_nomatch"], "ZEROs_int32", axis=0)
        _, loss_track_match = model.SoftmaxWithLoss(['track_score_match', 'ONEs_int32'], ['track_prob_match', 'loss_track_match'])
        _, loss_track_nomatch = model.SoftmaxWithLoss(['track_score_nomatch', 'ZEROs_int32'], ['track_prob_nomatch', 'loss_track_nomatch'])
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_track_match, loss_track_nomatch])
        model.AddLosses(['loss_track_match', 'loss_track_nomatch'])
        return loss_gradients
    elif cfg.TRCNN.LOSS == 'CrossEntropyWeighted':
        model.Sub(["ONE_int32", "track_int32"], "track_nomatch_int32")
        model.SumElementsInt("track_int32", "track_n_match_")
        model.SumElementsInt("track_nomatch_int32", "track_n_nomatch_")
        model.ExpandDims("track_n_match_", "track_n_match", dims=[0, 1])
        model.ExpandDims("track_n_nomatch_", "track_n_nomatch", dims=[0, 1])
        model.Cast("track_n_match", "track_n_match_float32", to=1)
        model.Cast("track_n_nomatch", "track_n_nomatch_float32", to=1)
        model.Sum(["track_n_match_float32", "track_n_nomatch_float32"], "track_n_tot")
        model.Concat(["track_n_match_float32", "track_n_nomatch_float32"], "track_loss_weights_inv")
        model.Div(["ONE_float32", "track_loss_weights_inv"], "track_loss_weights_")
        model.Mul(["track_loss_weights_", "track_n_tot"], "track_loss_weights")

        model.SoftmaxWithLoss(['track_score', 'track_int32', "track_loss_weights"], ['track_prob', 'loss_track_raw'])
    else:
        raise ValueError

    loss_track = model.Scale("loss_track_raw", "loss_track", scale=cfg.TRCNN.LOSS_WEIGHT)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_track])
    model.AddLosses(['loss_track'])
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Tracking heads
# ---------------------------------------------------------------------------- #

def add_track_head(model, blob_in, dim_in, spatial_scale):
    """Add a Mask R-CNN track head."""
    head_dim = cfg.TRCNN.MLP_HEAD_DIM
    roi_size = cfg.TRCNN.ROI_XFORM_RESOLUTION
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
        head_dim,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0),
    )
    track_fc = model.Relu("track_fc", "track_fc")

    return track_fc, head_dim
