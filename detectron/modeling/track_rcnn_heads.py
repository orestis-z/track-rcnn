"""
Tracking head
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


# ------------------------------------------------------------------ #
# Tracking outputs and losses
# ------------------------------------------------------------------ #

def add_track_outputs(model, blob_in, dim):    
    model.EnsureCPUOutput("track_n_rois", "track_n_rois_cpu")
    
    if model.train:
        model.Split(["track_ids_int32", "track_n_rois_cpu"],
            ["track_ids_one_int32", "track_ids_two_int32"], axis=0)
        model.GenerateTrackingLabels(["track_ids_one_int32",
            "track_ids_two_int32"], "track_int32")

    model.Split([blob_in, "track_n_rois_cpu"],
        ["track_fc_one", "track_fc_two"], axis=0)
    
    repeat_outputs = ["track_fc_one_repeat"]
    if model.train:
        repeat_outputs.append("track_fc_one_repeat_lengths")
    model.Repeat(["track_fc_one", "track_n_rois_two"], repeat_outputs) # (n_pairs, mlp_dim)

    model.Tile(["track_fc_two", "track_n_rois_one"],
    "track_fc_two_tile", axis=0) # (n_pairs, mlp_dim)

    # Cosine tracking head architecture
    if cfg.TRCNN.OUTPUT == 'Cosine':
        model.CosineSimilarity(["track_fc_one_repeat",
        "track_fc_two_tile"], "track_cos_similarity") # (n_pairs,)
     
        blob_out = model.ExpandDims("track_cos_similarity",
        "track_similarity", dims=[0]) # (1, n_pairs)
    # MatchNet tracking head architecture
    elif cfg.TRCNN.OUTPUT == 'MatchNet':
        hidden_dim = cfg.TRCNN.MLP_HIDDEN_DIM
        model.Concat(["track_fc_one_repeat", "track_fc_two_tile"],
            "track_pairs")
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

    # L_cos (loss based on cosine similarity)
    if cfg.TRCNN.LOSS == 'Cosine':
        model.CosineSimilarity(["track_similarity", "track_int32"],
            "track_cosine_similarity")
        model.Negative("track_cosine_similarity", "track_cosine_similarity_neg")
        model.Add(["track_cosine_similarity_neg", "ONE_float32"],
            "loss_track_raw")
    # L2sq / n_pairs (loss based on normalized squared L2)
    elif cfg.TRCNN.LOSS == 'L2':
        model.SquaredL2Distance(["track_similarity", "track_int32"],
            "track_l2_loss")
        model.Shape("track_similarity", "track_n_pairs", axes=[1])
        model.Cast("track_n_pairs", "track_n_pairs_float", to=1) # FLOAT=1
        model.StopGradient("track_n_pairs_float",
            "track_n_pairs_float")
        model.Div(["track_l2_loss", "track_n_pairs_float"],
            "loss_track_raw")
    # 0.5 * (L2sq_match / n_match_pairs + L2sq_nomatch /
    # n_nomatch_pairs) (Normalized squared L2 loss which additionally)
    # ensures balanced matching and non-matching pairs)
    elif cfg.TRCNN.LOSS == 'L2Balanced':
        b = model.Cast("track_int32", "track_float32", to=1)
        model.StopGradient(b, b)
        # Create non-match ground truth vector
        model.Negative("track_float32", "track_float32_neg")
        b = model.Add(["track_float32_neg", "ONE_float32"], "track_float32_nomatch")
        model.StopGradient(b, b)
        # Linear error vector between predicted similarities and 
        # ground-truth
        model.Sub(["track_similarity", "track_float32"], "track_delta")
        # Squared error vector
        model.Sqr("track_delta", "track_delta_sq")
        # Squared match error
        model.DotProduct(["track_delta_sq", "track_int32"], "loss_track_match_raw")
        # Squared non-match error
        model.DotProduct(["track_delta_sq", "track_float32_nomatch"], "loss_track_nomatch_raw")
        b = model.SumElements("track_float32", "track_n_match")
        model.StopGradient(b, b)
        b = model.SumElements("track_float32_nomatch", "track_n_nomatch")
        model.StopGradient(b, b)
        # Normalize match and non-match errors
        model.Div(["loss_track_match_raw", "track_n_match"], "loss_track_match")
        model.Div(["loss_track_nomatch_raw", "track_n_nomatch"], "loss_track_nomatch")
        model.Add(["loss_track_match", "loss_track_nomatch"], "loss_track_sum")
        model.Scale("loss_track_sum", "loss_track_raw", scale=0.5)
    # Cross entropy loss
    elif cfg.TRCNN.LOSS == 'CrossEntropy':
        model.SoftmaxWithLoss(['track_score', 'track_int32'], ['track_prob', 'loss_track_raw'])
    # Balanced cross entropy loss (cross entropy loss which 
    # additionally ensures balanced matching and non-matching pairs)
    elif cfg.TRCNN.LOSS == 'CrossEntropyBalanced':
        model.Sub(["ONE_int32", "track_int32"], "track_nomatch_int32")
        # Extract ground-truth matches
        model.LengthsTile(["track_score", "track_int32"],
            "track_score_match")
        # Extract ground-truth non-matches
        model.LengthsTile(["track_score", "track_nomatch_int32"],
            "track_score_nomatch")
        # Extract number of ground-truth matches
        model.SumElementsInt("track_int32", "track_n_match_")
        # Extract number of ground-truth non-matches
        model.SumElementsInt("track_nomatch_int32",
            "track_n_nomatch_")
        model.ExpandDims("track_n_match_", "track_n_match", dims=[0])
        model.ExpandDims("track_n_nomatch_", "track_n_nomatch",
            dims=[0])
        # Create ground-truth matches vector
        model.Tile(['ONE_int32', "track_n_match"], "ONEs_int32", axis=0)
        # Create ground-truth non-matches vector
        model.Tile(['ZERO_int32', "track_n_nomatch"], "ZEROs_int32",
            axis=0)
        _, loss_track_match = model.SoftmaxWithLoss(['track_score_match', 'ONEs_int32'],
            ['track_prob_match', 'loss_track_match'])
        _, loss_track_nomatch = model.SoftmaxWithLoss(['track_score_nomatch', 'ZEROs_int32'],
            ['track_prob_nomatch', 'loss_track_nomatch'])
        model.Sum(['loss_track_match', 'loss_track_nomatch'],
            "loss_track_raw")
    else:
        raise ValueError

    loss_track = model.Scale("loss_track_raw", "loss_track", scale=cfg.TRCNN.LOSS_WEIGHT)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_track])
    model.AddLosses(['loss_track'])
    return loss_gradients


# ------------------------------------------------------------------ #
# Tracking heads
# ------------------------------------------------------------------ #

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
    # Bottleneck operation
    if cfg.TRCNN.MLP_HEAD_ON:
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
    # No bottleneck operation -> flattern feature vector
    else:
        model.Flatten(roi_feat, "track_fc")
        track_fc = model.Relu("track_fc", "track_fc")
        return roi_feat, dim_in * roi_size * roi_size
