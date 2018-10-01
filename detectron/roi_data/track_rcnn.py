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

"""Construct minibatches for Track R-CNN training. Handles the minibatch blobs
that are specific to Track R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils


def add_track_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Track R-CNN specific blobs to the input blob dictionary."""
    # Prepare the track targets by associating one gt track to each training roi
    # that has a fg (non-bg) class label.
    track_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    track_gt = np.array([roidb['track_ids'][i] for i in track_gt_inds])
    boxes_gt = np.array([roidb['boxes'][i] for i in track_gt_inds])
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_is_fg = blobs['labels_int32'].copy()
    roi_is_fg[roi_is_fg > 0] = 1
    rois_fg = sampled_boxes[fg_inds]

    overlaps_bbfg_bbgt = box_utils.bbox_overlaps(
        rois_fg.astype(np.float32, copy=False),
        boxes_gt.astype(np.float32, copy=False)
    )
    # Map from each fg rois to the index of the bbox with highest overlap
    # (measured by bbox overlap)
    fg_gt_inds = np.argmax(overlaps_bbfg_bbgt, axis=1)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Track R-CNN blobs
    blobs['track_rois'] = rois_fg
    blobs['track_ids_int32'] = np.array(track_gt)[fg_gt_inds]
    blobs['track_n_rois'] = np.array([len(rois_fg)], dtype=np.int32)
