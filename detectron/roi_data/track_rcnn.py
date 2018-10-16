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

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils


def add_track_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Track R-CNN specific blobs to the input blob dictionary."""
    # Prepare the track targets by associating one gt track to each training roi
    # that has a fg (non-bg) class label.
    max_track = roidb['track_ids'][roidb['box_to_gt_ind_map']]

    sampled_rois = roidb['boxes']

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Update blobs dict with Track R-CNN blobs
    blobs['track_rois'] = sampled_rois
    blobs['track_ids_int32'] = max_track
    blobs['track_n_rois'] = np.array([len(sampled_rois)], dtype=np.int32)

    if cfg.TRAIN.DEBUG:
        import cv2 
        from detectron.utils.vis import vis_bbox, vis_class, distinct_colors
        import matplotlib.pyplot as plt

        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        rpn_inds = np.where(roidb['gt_classes'] == 0)[0]

        boxes = roidb['boxes'] / im_scale
        boxes_gt = boxes[gt_inds]
        boxes_rpn = boxes[rpn_inds]
        max_track_gt = max_track[gt_inds]
        max_track_rpn = max_track[rpn_inds]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas_gt = (boxes_gt[:, 2] - boxes_gt[:, 0]) * (boxes_gt[:, 3] - boxes_gt[:, 1])
        areas_rpn = (boxes_rpn[:, 2] - boxes_rpn[:, 0]) * (boxes_rpn[:, 3] - boxes_rpn[:, 1])
        sorted_inds = np.argsort(-areas)
        sorted_inds_gt = np.argsort(-areas_gt)
        sorted_inds_rpn = np.argsort(-areas_rpn)

        n_ids = len(max_track_gt)
        colors = distinct_colors(n_ids)

        
        im = cv2.imread(roidb['image'])
        im_save = im.copy()
        im_save_gt = im.copy()
        im_save_rpn = im.copy()

        for i in sorted_inds:
            bbox = boxes[i, :4]
            track_id = max_track[i]

            im_save = vis_bbox(
                im_save, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), color=colors[max_track.tolist().index(track_id)])
            im_save = vis_class(im_save, (bbox[0], bbox[1] - 2), str(track_id))

        for i in sorted_inds_gt:
            bbox = boxes_gt[i, :4]
            track_id = max_track_gt[i]

            im_save_gt = vis_bbox(
                im_save_gt, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), color=colors[max_track_gt.tolist().index(track_id)])
            im_save_gt = vis_class(im_save_gt, (bbox[0], bbox[1] - 2), str(track_id))

        for i in sorted_inds_rpn:
            bbox = boxes_rpn[i, :4]
            track_id = max_track_rpn[i]

            im_save_rpn = vis_bbox(
                im_save_rpn, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), color=colors[max_track_gt.tolist().index(track_id)])
            im_save_rpn = vis_class(im_save_rpn, (bbox[0], bbox[1] - 2), str(track_id))

        img_id = roidb['id']

        cv2.imwrite("train_track_rois_{}.png".format(img_id), im_save)
        cv2.imwrite("train_track_{}.png".format(img_id), im_save_gt)
        cv2.imwrite("train_track_rpn_{}.png".format(img_id), im_save_rpn)


def finalize_track_minibatch(blobs):
    blobs['track_n_rois_one'] = np.array([blobs['track_n_rois'][0]], dtype=np.int32)
    blobs['track_n_rois_two'] = np.array([blobs['track_n_rois'][1]], dtype=np.int32)
