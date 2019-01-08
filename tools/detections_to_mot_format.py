#!/usr/bin/env python

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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import numpy as np

from detectron.core.config import cfg
from detectron.datasets import task_evaluation
from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
from detectron.utils.logging import setup_logging
from detectron.utils.tracking import Tracking
import detectron.core.config as core_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'detections',   help='detections path', type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for bounding boxes',
        default=0.7,
        type=float
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def convert_from_cls_format(cls_boxes):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, classes

def main(args):
    # dataset = JsonDataset(dataset_name)
    dets = load_object(args.detections)
    
    all_boxes = dets['all_boxes']
    all_track = dets['all_track']
    results = []
    obj_cnt = 0

    tracking = Tracking(args.thresh)

    result_str = "{frame}, {obj_id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {conf}, {x}, {y}, {z}"

    all_boxes = np.array(all_boxes)
    all_boxes = [convert_from_cls_format(all_boxes[:, i]) for i in xrange(all_boxes.shape[1])]

    for i, cls_boxes in enumerate(all_boxes):
        keep_idx = []
        boxes, classes = cls_boxes
        
        # filter out detectios with low bounding box detection threshold
        for j, cls_box in enumerate(boxes):
            if cls_box[-1] >= args.thresh:
                keep_idx.append(j)
        if i > 0:
            tracking.accumulate(boxes, all_track[i])
        else:
            tracking.accumulate(boxes)


       
        # assign_inds_prev, assign_inds = linear_sum_assignment(-track).tolist()
        
        # for idx in keep_idx:
        #     i_assign = assign_inds.index(idx)
        #     idx_prev = assign_inds_prev[i_assign]
        #     if idx_prev not in keep_idx_prev:
        #         continue

        #     obj_cnt += 1

        #     bbox = boxes[idx][:4]
        #     bbox_width = bbox[2] - bbox[0]
        #     bbox_height = bbox[3] - bbox[1]
        #     results.append(result_str.format(frame=i, obj_id=, bb_left=bbox[0], bb_top=bbox[1] + bbox_height, bb_width=bbox_width, bb_height=bbox_height, conf=-1, x=-1, y=-1, z=-1))
        # if i == 0:
        #     continue
    # for line in tracking.get_all_associations(False):
    for line in tracking.get_all_associations(True):
        print(line)

if __name__ == '__main__':
    setup_logging(__name__)
    args = parse_args()
    main(args)
