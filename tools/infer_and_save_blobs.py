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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import numpy as np

from caffe2.python import workspace
from caffe2.python import core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights_list',
        help='list of weights model files (/path/to/model_weights.pkl)',
        default=[],
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--preffixes',
        dest='preffix_list',
        help='preffixes for the corresponding weights file',
        default=[],
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        default='outputs/blobs',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--blobs',
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--im-dir',
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    for i, weights_file in enumerate(args.weights_list):
        args.weights_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    preffix_list = args.preffix_list if len(args.preffix_list) else [""] * len(args.weights_list)
    model = infer_engine.initialize_mixed_model_from_cfg(args.weights_list, preffix_list=preffix_list)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_dir):
        im_list = glob.iglob(args.im_dir + '/*.' + args.image_ext)
    else:
        im_list = [args.im_dir]

    for i, im_name in enumerate(im_list):
        logger.info("Processing {}".format(im_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            try:
                infer_engine.multi_im_detect_all(
                    model, [im], [None], timers=timers, tracking=True
                )
            except:
                pass
            blobs = [(blob_name, workspace.FetchBlob(core.ScopedName(blob_name))) for blob_name in args.blobs]
        invalid_blobs = [blob for blob in blobs if not hasattr(blob[1], 'shape')]
        assert len(invalid_blobs) == 0, "Blobs not found: {}".format([blob[0] for blob in invalid_blobs])
        save_file = os.path.join(args.output_dir, im_name.split("/")[-1].split(".")[0] + ".npz")
        logger.info("Saving to {}".format(save_file))
        # pickle.dump(blobs, open(save_file, "w+"))
        np.savez(open(save_file, "w+"), *[blob[1] for blob in blobs])

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
