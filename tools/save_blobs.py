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
from collections import deque
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
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
from detectron.modeling import model_builder
from detectron.utils.train import optimize_memory, add_model_training_inputs
import detectron.utils.net as nu
from detectron.roi_data.loader import RoIDataLoader
from detectron.datasets.roidb import combined_roidb_for_training
import detectron.roi_data.minibatch as roi_data_minibatch

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
        dest='weights',
        help='list of weights model files (/path/to/model_weights.pkl)',
        type=str,
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
        '--datasets',
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--ds-len',
        dest="ds_len",
        type=int,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

class RoIDataLoaderSimple(RoIDataLoader):
    def _shuffle_roidb_inds(self):
        self._perm = np.arange(len(self._roidb))
        self._perm = deque(self._perm)
        self._cur = 0

def create_model(weights_file):
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    if cfg.MEMONGER:
        optimize_memory(model)
    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)

    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    # To make debugging easier you can set cfg.DATA_LOADER.NUM_THREADS = 1
    model.roi_data_loader = RoIDataLoaderSimple(
        roidb,
        num_loaders=cfg.DATA_LOADER.NUM_THREADS,
        minibatch_queue_size=cfg.DATA_LOADER.MINIBATCH_QUEUE_SIZE,
        blobs_queue_capacity=cfg.DATA_LOADER.BLOBS_QUEUE_CAPACITY
    )
    orig_num_op = len(model.net._net.op)
    blob_names = roi_data_minibatch.get_minibatch_blob_names(is_training=True)
    with c2_utils.NamedCudaScope(0):
        for blob_name in blob_names:
            workspace.CreateBlob(core.ScopedName(blob_name))
        model.net.DequeueBlobs(
            model.roi_data_loader._blobs_queue_name, blob_names
        )
    # A little op surgery to move input ops to the start of the net
    diff = len(model.net._net.op) - orig_num_op
    new_op = model.net._net.op[-diff:] + model.net._net.op[:-diff]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)

    nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    nu.broadcast_parameters(model)

    workspace.CreateBlob("gpu_0/track_n_rois_two")
    workspace.CreateNet(model.net)

    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    model.roi_data_loader.start(prefill=True)
    return model

def main(args):
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    if len(args.datasets):
        cfg.TRAIN.DATASETS = tuple(args.datasets)
    assert_and_infer_cfg(cache_urls=False)
    cfg.immutable(False)
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.immutable(True)

    model = create_model(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    for i in xrange(3):
        logger.info("Processing {}".format(i +))
        with c2_utils.NamedCudaScope(0):
            try:
                workspace.RunNet(model.net.Proto().name)
            except:
                pass
            blobs = [(blob_name, workspace.FetchBlob(core.ScopedName(blob_name))) for blob_name in args.blobs]
        invalid_blobs = [blob for blob in blobs if not hasattr(blob[1], 'shape')]
        assert len(invalid_blobs) == 0, "Blobs not found: {}".format([blob[0] for blob in invalid_blobs])
        save_file = os.path.join(args.output_dir, str(i) + ".npz")
        logger.info("Saving to {}".format(save_file))
        to_save = [blob[1] for blob in blobs]
        np.savez(open(save_file, "w+"), *to_save)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
