"""Script to merge weights from multiple weights files"""

import logging
import argparse
import cv2
import os, sys

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.modeling import model_builder
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import detectron.utils.net as nu
from detectron.utils.io import save_object

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights_list',
        help='list of weights model files (/path/to/model_weights.pkl)',
        default=None,
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
        '--output-wts',
        dest='output_wts',
        default='outputs/model_merged.pkl',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    for i, weights_file in enumerate(args.weights_list):
        args.weights_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    preffix_list = args.preffix_list if len(args.preffix_list) \
        else [""] * len(args.weights_list)
    model = model_builder.create(cfg.MODEL.TYPE, train=False)
    # Initialize GPU from weights files
    for i, weights_file in enumerate(args.weights_list):
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0,
            preffix=preffix_list[i])
    nu.broadcast_parameters(model)
    blobs = {}
    # Save all parameters
    for param in model.params:
        scoped_name = str(param)
        unscoped_name = c2_utils.UnscopeName(scoped_name)
        if unscoped_name not in blobs:
            if workspace.HasBlob(scoped_name):
                blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save merged weights file
    save_object(dict(blobs=blobs), args.output_wts)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
