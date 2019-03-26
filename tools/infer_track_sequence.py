"""Script to infer tracking on an image sequence

Configurable to output visualizations and to store detection to a file.
"""

import logging
import argparse
import cv2
import os, sys
import pprint
import numpy as np
import pickle

from caffe2.python import workspace
from caffe2.python import core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.utils.tracking import Tracking, infer_track_sequence

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
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs '
             '(default: outputs/infer_track_sequence)',
        default='outputs/infer_track_sequence',
        type=str
    )
    parser.add_argument(
        '--output-file',
        dest='output_file',
        help='file for detections (default: outputs/infer_track_sequence/'
             'detections.txt)',
        default='outputs/infer_track_sequence/detections.txt',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--track-thresh',
        dest='track_thresh',
        help='Threshold for visualizing matches',
        default=cfg.TRCNN.DETECTION_THRESH,
        type=float
    )
    parser.add_argument(
        '--im-dir',
        dest='im_dir',
        help='image or folder of images',
        default=None
    )
    parser.add_argument(
        '--n-colors',
        dest='n_colors',
        default=10,
        type=int
    )
    parser.add_argument(
        '--proposals',
        help='path to proposals'
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        default=[],
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    logger = logging.getLogger(__name__)
  
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    if "mot-classes" in args.opts:
        dummy_dataset = dummy_datasets.get_mot_dataset()
        cfg.NUM_CLASSES = 14
    else:
        dummy_dataset = dummy_datasets.get_coco_dataset()
        cfg.NUM_CLASSES = 81
    for i, weights_file in enumerate(args.weights_list):
        args.weights_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    preffix_list = args.preffix_list if len(args.preffix_list) \
        else [""] * len(args.weights_list)
    model = infer_engine.initialize_mixed_model_from_cfg(args.weights_list,
        preffix_list=preffix_list)
    # Initialize tracking accumulator
    tracking = Tracking(args.thresh, cfg.TRCNN.MAX_BACK_TRACK)
    vis = {
        "output-dir": args.output_dir,
        "dummy-dataset": dummy_dataset,
        "show-class": "show-class" in args.opts,
        "show-track": "show-track" in args.opts,
        "thresh": args.thresh,
        "kp-thresh": args.kp_thresh,
        "track-thresh": args.track_thresh,
        "n-colors": args.n_colors,
    }
    # Load proposals if specified
    if args.proposals is not None:
        proposals = pickle.load(open(args.proposals, 'r'))
    else:
        proposals = None
    # Run inference
    infer_track_sequence(model, args.im_dir, tracking, vis=vis,
        det_file=args.output_file, proposals=proposals,
        mot=("all-dets" not in args.opts))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
