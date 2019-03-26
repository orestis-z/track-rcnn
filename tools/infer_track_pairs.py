"""Script for object association inference for image pairs"""

import logging
import argparse
import cv2
import os, sys
import numpy as np
import pprint

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.tracking import Tracking
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

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
        help='directory for visualization pdfs (default: outputs/infer_simple)',
        default='outputs/infer_track_pairs',
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
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--im-dir',
        dest='im_dir',
        help='Folder of images',
        default=None
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    im_names = os.listdir(args.im_dir)
    assert len(im_names) > 1, "Sequence must contain > 1 images"
    im_names.sort()
    im_paths = [os.path.join(args.im_dir, im_name) for im_name in im_names]
    im_names = [im_name.split(".")[0] for im_name in im_names]

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    if "mot-classes" in args.opt:
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

    # Iterate through the image sequence
    for path_i, im_path in enumerate(im_paths[:-2]):
        tracking = Tracking(args.thresh)
        im_list = [cv2.imread(im_path), cv2.imread(im_paths[path_i + 1])]
        with c2_utils.NamedCudaScope(0):
            print("Processing {}, {}, {}".format(args.output_dir,
                im_names[path_i], im_names[path_i + 1]))
            # Image pair detections
            cls_boxes_list, cls_segms_list, cls_keyps_list, track_mat_i, _ = \
                infer_engine.multi_im_detect_all(model, im_list, [None, None])
            tracking.accumulate(cls_boxes_list[0], cls_segms_list[0],
                cls_keyps_list[0])
            tracking.accumulate(cls_boxes_list[1], cls_segms_list[1],
                cls_keyps_list[1], track_mat_i)
        # Visualize image pair associations
        for i in xrange(2):
            im_list[i] = vis_utils.vis_detections_one_image_opencv(
                im_list[i],
                detections=tracking.detection_list[i],
                detections_prev=tracking.detection_list[0] if i == 1 else [],
                dataset=dummy_mot_dataset,
                show_class=('show-class' in args.opts),
                show_track=('show-track' in args.opts),
                show_box=True,
                thresh=args.thresh,
                kp_thresh=args.kp_thresh,
                track_thresh=args.track_thresh,
            )
        # Merge image pair with visualizations
        im_pred = np.vstack(im_list)
        cv2.imwrite("{}/{}-{}_pred.png".format(args.output_dir,
            im_names[path_i], im_names[path_i + 1]), im_pred)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
