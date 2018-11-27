import logging
import argparse
import cv2
import os, sys
import numpy as np

from caffe2.python import workspace

from detectron.core.config import cfg, assert_and_infer_cfg, merge_cfg_from_file
from detectron.utils.logging import setup_logging
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
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
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
        help='image or folder of images',
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
    
    dummy_mot_dataset = dummy_datasets.get_mot_dataset()
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)

    for i, im_path in enumerate(im_paths[:-2]):
        im_one = cv2.imread(im_path)
        im_two = cv2.imread(im_paths[i+1])
        with c2_utils.NamedCudaScope(0):
            print("Processing {}, {}, {}".format(args.output_dir, im_names[i], im_names[i + 1]))
            cls_boxes_list, cls_segms_list, cls_keyps_list, cls_track, _, _, _, _ = infer_engine.multi_im_detect_all(
            model, [im_one, im_two], [None, None])
        im_pred_one, im_pred_two, _, _, _ = vis_utils.vis_image_pair_opencv(
            [im_one, im_two],
            cls_boxes_list,
            cls_segms_list,
            cls_keyps_list,
            cls_track,
            dataset=dummy_mot_dataset,
            show_class=('show_class' in args.opts),
            show_track=True,
            show_box=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            track_thresh=args.track_thresh,
        )
        im_pred = np.vstack((im_pred_one, im_pred_two))
        cv2.imwrite("{}/{}-{}_pred.png".format(args.output_dir, im_names[i], im_names[i + 1]), im_pred)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
