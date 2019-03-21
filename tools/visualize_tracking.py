"""Script to visuallize tracking given a detection file
"""

import logging
import argparse
import cv2
import os, sys
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.utils.tracking import Tracking

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: outputs/infer_track_sequence)',
        default='outputs/infer_track_sequence',
        type=str
    )
    parser.add_argument(
        '--detections-file',
        dest='detections_file',
        help='file for detections (default: outputs/infer_track_sequence/detections.txt)',
        type=str
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
    
    def parse(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    detections_list_raw = []
    with open(args.detections_file, "r") as f:
        for line in f:
            detections_list_raw.append([parse(s) for s in line.split(",")])

    obj_ids = np.unique([detection[1] for detection in detections_list_raw]).tolist()
    frames = np.unique([detection[0] for detection in detections_list_raw]).tolist()
    assert len(frames) == len(im_paths)
    n_ids = len(obj_ids)

    dummy_mot_dataset = dummy_datasets.get_mot_dataset()

    colors_temp = vis_utils.distinct_colors(n_ids)
    colors = [colors_temp[obj_ids.index(obj_id)] if obj_id in obj_ids else None for obj_id in xrange(obj_ids[-1] + 1)]

    detections_list = [[] for _ in xrange(len(frames))]
    for detection in detections_list_raw:
        frame = detection[0] - 1
        detections_list[frame].append(detection)

    for i, im_path in enumerate(im_paths):
        print("Processing {}".format(im_path))
        im = cv2.imread(im_path)
        im = vis_utils.vis_tracking_one_image_opencv(
            im,
            detections=detections_list[i],
            dataset=dummy_mot_dataset,
            show_class=('show-class' in args.opts),
            colors=colors,
        )
        cv2.imwrite("{}/{}_pred.png".format(args.output_dir, im_names[i]), im)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
