import logging
import argparse
import cv2
import os, sys
import pprint
import numpy as np
import re

from caffe2.python import workspace
from caffe2.python import core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import get_output_dir
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
from detectron.datasets.dataset_catalog import get_im_dir
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.utils.tracking import Tracking, back_track, infer_track_sequence, get_matlab_engine, eval_detections_matlab

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
        '--wts-pre',
        dest='weights_pre_list',
        help='list of extra pre weights model files (/path/to/model_weights.pkl)',
        default=[],
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--wts-post',
        dest='weights_post_list',
        help='list of extra post weights model files (/path/to/model_weights.pkl)',
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
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--devkit-path',
        dest='devkit_path',
        default='~/repositories/motchallenge-devkit',
        type=str
    )
    parser.add_argument(
        '--start-at',
        dest='start_at',
        default=0,
        type=int
    )
    parser.add_argument(
        '--skip',
        default=0,
        type=int
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
    for i, weights_file in enumerate(args.weights_pre_list):
        args.weights_pre_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    for i, weights_file in enumerate(args.weights_post_list):
        args.weights_post_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    train_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    test_dir = get_output_dir(cfg.TEST.DATASETS, training=False)

    model_list = []
    files = os.listdir(train_dir)
    for f in files:
        if f.startswith("model_") and f.endswith(".pkl"):
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                model_list.append((f, int(iter_string[0])))
    model_list.sort(key=lambda tup: tup[1])
    if "model_final.pkl" in files:
        model_list.append(("model_final.pkl", "final"))

    seq_map_path = os.path.join(test_dir, "seq_map.txt")
    with open(seq_map_path, "w+") as seq_map:
        seq_map.write("name\n")
        for dataset in cfg.TEST.DATASETS:
            seq_map.write(get_im_dir(dataset).split("/")[-2] + '\n')

    seq_map_path = os.path.relpath(os.path.abspath(seq_map_path), os.path.expanduser(os.path.join(args.devkit_path, "seqmaps")))

    matlab_eng = get_matlab_engine(args.devkit_path)
    eval_datections = lambda res_dir, gt_dir: eval_detections_matlab(matlab_eng, seq_map_path, res_dir, gt_dir, 'MOT17')

    for i, (model, it) in enumerate(model_list):
        if i % (args.skip + 1) != 0:
            logger.info("Skipping {}".format(model))
            continue
        if it >= args.start_at:
            weights_list = args.weights_pre_list + [os.path.join(train_dir, model)] + args.weights_post_list
            preffix_list = args.preffix_list if len(args.preffix_list) else [""] * (len(args.weights_pre_list) + len(args.weights_post_list) + 1)
            workspace.ResetWorkspace()
            model = infer_engine.initialize_mixed_model_from_cfg(weights_list, preffix_list=preffix_list)
            logger.info("Processing model {}".format(it))
            tracking = Tracking(args.thresh, cfg.TRCNN.MAX_BACK_TRACK)
            for dataset in cfg.TEST.DATASETS:
                logger.info("Processing dataset {}".format(dataset))
                im_dir = get_im_dir(dataset)
                output_file = os.path.join(test_dir, str(it), im_dir.split("/")[-2] + ".txt")
                head, tail = os.path.split(output_file)
                if not os.path.exists(head):
                    os.makedirs(head)
                # import pickle
                # proposals = pickle.load(open('/home/orestis/datasets/MOT17/train/MOT17-13-FRCNN/det/proposals.pkl', 'r'))
                proposals None
                infer_track_sequence(model, im_dir, tracking, proposals=proposals, vis=None, det_file=output_file)

                eval_datections(os.path.abspath(head) + "/", os.path.abspath(os.path.join(*im_dir.split("/")[:-2])) + "/")


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
