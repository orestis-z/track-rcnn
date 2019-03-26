"""Script to test and validate tracking

Testing with respect to models from different training steps or different
inference hyper-parameters.
Configurable to output visualizations and to store detection to a file.
Testing setting allows to evaluate the tracking by interfacing the matlab engine.
"""

import logging
import argparse
import cv2
import os, sys
import pprint
import numpy as np
import re
import pickle
import time

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
from detectron.utils.tracking import Tracking, back_track, \
    infer_track_sequence, get_matlab_engine, eval_detections_matlab

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
        help='Path of the MOT devkit',
        default='~/repositories/motchallenge-devkit',
        type=str
    )
    parser.add_argument(
        '--start-at',
        dest='start_at',
        help='start evaluation at iteration `start-at`',
        default=0,
        type=int
    )
    parser.add_argument(
        '--skip',
        help='evaluate each `skip` + 1 model',
        default=0,
        type=int
    )
    parser.add_argument(
        '--offset',
        help='initially skip `offset` models',
        default=0,
        type=int
    )
    parser.add_argument(
        '--model',
        help='weights model file (/path/to/model_weights.pkl)',
        default=0,
        type=str
    )
    parser.add_argument(
        '--model-suffix',
        dest="model_suffix",
        help="extra folder suffix",
        default="",
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

# Hard-coded hyper parameter inference  
HYPER_PARAM = None
# HYPER_PARAM = "DETECTION_THRESH"
# PARAM_RANGE = np.linspace(0.6, 0.8, 11)
# PARAM_RANGE = np.linspace(0.52, 0.58, 4)
# PARAM_RANGE = np.linspace(0.0, 1.0, 11)
# PARAM_RANGE = np.linspace(0.62, 0.8, 10)
# PARAM_RANGE = np.linspace(0.67, 0.73, 4)
# HYPER_PARAM = "MAX_BACK_TRACK"
# PARAM_RANGE = xrange(0, 11)

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

    # If True: Evaluation with respect to specified parameters (model from
    # specifig training iterations or inference hyper-parameters)
    # If False: Infer test sequence for evaluation on the MOT benchmark server
    EVAL = "eval" in args.opts 

    train_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    train_dir_split = train_dir.split("/")
    train_dir = os.path.join("/".join(train_dir_split[:-1]) + \
        args.model_suffix, train_dir_split[-1])

    model_list = []
    files = os.listdir(train_dir)

    if EVAL:
        test_dir = "/".join(get_output_dir(cfg.TEST.DATASETS,
            training=False).split("/")[:-1]) + args.model_suffix
        # Evaluation with respect to inference hyper-parameters
        if HYPER_PARAM is not None:
            test_dir = os.path.join(test_dir, HYPER_PARAM.lower())
            model_param = ((args.model, param) for param in PARAM_RANGE)
        # Evaluation with respect to weights from specific training iterations
        else:
            model_param = []
            for f in files:
                if f.startswith("model_") and f.endswith(".pkl"):
                    iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
                    if len(iter_string) > 0:
                        model_param.append((f, int(iter_string[0])))
            model_param.sort(key=lambda tup: tup[1])
            if "model_final.pkl" in files:
                model_param.append(("model_final.pkl", "final"))
        # Tracking evaluation by interface to matlab engine
        seq_map_path = os.path.join(test_dir, "seq_map.txt")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        with open(seq_map_path, "w+") as seq_map:
            seq_map.write("name\n")
            for dataset in cfg.TEST.DATASETS:
                seq_map.write(get_im_dir(dataset).split("/")[-2] + '\n')
        seq_map_path = os.path.relpath(os.path.abspath(seq_map_path),
            os.path.expanduser(os.path.join(args.devkit_path, "seqmaps")))
        matlab_eng = get_matlab_engine(args.devkit_path)
        eval_datections = lambda res_dir, gt_dir: eval_detections_matlab(
            matlab_eng, seq_map_path, res_dir, gt_dir, 'MOT17')
    else:
        if args.model is not None:
            model_param = ((args.model, None),)
        else:
            model_param = (("model_final.pkl", "final"),)

    # Iterate through (model, parameter) tuples
    for i, (model, param) in enumerate(model_param):
        if EVAL and (i + 1 + args.offset) % (args.skip + 1) != 0:
            logger.info("Skipping {}".format(model))
            continue
        # Hyper parameter inference
        elif HYPER_PARAM is not None:
                cfg.immutable(False)
                setattr(cfg.TRCNN, HYPER_PARAM, param)
                assert_and_infer_cfg(cache_urls=False)
                print(cfg.TRCNN)
        if not EVAL or param >= args.start_at:
            weights_list = args.weights_pre_list + [os.path.join(train_dir, model)] + \
                args.weights_post_list
            preffix_list = args.preffix_list if len(args.preffix_list) \
                else [""] * (len(args.weights_pre_list) + len(args.weights_post_list) + 1)
            workspace.ResetWorkspace()
            model = infer_engine.initialize_mixed_model_from_cfg(weights_list,
                preffix_list=preffix_list)
            logger.info("Processing {}".format(param))
            timing = []
            # iterate through test sequences
            for dataset in cfg.TEST.DATASETS:
                tracking = Tracking(args.thresh, cfg.TRCNN.MAX_BACK_TRACK)
                logger.info("Processing dataset {}".format(dataset))
                im_dir = get_im_dir(dataset)
                vis = None
                if EVAL:
                    output_file = os.path.join(test_dir, str(param),
                        im_dir.split("/")[-2] + ".txt")
                # Visualize detections along with tracking detection file creation
                else:
                    output_dir = os.path.join("outputs/MOT17", im_dir.split("/")[-2])
                    if "vis" in args.opts:
                        vis = {
                            "output-dir": output_dir,
                            "dummy-dataset": dummy_dataset,
                            "show-class": "show-class" in args.opts,
                            "show-track": "show-track" in args.opts,
                            "thresh": args.thresh,
                            "track-thresh": cfg.TRCNN.DETECTION_THRESH,
                            "n-colors": 15,
                        }
                    output_file = os.path.join("outputs/MOT17",
                        im_dir.split("/")[-2] + ".txt")
                # Use custom proposals if provided
                if "proposals" in args.opts:
                    proposals = pickle.load(open(os.path.join(*(im_dir.split("/")[:-1] + \
                        ["det/proposals.pkl"])), 'r'))
                else:
                    proposals = None
                head, tail = os.path.split(output_file) 
                if not os.path.exists(head):
                    os.makedirs(head)
                start = time.time()
                # Run inference
                infer_track_sequence(model, im_dir, tracking, proposals=proposals,
                    vis=vis, det_file=output_file)
                delta = time.time() - start
                freq = float(len(os.listdir(im_dir))) / delta
                timing.append(freq)

            # Save evaluation results
            if EVAL:
                val_directory = os.path.abspath(head) + "/"
                eval_datections(val_directory,
                    os.path.abspath(os.path.join(*im_dir.split("/")[:-2])) + "/")
                with open(val_directory + "eval.txt", "r") as f:
                    temp = f.readline().strip()
                with open(val_directory + "eval.txt", "w+") as f:
                    f.write("{},{}".format(temp, np.average(timing)))


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
