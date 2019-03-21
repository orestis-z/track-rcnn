"""Script to visualize the network architecture as a computation graph.

Able to visualize both training and inference architecture. 
Operations are represented as green nodes and blobs as white nodes. Edges are linking
input and output blobs to operations.
Backbone and network heads are grouped with distinct colors.
Blob dimension examination is also supported.
"""

import argparse
import subprocess
import sys
import os

import cv2

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.utils.net_drawer as net_drawer
from detectron.modeling import model_builder
import detectron.core.test_engine as infer_engine
import detectron.utils.blob as blob_utils
import detectron.utils.c2 as c2_utils
from detectron.datasets.roidb import combined_roidb_for_training
from detectron.datasets.json_dataset import JsonDataset
import detectron.utils.net as nu

c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--model',
        dest='model_file',
        # help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--net',
        dest='net_names',
        # help='Config file for training (and optionally testing)',
        default=['net', 'conv_body_net', 'mask_net', 'keypoint_net', 'track_net', 'track_rec_net', 'param_init_net'],
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--input_shape',
        dest='input_shape',
        # help='Config file for training (and optionally testing)',
        default=None,
        type=lambda val: eval(val),
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='outputs/graphs',
        type=str
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    MINIMAL = False
    TRAIN = False
    FORWARD = False
    SHAPES = False
    HIDE_PARAMS = True
    if args.opts is not None:
        if 'minimal' in args.opts:
            MINIMAL = True
        if 'train' in args.opts:
            TRAIN = True
        if 'forward' in args.opts:
            FORWARD = True
        if 'shapes' in args.opts:
            SHAPES = True
        if 'params' in args.opts:
            HIDE_PARAMS = False

    if SHAPES and args.model_file is None:
        raise ValueError('Specify model file')
    MODEL_FILE = args.model_file
    NET_NAMES = args.net_names

    if MINIMAL:
        get_dot_graph = lambda net, shapes: net_drawer.GetPydotGraphMinimal(net, rankdir="BT")
    else:
        get_dot_graph = lambda net, shapes: net_drawer.GetPydotGraph(net, rankdir="BT", shapes=shapes, hide_params=HIDE_PARAMS)

    # get model
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    cfg.NUM_GPUS = 1
    cfg.VIS_NET = True
    if FORWARD:
        cfg.MODEL.FORWARD_ONLY = True
    assert_and_infer_cfg(cache_urls=False)

    if SHAPES and TRAIN:
        raise NotImplementedError

    if SHAPES:
        model = infer_engine.initialize_model_from_cfg(MODEL_FILE)
        workspace.RunNetOnce(model.param_init_net)
        nu.broadcast_parameters(model)

        dataset = JsonDataset(cfg.TRAIN.DATASETS[0])
        roidb = dataset.get_roidb()

        with c2_utils.NamedCudaScope(0):
            if cfg.MODEL.TRACKING_ON:
                roidb_min = [roidb[0], roidb[1]]
                im_list = [cv2.imread(e['image']) for e in roidb_min]
                infer_engine.multi_im_detect_all(
                    model, im_list, [None, None]
                )
            else:
                infer_engine.im_detect_all(
                    model, roidb[0]['image'], None
                )
    else:
        model = model_builder.create(cfg.MODEL.TYPE, train=TRAIN)

    subprocess.call(["killall", "xdot"])

    for net_name in NET_NAMES:
        net = getattr(model, net_name, None)
        if net:
            print('processing graph {}...'.format(net_name))
            g = get_dot_graph(net.Proto(), shapes=SHAPES)
            name = net_name
            if TRAIN:
                name_append = 'train'
            else:
                name_append = 'infer'
            graph_dir =  os.path.join(args.output_dir, cfg.MODEL.TYPE)
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            dot_name = os.path.join(graph_dir, '{}_{}.dot'.format(net_name, name_append))
            g.write_dot(dot_name)
            subprocess.Popen(['xdot', dot_name])


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
