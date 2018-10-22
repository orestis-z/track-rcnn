import argparse
import subprocess
import sys
import os
import pydot
from collections import defaultdict

import cv2
import numpy as np

from caffe2.python import net_drawer
from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
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
        default=['net', 'mask_net', 'keypoint_net', 'track_net', 'track_rec_net', 'param_init_net'],
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
        default='.outputs/graphs',
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

def GetPydotGraph(
    operators_or_net,
    name=None,
    rankdir='LR',
    node_producer=None,
    shapes=False,
):
    if node_producer is None:
        node_producer = net_drawer.GetOpNodeProducer(False, **net_drawer.OP_STYLE)
    operators, name = net_drawer._rectify_operator_and_name(operators_or_net, name)
    graph = pydot.Dot(name, rankdir=rankdir)
    sub_graph_frcnn = pydot.Cluster('sub_graph_class',label='fast-rcnn head', fillcolor="tomato", style="filled")
    sub_graph_mask = pydot.Cluster('sub_graph_mask',label='mask head', fillcolor="deepskyblue", style="filled")
    sub_graph_kps = pydot.Cluster('sub_graph_kps',label='keypoints head', fillcolor="gold", style="filled")
    sub_graph_track = pydot.Cluster('sub_graph_track',label='track head', fillcolor="gold", style="filled")
    sub_graph_fpn = pydot.Cluster('sub_graph_fpn',label='FPN', fillcolor="moccasin", style="filled")
    graph.add_subgraph(sub_graph_frcnn)
    graph.add_subgraph(sub_graph_mask)
    graph.add_subgraph(sub_graph_kps)
    graph.add_subgraph(sub_graph_track)
    # graph.add_subgraph(sub_graph_fpn)
    pydot_nodes = {}
    pydot_node_counts = defaultdict(int)
    for op_id, op in enumerate(operators):
        op_node = node_producer(op, op_id)
        graph.add_node(op_node)
        for input_name in op.input:
            if input_name[-2] == "_":
                continue
            if input_name not in pydot_nodes:
                label = net_drawer._escape_label(input_name)
                if shapes:
                    shape = workspace.FetchBlob(input_name).shape
                    label += "\n" + str(shape)
                input_node = pydot.Node(
                    net_drawer._escape_label(
                        input_name + str(pydot_node_counts[input_name])),
                    label=label,
                    style="filled", fillcolor="white",
                    **net_drawer.BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            graph.add_node(input_node)
            graph.add_edge(pydot.Edge(input_node, op_node))
            sub_graph = None
            if 'mask' in input_name:
                sub_graph = sub_graph_mask
            elif 'kps' in input_name or 'keypoint' in input_name or 'pose' in input_name or 'conv_fcn' in input_name:
                sub_graph = sub_graph_kps
            elif 'track' in input_name:
                sub_graph = sub_graph_track
            elif ('cls' in input_name or 'bbox' in input_name or 'fc' in input_name) and 'rpn' not in input_name and 'fcn' not in input_name:
                sub_graph = sub_graph_frcnn
            if '/fpn' in input_name:
                sub_graph = sub_graph_fpn
            if sub_graph:
                sub_graph.add_node(input_node)
                sub_graph.add_node(op_node)
        for output_name in op.output:
            # if output_name[-2] == "_":
            #     continue
            if output_name in pydot_nodes:
                # we are overwriting an existing blob. need to updat the count.
                pydot_node_counts[output_name] += 1
            label = net_drawer._escape_label(output_name)
            if shapes:
                shape = workspace.FetchBlob(output_name).shape
                label += "\n" + str(shape)
            output_node = pydot.Node(
                net_drawer._escape_label(
                    output_name + str(pydot_node_counts[output_name])),
                label=label,
                style="filled", fillcolor="white",
                **net_drawer.BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            graph.add_node(output_node)
            graph.add_edge(pydot.Edge(op_node, output_node))
            sub_graph = None
            if 'mask' in output_name:
                sub_graph = sub_graph_mask
            elif 'kps' in output_name or 'keypoint' in output_name or 'pose' in output_name or 'conv_fcn' in output_name:
                sub_graph = sub_graph_kps
            elif 'track' in output_name:
                sub_graph = sub_graph_track
            elif ('cls' in output_name or 'bbox' in output_name or 'fc' in output_name) and 'rpn' not in output_name and 'fcn' not in output_name:
                sub_graph = sub_graph_frcnn
            elif '/fpn' in output_name:
                sub_graph = sub_graph_fpn
            if sub_graph:
                sub_graph.add_node(output_node)
    return graph

def main():
    MINIMAL = False
    TRAIN = False
    FORWARD = False
    SHAPES = False
    args = parse_args()
    if args.opts is not None:
        if 'minimal' in args.opts:
            MINIMAL = True
        if 'train' in args.opts:
            TRAIN = True
        if 'forward' in args.opts:
            FORWARD = True
        if 'shapes' in args.opts:
            SHAPES = True

    if args.model_file is None:
        raise ValueError('Specify model file')
    MODEL_FILE = args.model_file
    NET_NAMES = args.net_names

    if MINIMAL:
        get_dot_graph = lambda net, shapes: net_drawer.GetPydotGraphMinimal(net, rankdir="BT")
    else:
        get_dot_graph = lambda net, shapes: GetPydotGraph(net, rankdir="BT", shapes=shapes)

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

        dataset = JsonDataset(cfg.TEST.DATASETS[0])
        roidb = dataset.get_roidb()

        with c2_utils.NamedCudaScope(0):
            if cfg.MODEL.TRACKING_ON:
                roidb_min = [roidb[0], roidb[1]]
                im_list = [cv2.imread(e['image']) for e in roidb_min]
                infer_engine.im_detect_all_multi(
                    model, im_list, [None, None]
                )
            else:
                infer_engine.im_detect_all_multi(
                    model, roidb[0]['image'], None
                )
    else:
        model = model_builder.create(cfg.MODEL.TYPE, train=TRAIN)

    subprocess.call(["killall", "xdot"])

    for net_name in NET_NAMES:
        net = getattr(model, net_name, None)
        if net:
            print('processing graph {}...'.format(net_name))
            g = get_dot_graph(net, shapes=SHAPES)
            name = net_name
            if TRAIN:
                name_append = 'train'
            else:
                name_append = 'infer'
            graph_dir =  os.path.join(args.output_dir, MODEL_FILE)
            if not os.path.exists(graph_dir):
                os.makedirs(graph_dir)
            dot_name = os.path.join(graph_dir, '{}_{}.dot'.format(net_name, name_append))
            g.write_dot(dot_name)
            subprocess.Popen(['xdot', dot_name])


if __name__ == '__main__':
    main()
