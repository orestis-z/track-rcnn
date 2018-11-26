import pydot
import numpy as np

from caffe2.python import net_drawer
from caffe2.python import workspace

def defaultdict(default_type):
    class DefaultDict(dict):
        def __getitem__(self, key):
            if key not in self:
                dict.__setitem__(self, key, default_type())
            return dict.__getitem__(self, key)
    return DefaultDict()

def GetPydotGraph(
    operators_or_net,
    name=None,
    rankdir='LR',
    node_producer=None,
    shapes=False,
    hide_params=True
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
            if hide_params and input_name[-2] == "_":
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


def GetPydotGraphMinimal(net, rankdir="BT"):
    return net_drawer.GetPydotGraphMinimal(net, rankdir="BT")
