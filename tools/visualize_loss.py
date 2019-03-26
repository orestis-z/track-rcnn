"""Visualizes loss logged during training"""

from __future__ import division

import sys
import json

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np


with open(sys.argv[-1]) as f:
    data_raw = [json.loads(line) for line in f.readlines()]
    data = {'main': {}, 'loss_rpn_cls': {}, 'loss_rpn_bbox': {}, 'meta': {}}
    step = [el['iter'] for el in data_raw]
    data['main']['loss'] = [el.get('loss', np.NaN) for el in data_raw]
    data['main']['loss_cls'] = [el.get('loss_cls', np.NaN) for el in data_raw]
    data['main']['accuracy_cls'] = [el.get('accuracy_cls', np.NaN) for el in data_raw]
    data['main']['loss_bbox'] = [el.get('loss_bbox', np.NaN) for el in data_raw]
    data['main']['loss_mask'] = [el.get('loss_mask', np.NaN) for el in data_raw]
    data['main']['loss_kps'] = [el.get('loss_kps', np.NaN) for el in data_raw]
    data['main']['loss_track'] = [el.get('loss_track', np.NaN) for el in data_raw]
    data['main']['loss_track_match'] = [el.get('loss_track_match', np.NaN) for el in data_raw]
    data['main']['loss_track_nomatch'] = [el.get('loss_track_nomatch', np.NaN) for el in data_raw]
    data['loss_rpn_cls']['fpn2'] = [el.get('loss_rpn_cls_fpn2', np.NaN) for el in data_raw]
    data['loss_rpn_cls']['fpn3'] = [el.get('loss_rpn_cls_fpn3', np.NaN) for el in data_raw]
    data['loss_rpn_cls']['fpn4'] = [el.get('loss_rpn_cls_fpn4', np.NaN) for el in data_raw]
    data['loss_rpn_cls']['fpn5'] = [el.get('loss_rpn_cls_fpn5', np.NaN) for el in data_raw]
    data['loss_rpn_cls']['fpn6'] = [el.get('loss_rpn_cls_fpn6', np.NaN) for el in data_raw]
    data['loss_rpn_bbox']['fpn2'] = [el.get('loss_rpn_bbox_fpn2', np.NaN) for el in data_raw]
    data['loss_rpn_bbox']['fpn3'] = [el.get('loss_rpn_bbox_fpn3', np.NaN) for el in data_raw]
    data['loss_rpn_bbox']['fpn4'] = [el.get('loss_rpn_bbox_fpn4', np.NaN) for el in data_raw]
    data['loss_rpn_bbox']['fpn5'] = [el.get('loss_rpn_bbox_fpn5', np.NaN) for el in data_raw]
    data['loss_rpn_bbox']['fpn6'] = [el.get('loss_rpn_bbox_fpn6', np.NaN) for el in data_raw]
    data['meta']['mem'] = [el.get('mem', np.NaN) for el in data_raw]
    data['meta']['lr'] = [el.get('lr', np.NaN) for el in data_raw]
    data['meta']['mb_qsize'] = [el.get('mb_qsize', np.NaN) for el in data_raw]
    data['meta']['time'] = [el.get('time', np.NaN) for el in data_raw]

smooth = lambda data, sigma=3: gaussian_filter1d(data, sigma)

color_1 = (31 / 255, 119 / 255, 180 / 255)
color_2 = (1, 127 / 255, 14 / 255)
lw = 1

figures = []

for key in data.keys():
    fig = plt.figure()
    fig.suptitle(key)
    figures.append(fig)
    n_vars = len(data[key])
    n_cols = int(n_vars ** 0.5)
    n_rows = int(n_vars / n_cols + 0.5)
    data_key = data[key].keys()
    data_key.sort()
    for i, var in enumerate(data_key):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.plot(step, data[key][var], color=color_1 + (0.5,), lw=lw)
        ax.plot(step, smooth(data[key][var]), color=color_1, lw=lw, label=var)
        ax.legend()
        ax.grid()

plt.show()
