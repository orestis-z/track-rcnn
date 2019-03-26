"""Script to visualize MOT evaluation results

Metrics, labels, plots etc. to be shown are hardcoded due to high variance of configurability.
"""

from __future__ import division

import logging
import argparse
import numpy as np
import sys, os
import json
import re

import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

from detectron.utils.logging import setup_logging


# KEYS : ([Metric name], [1: the higher the better, 0: the lower the better, -1: unknown],
# [unit], [bounds])
# bounds: None indicates unbounded
KEYS = (
    ("IDF1", 1, "%", (14, None)),
    ("IDP", 1, "-", (15, None)),
    ("IDR", 1, "-", (13, None)),
    ("RCLL", 1, "%", (None, None)),
    ("PRCN", 1, "%", (None, None)),
    ("FAR", -1, "?", (None, None)),
    ("GT", -1, "?", (None, None)),
    ("MT", 1, "%", (None, None)),
    ("PT", -1, "?", (None, None)),
    ("ML", 0, "%", (None, None)),
    ("FP", 0, "-", (None, None)),
    ("FN", 0, "-", (None, None)),
    ("IDS", 0, "-", (None, 2000)),
    ("FM", -1, "?", (None, 750)),
    # ("MOTA", 1, "%", (35, 40)),
    # ("MOTA", 1, "%", (30, None)),
    ("MOTA", 1, "%", (None, None)),
    ("MOTP", 1, "%", (None, None)),
    ("MOTAL", -1, "?", (None, None)),
    ("Hz", 1, "$s^{-1}$", (None, None)),
)
font = {'family' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)

# Hardcoded subplots to exclude + subplot shape
exclude = (
    "FAR", "GT", "PT", "FM", "MOTAL",
    "IDF1", "IDP", "IDR", "MT", "ML", "FP", "FN", "IDS", "MOTP",
    "RCLL", "PRCN",
    # "Hz",
)
subplot_shape = (1, 2)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--eval-dir',
        dest='eval_dir',
        help="evaluation directory",
        default=None,
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--smooth-sigma',
        dest='smooth_sigma',
        help='gaussian smoothing strength (sigma)',
        default=1,
        type=float,
    )
    parser.add_argument(
        '--iter-max',
        dest='iter_max',
        help='maximum x value (iteration)',
        default=None,
        type=int,
    )
    parser.add_argument(
        '--x-label',
        dest='x_label',
        help='x label',
        default="iter",
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    res_list = []
    folder_list_list = [] # :D
    # Iterate through evaluation directory 
    for l, eval_dir in enumerate(args.eval_dir):
        folder_list = []
        files = os.listdir(eval_dir)
        for f in files:
            if re.match("^\d+(\.)?(\d+)?$", f) is not None:
                folder_list.append(f)
        folder_list.sort(key=lambda x: float(x))

        res = [] # list to store the evaluation results
        folder_list_temp = list(folder_list)
        for m, folder in enumerate(folder_list):
            res_path = os.path.join(eval_dir, folder, "eval.txt")
            if os.path.isfile(res_path):
                with open(res_path) as f:
                    line = f.readline().strip().split(",")
                    line = [float(x) for i, x in enumerate(line)]
                    res.append(line)
            else:
                print(res_path + " not found")
                folder_list_temp.remove(folder)
        folder_list = folder_list_temp
        folder_list_list.append(folder_list)
        res = np.array(res).astype(np.float).T
        res_list.append(res)

    # Gaussian smoothing lambda
    smooth = lambda data, sigma=args.smooth_sigma: gaussian_filter1d(data, sigma)

    colors = (
        (31 / 255, 119 / 255, 180 / 255),
        (1, 127 / 255, 14 / 255),
    )
    labels = (
        "B=256, F=512",
        "B=64, F=128",
    )
    lw = 2 # line width

    exc_c = 0 
    fig = plt.figure()
    fig.set_size_inches(12, 6) # figure size
    for i, key in enumerate(KEYS):
        if key[0] in exclude:
            exc_c += 1
            continue
        if len(res) <= i:
            continue
        # Iterate through results
        for j, res in enumerate(res_list):
            folder_list = folder_list_list[j]
            t = [float(f) for f in folder_list]
            ax = fig.add_subplot(subplot_shape[0], subplot_shape[1],
                i - exc_c + 1)
            # No smoothing
            if smooth_sigma == 0:
                ax.plot(t, res[i], color=colors[j], lw=lw, label=labels[j])
            # Gaussian smoothing
            else:
                ax.plot(t, res[i], color=colors[j] + (0.4,),  lw=lw)
                ax.plot(t, smooth(res[i]), color=colors[j], lw=lw,
                    label=labels[j])
            better = " $\uparrow$" if key[1] == 1 else " $\downarrow$" \
                if key[1] == 0 else ""
            ax.set_title("{}{}".format(key[0], better))
            if key[3][0] is not None:
                ax.set_ylim(bottom=key[3][0])
            if key[3][1] is not None:
                ax.set_ylim(top=key[3][1])
            if args.iter_max is not None:
                ax.set_xlim(left=0, right=args.iter_max)
        ax.legend()
        ax.set_ylabel("[{}]".format(key[2]), rotation=0, labelpad=20)
        ax.set_xlabel(args.x_label)
        if opts.scientific:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid()
    # Plot adjustments
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    #fig.subplots_adjust(left=0.15)
    plt.show()

if __name__ == '__main__':
    setup_logging(__name__)
    args = parse_args()
    main(args)
