"""Script to visualize tennsorboard metrics

Assumes that provided file is in csv format.
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


font = {'family' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--file',
        help='one or more CSV files of the losses from tensorboard',
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
    # Iterate result files
    for file in args.file:
        with open(file) as f:
            res = np.array([[float(x) for x in line.split(",")] \
                for i, line in enumerate(f.readlines()) if i > 0])
        res_list.append(res)

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

    fig = plt.figure()
    fig.set_size_inches(6, 6) # figure size
    # Iterate through results
    for j, res in enumerate(res_list):
        val = res[:, 2]
        t = res[:, 1]
        plt.plot(t, val, color=colors[j] + (0.5,), lw=lw)
        plt.plot(t, smooth(val), color=colors[j], lw=lw, label=labels[j])
        plt.title("Cross-entropy $\downarrow$")
        if args.iter_max is not None:
            plt.xlim(left=0, right=args.iter_max)
        plt.legend()
        plt.ylabel("[-]", rotation=0, labelpad=10)
        plt.xlabel(x_label)
        # scientific x label format
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    setup_logging(__name__)
    args = parse_args()
    main(args)
