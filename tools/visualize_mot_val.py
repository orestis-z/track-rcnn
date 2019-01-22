from __future__ import division

import logging
import argparse
import numpy as np
import sys, os
import json

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

from detectron.utils.logging import setup_logging


KEYS = (
    "IDF1",
    "IDP",
    "IDR",
    "RCLL",
    "PRCN",
    "FAR",
    "GT",
    "MT",
    "PT",
    "ML",
    "FP",
    "FN",
    "IDS",
    "FM",
    "MOTA",
    "MOTP",
    "MOTAL")

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        default=None,
        type=str
    )
    parser.add_argument(
        '--delta',
        dest='delta',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--smooth-sigma',
        dest='smooth_sigma',
        default=1,
        type=float,
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

    folder_list = []
    files = os.listdir(args.model_dir)
    for f in files:
        if f.isdigit():
            folder_list.append(f)
    folder_list.sort(key=lambda x: int(x))
    if "final" in files:
        folder_list.append("final")

    res = []
    for folder in folder_list:
        res_path = os.path.join(args.model_dir, folder, "eval.txt")
        if os.path.isfile(res_path):
            with open(res_path) as f:
                res.append([float(x) for x in f.readline().split(",")])
        else:
            print(res_path + " not found")
    res = np.array(res).T

    smooth = lambda data, sigma=args.smooth_sigma: gaussian_filter1d(data, sigma)

    color_1 = (31 / 255, 119 / 255, 180 / 255)
    color_2 = (1, 127 / 255, 14 / 255)
    lw = 1

    fig = plt.figure()
    for i, var in enumerate(KEYS):
        t = np.array(range(len(res[i]))) * args.delta + args.delta
        ax = fig.add_subplot(5, 4, i + 1)
        ax.plot(t, res[i], '-o', color=color_1 + (0.5,), lw=lw)
        ax.plot(t, smooth(res[i]), color=color_1, lw=lw, label=var)
        ax.legend()
        ax.grid()

    plt.show()

if __name__ == '__main__':
    setup_logging(__name__)
    args = parse_args()
    main(args)
