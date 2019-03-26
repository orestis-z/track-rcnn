"""Filters keypoint trajectories.

Applies median or gaussian filter.
"""

import argparse
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import medfilt

import detectron.utils.keypoints as keypoint_utils


dataset_keypoints, _ = keypoint_utils.get_keypoints()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kps-3d',
        dest='kps_3d',
        help='Pre-computed 3d keypoints in world-frame',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest="output_dir",
        help="output directory for the filtered keypoints",
        type=str
    )
    parser.add_argument(
        '--filter',
        dest="filter_type",
        help='guassian or median filter',
        default='median',
        type=str
    )
    parser.add_argument(
        '--filter-var',
        dest="filter_var",
        help='Filter size parameter',
        type=float
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


def filter_valid(p, sc, valid, kp_thresh=2):
    """filter valid keypoints
    """
    return np.where(
        np.logical_and(
            sc > kp_thresh,
            valid
        ),
        p, np.zeros(p.shape) * np.NaN)

def main(args):
    if args.filter_type == "guassian":
        filt = lambda data: gaussian_filter1d(data, args.filter_var)
    elif args.filter_type == "median":
        filt = lambda data: medfilt(data, int(args.filter_var))
    else:
        raise ValueError

    kps_3d_arr = np.load(open(args.kps_3d))
    tracks = {}
    for i, dets in enumerate(kps_3d_arr):
        for det in dets:
            obj_id, kps_3d, valid_3d = det
            if obj_id not in tracks:
                tracks[obj_id] = []
            tracks[obj_id].append((i, kps_3d, valid_3d))

    for obj_id in tracks.keys():
        fig = None
        if "plot" in args.opts and len(tracks[obj_id]) > 65:
            fig = plt.figure()
        for i in xrange(19):
            p = np.array([det[1][:3, i] for det in tracks[obj_id]])
            sc = np.array([[det[1][3, i] for det in tracks[obj_id]]] * 3).T
            valid = np.array([[det[2][i] for det in tracks[obj_id]]] * 3).T
            p_valid = filter_valid(p, sc, valid)

            x = p_valid[:, 0]
            y = p_valid[:, 1]
            z = p_valid[:, 2]

            x_filtered = filt(x)
            y_filtered = filt(y)
            z_filtered = filt(z)

            p_filtered = np.vstack((x_filtered, y_filtered, z_filtered)).T

            for j in xrange(len(tracks[obj_id])):
               tracks[obj_id][j][1][:3, i] = p_filtered[j]

            if fig is not None:
                ax = fig.add_subplot(4, 5, i + 1, projection='3d', proj_type='ortho')
                ax.set_aspect('equal')
            
                ax.plot(x, y, z, "-")
                ax.plot(x_filtered, y_filtered, z_filtered, "g-")
                len_dataset = len(dataset_keypoints)
                ax.set_title(dataset_keypoints[i] if len_dataset > i else ["mid_shoulder", "mid_hip"][i - len_dataset])

                x_max = np.max(p[:, 0])
                x_min = np.min(p[:, 0])
                y_max = np.max(p[:, 1])
                y_min = np.min(p[:, 1])
                z_max = np.max(p[:, 2])
                z_min = np.min(p[:, 2])
                max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_max + x_min)
                Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_max + y_min)
                Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_max + z_min)
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax.plot([xb], [yb], [zb], 'w')

    # put everything back to same format
    kps_3d_arr_filt = [[] for _ in xrange(len(kps_3d_arr))]
    for obj_id, dets in tracks.items():
        for det in dets:
            i, kps_3d, valid_3d = det
            kps_3d_arr_filt[i].append((obj_id, kps_3d, valid_3d))

    np.save(open(os.path.join(args.output_dir, 'kps_3d-{}-{}.npy'.format(args.filter_type, args.filter_var)), 'w'), np.array(kps_3d_arr_filt))

    if "plot" in args.opts:
        plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
