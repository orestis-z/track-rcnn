"""visualize 3D human pose tracking

Live visualization and / or recording of the 3D human pose.
Input are RGB-D image sequence and 2D detections.
Transforms the detections to the world-frame and applies filter to depth.
Detections are eigher computed on the fly or provided as pre-computed.
"""

import argparse
import os
import sys
import cv2
import pickle
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import medfilt2d

import detectron.utils.keypoints as keypoint_utils
import detectron.utils.vis as vis_utils

# Axis permutation
# I = [0, 1, 2]
I = [2, 0, 1]
# I = [1, 2, 0]

# Tolerance in difference of timestamps for rgb and depth
DELTA_T_MAX = 0.02

dataset_keypoints, _ = keypoint_utils.get_keypoints()
kp_lines = vis_utils.kp_connections(dataset_keypoints)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        help="data dir",
        default=None,
        type=str
    )
    parser.add_argument(
        '--dataset',
        default='tum',
        help='dataset type. tum or princeton',
        type=str
    )
    parser.add_argument(
        '--kps-3d',
        dest='kps_3d',
        help='Pre-computed 3d keypoints in world-frame',
        default=None,
        type=str
    )
    parser.add_argument(
        '--mode',
        help="0: pointcloud with image. 1: pointcloud projected keypoints. 2: 0 and 1.",
        default=0,
        type=int
    )
    parser.add_argument(
        '--k-size',
        dest='k_size',
        help='Depth median filter size',
        default=3,
        type=int
    )
    parser.add_argument(
        '--shrink-factor',
        dest='shrink_factor',
        help='image + depth shrink factor to speed up rendering',
        default=1,
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

def q_mult(q1, q2):
    """Quaternion multiplication"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    """Quaternion inverse"""
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    """Quaternion-vector multiplication"""  
    q2 = [0.0] + v1.tolist()
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def map_kps_3d(kps, p_map):
    """Map keypoints using `p_map` to 3D and filter invalid points
    """
    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    
    valid_3d = []
    kps_3d = np.zeros((4, len(dataset_keypoints) + 2))
    # add all kps
    kps_3d[3, :len(dataset_keypoints)] = kps[2, :]
    points_3d, valid = zip(*[p_map(np.rint(kp).astype(np.int)) for kp in kps[:2].T])
    kps_3d[:3, :len(dataset_keypoints)] = np.array(points_3d).T
    valid_3d += valid
    # add mid shoulder & mid hip
    kps_3d[3, len(dataset_keypoints):] = (sc_mid_shoulder, sc_mid_hip)
    points_3d, valid = zip(*[p_map(np.rint(kp).astype(np.int)) for kp in (mid_shoulder, mid_hip)])
    kps_3d[:3, len(dataset_keypoints):] = np.array(points_3d).T
    valid_3d += valid

    return kps_3d, valid_3d

obj_id_to_i_cmap = {}
cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def vis_keypoints_3d(ax, kps_3d, valid_3d, obj_id, kp_thresh=2, n_cmaps=len(cmaps)):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    global obj_id_to_i_cmap
    i_cmap = obj_id_to_i_cmap.get(obj_id)
    if i_cmap is None:
        i_cmap = random.randint(0, n_cmaps - 1)
        obj_id_to_i_cmap[obj_id] = i_cmap
    cmap = plt.get_cmap(cmaps[i_cmap])
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    point_kwargs = {
        'marker': 'o',
        'markersize': 3,
    }

    nose_idx = dataset_keypoints.index('nose')

    nose, nose_valid = (kps_3d[:3, nose_idx], valid_3d[nose_idx])
    mid_shoulder, mid_shoulder_valid = (kps_3d[:3, len(dataset_keypoints)], valid_3d[len(dataset_keypoints)])
    mid_hip, mid_hip_valid = (kps_3d[:3, len(dataset_keypoints) + 1], valid_3d[len(dataset_keypoints) + 1])
    sc_mid_shoulder = kps_3d[3, len(dataset_keypoints)]
    sc_mid_hip = kps_3d[3, len(dataset_keypoints) + 1]
    
    if sc_mid_shoulder > kp_thresh and kps_3d[3, nose_idx] > kp_thresh:
        p1, valid1 = mid_shoulder, mid_shoulder_valid
        p2, valid2 = nose, nose_valid
        if valid1 and valid2:
            ax.plot([p1[I[0]], p2[I[0]]], [p1[I[1]], p2[I[1]]], [p1[I[2]], p2[I[2]]],
            color=colors[len(kp_lines)])
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        p1, valid1 = mid_shoulder, mid_shoulder_valid
        p2, valid2 = mid_hip, mid_hip_valid
        if valid1 and valid2:
            ax.plot([p1[I[0]], p2[I[0]]], [p1[I[1]], p2[I[1]]], [p1[I[2]], p2[I[2]]],
            color=colors[len(kp_lines) + 1])

    # Draw the keypoints.
    for l in xrange(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1, valid1 = kps_3d[:3, i1], valid_3d[i1]
        p2, valid2 = kps_3d[:3, i2], valid_3d[i2]
        if kps_3d[3, i1] > kp_thresh and kps_3d[3, i2] > kp_thresh and valid1 and valid2:
            ax.plot([p1[I[0]], p2[I[0]]], [p1[I[1]], p2[I[1]]], [p1[I[2]], p2[I[2]]], color=colors[l])
        if kps_3d[3, i1] > kp_thresh and valid1:
            ax.plot([p1[I[0]]], [p1[I[1]]], [p1[I[2]]], color=colors[l], **point_kwargs)
        if kps_3d[3, i2] > kp_thresh and valid2:  
            ax.plot([p2[I[0]]], [p2[I[1]]], [p2[I[2]]], color=colors[l], **point_kwargs)

def plane_to_cam(p, Z, k, shrink_factor=1):
    """Map pixels and corresponding depth to 3D camera coordinates
    """
    fx, cx, fy, cy = k
    u, v = p
    z = Z[v, u];
    x = (u * shrink_factor - cx) * z / fx;
    y = (v * shrink_factor - cy) * z / fy;
    return x, y, z

def cam_to_world(q, t, v):
    """Map camera coordinates to world-frame
    """
    return qv_mult(q, v) + t, v[2] > 0

def main(args):
    """Map pixels and corresponding depth to 3D camera coords
    """
    CAMERA_FRAME = 'cam-frame' in args.opts
    if CAMERA_FRAME:
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        z_min = 0
        z_max = 1
    else:
        x_min = -1.5
        x_max = 1.5
        y_min = -1.5
        y_max = 1.5
        z_min = 0
        z_max = 4
    
    if "record-kps" in args.opts:
        kps_3d_list = []

    kps_3d_arr = None
    if os.path.exists(args.kps_3d):
        kps_3d_arr = np.load(open(args.kps_3d))
    else:
        raise "{} does not exist".format(args.kps_3d)

    if args.dataset == 'tum':
        fx = 525.0  # focal length x
        fy = 525.0  # focal length y
        cx = 319.5  # optical center x
        cy = 239.5  # optical center y

        factor = 5000 # for the 16-bit PNG files

        with open(os.path.join(args.datadir, "groundtruth.txt")) as f:
            groundtruth_data = f.readlines()
        groundtruth_data = [line.strip().split() for line in groundtruth_data if not line.startswith("#")]
        groundtruth_data = [[float(v) for v in el] for el in groundtruth_data]
        gt_t_list, tx_list, ty_list, tz_list, qx_list, qy_list, qz_list, qw_list = zip(*groundtruth_data)

        with open(os.path.join(args.datadir, "depth.txt")) as f:
            depth_data = f.readlines()
        depth_data = [line.strip().split() for line in depth_data if not line.startswith("#")]
        depth_data = [(float(el[0]), el[1]) for el in depth_data]
        depth_t_list, depth_path_list = zip(*depth_data)
        
        with open(os.path.join(args.datadir, "rgb.txt")) as f:
            rgb_data = f.readlines()
        rgb_data = [line.strip().split() for line in rgb_data if not line.startswith("#")]
        rgb_data = [(float(el[0]), el[1]) for el in rgb_data]

    elif args.dataset == 'princeton':
        with open(os.path.join(args.datadir, 'frames.json')) as f:
            data = json.load(f)
            (fx, _, cx), (_, fy, cy), _ = data['K']
            factor = 1000
            rgb_data = [(float(timestamp) / 1000 / 1000, os.path.join(args.datadir, 'rgb/r-{}-{}.png'.format(timestamp, data['imageFrameID'][i]))) for i, timestamp in enumerate(data['imageTimestamp'])]
            depth_t_list = [float(timestamp) / 1000 / 1000 for timestamp in data['depthTimestamp']]
            depth_path_list = [os.path.join(args.datadir, 'depth/d-{}-{}.png'.format(timestamp, data['depthFrameID'][i])) for i, timestamp in enumerate(data['depthTimestamp'])]

    if args.mode in [1, 2]:
        all_dets = pickle.load(open(os.path.join(args.datadir, "detections.pkl")))

    titles = ["Front", "Side", "Top"]
    fig = plt.figure(figsize=(80, 60))
    ax_front = fig.add_subplot(1, 1, 1, projection='3d', proj_type='ortho')

    if CAMERA_FRAME or args.dataset == 'princeton':
        ax_front.view_init(elev=180, azim=0)
      
    else:
        ax.view_init(elev=0, azim=270)
    for i, (rgb_t, rgb_path) in enumerate(rgb_data):
        for j, ax in enumerate((ax_front,)):
            ax.cla()
            ax.set_title(titles[j])
            print("RGB timestamp  {}".format(rgb_t))
            depth_i, depth_t = min(enumerate(depth_t_list), key=lambda x: abs(x[1] - rgb_t))
            if abs(depth_t - rgb_t) > DELTA_T_MAX:
                print("WARNING: Depth timestamp could not be matched (delta {})".format(abs(depth_t - rgb_t)))
                continue
            depth_path = depth_path_list[depth_i]

            if CAMERA_FRAME or args.dataset == 'princeton':
                tx = 0
                ty = 0
                tz = 0
                qx = 0
                qy = 0
                qz = 0
                qw = 1
            else:
                gt_i, gt_t = min(enumerate(gt_t_list), key=lambda x: abs(x[1] - rgb_t))
                if abs(gt_t - rgb_t) > DELTA_T_MAX:
                    print("WARNING: GT timestamp could not be matched (delta {})".format(abs(gt_t - rgb_t)))
                    continue
                tx = tx_list[gt_i]
                ty = ty_list[gt_i]
                tz = tz_list[gt_i]
                qx = qx_list[gt_i]
                qy = qy_list[gt_i]
                qz = qz_list[gt_i]
                qw = qw_list[gt_i]

            rgb_img = cv2.imread(os.path.join(args.datadir, rgb_path))
            rgb_kps_img = cv2.imread(os.path.join(args.datadir, "dets", rgb_path.split("/")[-1].split(".")[0] + "_pred.png"))
            depth_img = cv2.imread(os.path.join(args.datadir, depth_path), -1)
            if args.dataset == 'princeton':
                depth_img = np.bitwise_or(np.right_shift(depth_img, 3), np.left_shift(depth_img, 16 - 3))
            shrink_factor = args.shrink_factor
            rgb_small = cv2.resize(rgb_img, None, fx=1. / shrink_factor, fy=1. / shrink_factor).astype(np.float32) / 255
            depth_small = cv2.resize(depth_img, None, fx=1. / shrink_factor, fy=1. / shrink_factor).astype(np.float32)

            Z = depth_small / factor

            if args.mode in [0, 2]:
                X = np.zeros(Z.shape)
                Y = np.zeros(Z.shape)
                for v in range(Z.shape[0]):
                    for u in range(Z.shape[1]):
                        x, y, z = plane_to_cam((u, v), Z, (fx, cx, fy, cy), shrink_factor)
                        X[v, u] = x
                        Y[v, u] = y

                mesh = np.dstack((X, Y, Z))
                mesh = np.apply_along_axis(lambda v: cam_to_world((qw, qx, qy, qz), np.array([tx, ty, tz]), v)[0], 2, mesh)
                X = mesh[:, :, 0]
                Y = mesh[:, :, 1]
                Z = mesh[:, :, 2]

                ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=rgb_small[...,::-1].reshape(Z.shape[0] * Z.shape[1], 3), s=np.sqrt(shrink_factor))
            if args.mode in [1, 2]:
                if kps_3d_arr is not None:
                    dets = kps_3d_arr[i]
                    for det in dets:
                        obj_id, kps_3d, valid_3d = det
                        vis_keypoints_3d(ax, kps_3d, valid_3d, obj_id)
                else:
                    p_map = lambda p: cam_to_world((qw, qx, qy, qz), np.array([tx, ty, tz]), np.array(plane_to_cam(p / shrink_factor, medfilt2d(depth_small, args.k_size) / factor, (fx, cx, fy, cy), shrink_factor)))
                    kps_3d_list_i = []
                    for det in all_dets[i]:
                        obj_id = det[1]
                        kps = det[-1]
                        kps_3d, valid_3d = map_kps_3d(kps, p_map)
                        if "record-kps" in args.opts:
                            kps_3d_list_i.append((obj_id, kps_3d, valid_3d))
                        # vis_keypoints_3d(ax, kps_3d, valid_3d, obj_id)
                    if "record-kps" in args.opts:
                        kps_3d_list.append(kps_3d_list_i)

            max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_max + x_min)
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_max + y_min)
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_max + z_min)
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(*[[Xb, Yb, Zb][idx] for idx in I]):
               ax.plot([xb], [yb], [zb], 'w')

            # Draw camera
            kwargs = {
                'color': 'grey',
                'markerfacecolor': 'k',
                'markeredgecolor': 'k',
                'marker': 'o',
                'markersize': 5,
                'alpha': 0.6,
            }
            p = qv_mult((qw, qx, qy, qz), np.array([tx - cx, ty - cy, tz + fx]))
            t = [tx, ty, tz]
            ax.plot([t[I[0]], p[I[0]]], [t[I[1]], p[I[1]]], [t[I[2]], p[I[2]]], **kwargs)
            p = qv_mult((qw, qx, qy, qz), np.array([t[I[0]] + cx, t[I[1]] - cy, t[I[2]] + fx]))
            ax.plot([t[I[0]], p[I[0]]], [t[I[1]], p[I[1]]], [t[I[2]], p[I[2]]], **kwargs)
            p = qv_mult((qw, qx, qy, qz), np.array([t[I[0]] - cx, t[I[1]] + cy, t[I[2]] + fx]))
            ax.plot([t[I[0]], p[I[0]]], [t[I[1]], p[I[1]]], [t[I[2]], p[I[2]]], **kwargs)
            p = qv_mult((qw, qx, qy, qz), np.array([t[I[0]] + cx, t[I[1]] + cy, t[I[2]] + fx]))
            ax.plot([t[I[0]], p[I[0]]], [t[I[1]], p[I[1]]], [t[I[2]], p[I[2]]], **kwargs)

            labels = ['X', 'Y', 'Z']
            limits = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            ax.set_xlabel(labels[I[0]])
            ax.set_ylabel(labels[I[1]])
            ax.set_zlabel(labels[I[2]])
            ax.set_xlim(*limits[I[0]])
            ax.set_ylim(*limits[I[1]])
            ax.set_zlim(*limits[I[2]])

        img = np.array((depth_img.T, depth_img.T, depth_img.T)).T.astype(np.float32)
        img /= np.max(img)
        rgb_kps_img = rgb_kps_img.astype(np.float32) / 255
        img = np.hstack((img, rgb_kps_img))

        plt.pause(0.05)
        fig.savefig(os.path.join(args.datadir, 'plt', '{}.png'.format(i)), bbox_inches='tight', dpi=fig.dpi)
        if 'auto-play' not in args.opts:
            raw_input()

    if "record-kps" in args.opts:
        np.save(open(os.path.join(args.datadir, 'kps_3d.npy'), 'w'), np.array(kps_3d_list))


if __name__ == '__main__':
    args = parse_args()
    main(args)
