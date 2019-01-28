import argparse
import os
import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import detectron.utils.keypoints as keypoint_utils
import detectron.utils.vis as vis_utils


fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y

factor = 5000 # for the 16-bit PNG files
# OR: factor = 1 # for the 32-bit float images in the ROS bag files

max_difference = 0.02

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        help="data dir",
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
        default=1,
        type=int
    )
    parser.add_argument(
        '--shrink-factor',
        dest='shrink_factor',
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
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):    
    q2 = [0.0] + v1.tolist()
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def vis_keypoints(ax, kps, p_map, kp_thresh=2):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = vis_utils.kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    point_kwargs = {
        'marker': 'o',
        'markersize': 3,
    }

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
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        p1, valid1 = p_map(np.rint(mid_shoulder).astype(np.int))
        p2, valid2 = p_map(np.rint(kps[:2, nose_idx]).astype(np.int))
        if valid1 and valid2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=colors[len(kp_lines)])
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        p1, valid1 = p_map(np.rint(mid_shoulder).astype(np.int))
        p2, valid2 = p_map(np.rint(mid_hip).astype(np.int))
        if valid1 and valid2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=colors[len(kp_lines) + 1])

    # Draw the keypoints.
    for l in xrange(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1, valid1 = p_map(np.rint([kps[0, i1], kps[1, i1]]).astype(np.int))
        p2, valid2 = p_map(np.rint([kps[0, i2], kps[1, i2]]).astype(np.int))
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh and valid1 and valid2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=colors[l])
        if kps[2, i1] > kp_thresh and valid1:
            ax.plot([p1[0]], [p1[1]], [p1[2]], color=colors[l], **point_kwargs)
        if kps[2, i2] > kp_thresh and valid2:  
            ax.plot([p2[0]], [p2[1]], [p2[2]], color=colors[l], **point_kwargs)


def plane_to_cam(p, Z, shrink_factor=1):
    u, v = p
    z = Z[v, u];
    x = (u * shrink_factor - cx) * z / fx;
    y = (v * shrink_factor - cy) * z / fy;
    return x, y, z

def cam_to_world(q, t, v):
    return qv_mult(q, v) + t, v[2] > 0

def main(args):
    CAMERA_FRAME = 'cam-frame' in args.opts
    if CAMERA_FRAME:
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        z_min = 0
        z_max = 1
    else:
        x_min = -2.5
        x_max = 1.5
        y_min = -2
        y_max = 1.5
        z_min = 0
        z_max = 3

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

    all_dets = pickle.load(open(os.path.join(args.datadir, "detections.pkl")))

    fig = plt.figure()
    ax = fig.gca(projection='3d', proj_type='ortho')
    ax.set_aspect('equal')
    if CAMERA_FRAME:
        ax.view_init(elev=270, azim=270)
    else:
        ax.view_init(elev=0, azim=270)

    for i, (rgb_t, rgb_path) in enumerate(rgb_data):
        print("RGB timestamp  {}".format(rgb_t))
        depth_i, depth_t = min(enumerate(depth_t_list), key=lambda x: abs(x[1] - rgb_t))
        if abs(depth_t - rgb_t) > max_difference:
            print("WARNING: Depth timestamp could not be matched (delta {})".format((depth_t - rgb_t)))
            continue
        depth_path = depth_path_list[depth_i]

        if CAMERA_FRAME:
            tx = 0
            ty = 0
            tz = 0
            qx = 0
            qy = 0
            qz = 0
            qw = 1
        else:
            gt_i, gt_t = min(enumerate(gt_t_list), key=lambda x: abs(x[1] - rgb_t))
            if abs(gt_t - rgb_t) > max_difference:
                print("WARNING: GT timestamp could not be matched (delta {})".format((gt_t - rgb_t)))
                continue
            tx = tx_list[gt_i]
            ty = ty_list[gt_i]
            tz = tz_list[gt_i]
            qx = qx_list[gt_i]
            qy = qy_list[gt_i]
            qz = qz_list[gt_i]
            qw = qw_list[gt_i]

        rgb_img = cv2.imread(os.path.join(args.datadir, rgb_path))
        depth_img = cv2.imread(os.path.join(args.datadir, depth_path), -1)
        # depth_img = cv2.GaussianBlur(depth_img, (args.k_size, args.k_size), 0)
        # W = 0 * depth.copy() + 1
        # W[depth_img == 0] = 0
        # WW = cv2.GaussianBlur(W, (args.k_size, args.k_size), 0)
        # depth /= WW
        depth_img = cv2.medianBlur(depth_img, args.k_size)
        # scaling_vals = cv2.boxFilter(depth_img, -1, (args.k_size, args.k_size), borderType=cv2.BORDER_CONSTANT)
        # depth_img = cv2.blur(depth_img, (args.k_size, args.k_size))
        shrink_factor = args.shrink_factor
        rgb_small = cv2.resize(rgb_img, None, fx=1. / shrink_factor, fy=1. / shrink_factor).astype(np.float32) / 255
        depth_small = cv2.resize(depth_img, None, fx=1. / shrink_factor, fy=1. / shrink_factor).astype(np.float32)

        plt.cla()
        Z = depth_small / factor

        if args.mode in [0, 2]:
            X = np.zeros(Z.shape)
            Y = np.zeros(Z.shape)
            for v in range(Z.shape[0]):
                for u in range(Z.shape[1]):
                    x, y, z = plane_to_cam((u, v), Z, shrink_factor)
                    X[v, u] = x
                    Y[v, u] = y

            mesh = np.dstack((X, Y, Z))
            mesh = np.apply_along_axis(lambda v: cam_to_world((qw, qx, qy, qz), np.array([tx, ty, tz]), v)[0], 2, mesh)
            X = mesh[:, :, 0]
            Y = mesh[:, :, 1]
            Z = mesh[:, :, 2]

            ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=rgb_small[...,::-1].reshape(Z.shape[0] * Z.shape[1], 3), s=np.sqrt(shrink_factor))
        if args.mode in [1, 2]:
            p_map = lambda p: cam_to_world((qw, qx, qy, qz), np.array([tx, ty, tz]), np.array(plane_to_cam(p / shrink_factor, depth_small / factor, shrink_factor)))
            for det in all_dets[i]:
                kps = det[-1]
                vis_keypoints(ax, kps, p_map)

        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x_max + x_min)
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y_max + y_min)
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z_max + z_min)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
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
        ax.plot([tx, p[0]], [ty, p[1]], [tz, p[2]], **kwargs)
        p = qv_mult((qw, qx, qy, qz), np.array([tx + cx, ty - cy, tz + fx]))
        ax.plot([tx, p[0]], [ty, p[1]], [tz, p[2]], **kwargs)
        p = qv_mult((qw, qx, qy, qz), np.array([tx - cx, ty + cy, tz + fx]))
        ax.plot([tx, p[0]], [ty, p[1]], [tz, p[2]], **kwargs)
        p = qv_mult((qw, qx, qy, qz), np.array([tx + cx, ty + cy, tz + fx]))
        ax.plot([tx, p[0]], [ty, p[1]], [tz, p[2]], **kwargs)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        plt.pause(0.005)
        if 'auto-play' not in args.opts:
            raw_input()

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)
