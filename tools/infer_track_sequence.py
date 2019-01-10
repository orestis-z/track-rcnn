import logging
import argparse
import cv2
import os, sys
import pprint
import numpy as np

from caffe2.python import workspace
from caffe2.python import core

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as infer_engine
from detectron.core.test import im_detect_track
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.utils.tracking import Tracking

c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights_list',
        help='list of weights model files (/path/to/model_weights.pkl)',
        default=None,
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--preffixes',
        dest='preffix_list',
        help='preffixes for the corresponding weights file',
        default=[],
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: outputs/infer_track_sequence)',
        default='outputs/infer_track_sequence',
        type=str
    )
    parser.add_argument(
        '--output-file',
        dest='output_file',
        help='file for detections (default: outputs/infer_track_sequence/detections.txt)',
        default='outputs/infer_track_sequence/detections.txt',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--track-thresh',
        dest='track_thresh',
        help='Threshold for visualizing matches',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--im-dir',
        dest='im_dir',
        help='image or folder of images',
        default=None
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

def back_track(model, tracking):
    detections = tracking.new_detections

    if not len(detections):
        return

    # reverse list and begin with second last
    lost_detections_list = list(tracking.lost_detections_deque)[-2::-1]
    # reverse list and begin with third last
    extras_list = list(tracking.extras_deque)[-3::-1]

    # assert len(lost_detections_list) == len(extras_list)

    im_scale, boxes_raw, fpn_res_sum = tracking.extras_deque[-1]
    # Filter out new detections
    assign_inds = [det.assign_ind for det in detections]
    boxes = boxes_raw[assign_inds]
    classes = [det.cls for det in detections]
    m_rois = len(assign_inds)

    # Search for matching pairs in previously lost detections
    for i, detections_prev in enumerate(lost_detections_list):
        if not len(detections_prev):
            continue

        # Filter out detections
        im_scale_prev, boxes_prev, fpn_res_sum_prev = extras_list[i]
        assign_inds_prev = [det.assign_ind for det in detections_prev]
        boxes_prev = boxes_prev[assign_inds_prev]
        classes_prev = [det.cls for det in detections_prev]
        n_rois = len(assign_inds_prev)

        # import pdb; pdb.set_trace();

        # Merge fpn_res_sums
        for blob_name, fpn_res_sum_prev_val in fpn_res_sum_prev.items():
            workspace.FeedBlob(core.ScopedName(blob_name), np.concatenate((
                fpn_res_sum_prev_val,
                fpn_res_sum[blob_name]
            )))
        # Compute matches
        with c2_utils.NamedCudaScope(0):
            track = im_detect_track(model, [im_scale_prev, im_scale], [boxes_prev, boxes], [fpn_res_sum_prev, fpn_res_sum])
            track_mat = track.reshape((n_rois, m_rois))
            track_mat = np.where(np.bitwise_and(np.array([[cls_prev == cls for cls in classes] for cls_prev in classes_prev]), track_mat > TRCNN.DETECTION_THRESH), track_mat, np.zeros((n_rois, m_rois)))
        assigned_inds_prev, assigned_inds = tracking.assign(detections_prev, detections, track_mat)

        print("Back tracking level {}:".format(i), [det.obj_id for j, det in enumerate(detections) if j in assigned_inds])

        # Filter out newly assigned detections
        assign_inds = [j for j, det in enumerate(detections) if j not in assigned_inds]
        detections = [det for j, det in enumerate(detections) if j in assign_inds]
        # Assign back
        tracking.new_detections = detections
        boxes = boxes_raw[assign_inds]
        classes = [det.cls for det in detections]
        m_rois = len(assign_inds)

        # Filter out newly assigned lost detections
        assign_inds_prev = [j for j, det in enumerate(detections_prev) if j not in assigned_inds_prev]
        detections_prev = [det for j, det in enumerate(detections_prev) if j in assign_inds_prev]
        # Assign back starting from second last
        tracking.lost_detections_deque[-(i + 2)] = detections_prev

        if not len(detections):
            break


def main(args):
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    im_names = os.listdir(args.im_dir)
    assert len(im_names) > 1, "Sequence must contain > 1 images"
    im_names.sort()
    im_paths = [os.path.join(args.im_dir, im_name) for im_name in im_names]
    im_names = [im_name.split(".")[0] for im_name in im_names]
    
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    if "mot-classes" in args.opts:
        dummy_dataset = dummy_datasets.get_mot_dataset()
        cfg.NUM_CLASSES = 14
    else:
        dummy_dataset = dummy_datasets.get_coco_dataset()
        cfg.NUM_CLASSES = 81
    for i, weights_file in enumerate(args.weights_list):
        args.weights_list[i] = cache_url(weights_file, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    preffix_list = args.preffix_list if len(args.preffix_list) else [""] * len(args.weights_list)
    model = infer_engine.initialize_mixed_model_from_cfg(args.weights_list, preffix_list=preffix_list)

    tracking = Tracking(args.thresh, cfg.TRCNN.MAX_BACK_TRACK)
    im_list = [cv2.imread(im_path) for im_path in im_paths[:2]]
    with c2_utils.NamedCudaScope(0):
        print("Processing {}".format(im_paths[0]))
        print("Processing {}".format(im_paths[1]))
        cls_boxes_list, cls_segms_list, cls_keyps_list, track_mat_i, extras = infer_engine.multi_im_detect_all(
            model, im_list, [None, None])
    im_scale_list, boxes_list, fpn_res_sum_list = extras
    for i in [0, 1]:
        tracking.accumulate(cls_boxes_list[i], cls_segms_list[i], cls_keyps_list[i], [im_scale_list[i], boxes_list[i], fpn_res_sum_list[i]], track_mat_i if i else None)
    fpn_res_sum_prev = fpn_res_sum_list[1]
    cls_boxes_prev = cls_boxes_list[1]
    cls_segms_prev = cls_segms_list[1]
    cls_keyps_prev = cls_keyps_list[1]
    boxes_prev = boxes_list[1]
    im_scale_prev = im_scale_list[1]

    with open(args.output_file, "w+") as output_file:
        for i, im in enumerate(im_list):
            im = vis_utils.vis_detections_one_image_opencv(
                im,
                detections=tracking.detection_list[i],
                detections_prev=tracking.detection_list[0] if i == 1 else [],
                dataset=dummy_dataset,
                show_class=('show-class' in args.opts),
                show_track=('show-track' in args.opts),
                show_box=True,
                thresh=args.thresh,
                kp_thresh=args.kp_thresh,
                track_thresh=args.track_thresh,
            )
            cv2.imwrite("{}/{}_pred.png".format(args.output_dir, im_names[i]), im)
            for ass in tracking.get_associations(i, True):
                output_file.write(",".join([str(x) for x in ass]) + "\n")

        for i, im_path in enumerate(im_paths[2:]):
            im = cv2.imread(im_path)
            with c2_utils.NamedCudaScope(0):
                print("Processing {}".format(im_path))
                cls_boxes, cls_segms, cls_keyps, track_mat_i, extras = infer_engine.im_detect_all_seq(
                    model,
                    im,
                    None,
                    (cls_boxes_prev, fpn_res_sum_prev, boxes_prev, im_scale_prev)
                )
            im_scale_prev, boxes_prev, fpn_res_sum_prev = extras
            tracking.accumulate(cls_boxes, cls_segms, cls_keyps, extras, track_mat_i)
            back_track(model, tracking)

            im_pred = vis_utils.vis_detections_one_image_opencv(
                im,
                detections=tracking.detection_list[-1],
                detections_prev=tracking.detection_list[-2],
                dataset=dummy_dataset,
                show_class=('show-class' in args.opts),
                show_track=('show-track' in args.opts),
                show_box=True,
                thresh=args.thresh,
                kp_thresh=args.kp_thresh,
                track_thresh=args.track_thresh,
            )
            cv2.imwrite("{}/{}_pred.png".format(args.output_dir, im_names[i + 2]), im_pred)
            cls_boxes_prev = cls_boxes
            cls_segms_prev = cls_segms
            cls_keyps_prev = cls_keyps
            for ass in tracking.get_associations(-1, True):
                output_file.write(",".join([str(x) for x in ass]) + "\n")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
