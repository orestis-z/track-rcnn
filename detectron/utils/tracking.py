"""Tracking utilities.

This file specifies the Tracking class which accumulates and associates object detections,
the Detection class storing information for a specific object detection and tracking inference,
back-tracking and evaluation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from collections import deque
import sys, os
import cv2
import pickle
from collections import defaultdict
import time

from scipy.optimize import linear_sum_assignment
import numpy as np
from caffe2.python import workspace
from caffe2.python import core

from detectron.utils.timer import Timer
import detectron.utils.c2 as c2_utils
from detectron.core.config import cfg
from detectron.core.test import im_detect_track, multi_im_detect_all, im_detect_all_seq
import detectron.utils.vis as vis_utils

logger = logging.getLogger(__name__)
# Same line logger
logger_flush = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(levelname)s %(filename)s:%(lineno)4d: %(message)s'))
logger_flush.addHandler(ch)
logger_flush.propagate = 0


class Tracking(object):

    def __init__(self, thresh, max_back_track):
        self.extras_deque = deque(maxlen=max_back_track+2)
        self.lost_detections_deque = deque(maxlen=max_back_track+1)
        self.detection_list = []
        self.new_detections = []
        self.keep_indices = []
        self.i_frame = 0
        self.obj_id_counter = 0
        self.thresh = thresh;

    def accumulate(self, cls_boxes, cls_segms, cls_keyps, extras, track_mat=None):
        # filter out detectios with low bounding box detection threshold
        boxes, segms, keypoints, classes = convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)
        if boxes is None:
            boxes = np.array([])

        keep_idx = []
        for j, cls_box in enumerate(boxes):
            if cls_box[-1] >= self.thresh:
                keep_idx.append(j)
        boxes = boxes[keep_idx]
        segms = [segm for i, segm in enumerate(segms) if i in keep_idx] if segms else [None] * len(boxes)
        keypoints = [kp for i, kp in enumerate(keypoints) if i in keep_idx] if keypoints else [None] * len(boxes)
        classes = [cls for i, cls in enumerate(classes) if i in keep_idx]

        if track_mat is not None:
            detections = [Detection(classes[i], box, segms[i], keypoints[i], self.i_frame, keep_idx[i]) for i, box in enumerate(boxes)]

            # filter out track probs with bounding box detections below threshold
            track_temp = np.zeros((len(self.keep_indices[-1]), len(keep_idx)))
            for i, row in enumerate(self.keep_indices[-1]):
                for j, col in enumerate(keep_idx):
                    track_temp[i, j] = track_mat[row, col]
            track_mat = track_temp
            
            detections_prev = self.detection_list[-1]

            # associate detections
            self.assign(detections_prev, detections, track_mat)

            # check for new detections
            self.new_detections = []
            for detection in detections:
                if detection.obj_id is None:
                    detection.obj_id = self.obj_id_counter
                    self.obj_id_counter += 1
                    self.new_detections.append(detection)

            # check for lost detections
            lost_detections = [det_prev for det_prev in detections_prev \
                if det_prev.obj_id not in [det.obj_id for det in detections]]
            self.lost_detections_deque.append(lost_detections)
            logger.debug("New:", [det.obj_id for det in self.new_detections])
            logger.debug("Lost:", [det.obj_id for det in lost_detections])
        else:
            detections = [Detection(classes[i], box, segms[i], keypoints[i], self.i_frame, keep_idx[i], i) \
                for i, box in enumerate(boxes)]
            self.obj_id_counter = len(detections)
            self.new_detections = detections
            logger.debug("New:", [det.obj_id for det in self.new_detections])
        
        self.extras_deque.append(extras)
        self.detection_list.append(detections)
        self.keep_indices.append(keep_idx)
        self.i_frame += 1

    def assign(self, detections_prev, detections, track_mat):
        # calculate optimal assignments
        assign_inds_prev, assign_inds = linear_sum_assignment(-track_mat) # returns row_ind, col_ind

        # associate detections
        assigned_inds = []
        assigned_inds_prev = []
        for i, assign_i in enumerate(assign_inds):
            assign_i_prev = assign_inds_prev[i]
            detection = detections[assign_i]
            detection_prev = detections_prev[assign_i_prev]
            conf = track_mat[assign_i_prev, assign_i]
            if conf >= cfg.TRCNN.DETECTION_THRESH:
                assigned_inds_prev.append(assign_i_prev)
                assigned_inds.append(assign_i)
                detection.associate_prev(detection_prev, conf)
                detection_prev.associate_next(detection, conf)
        return assigned_inds_prev, assigned_inds

    def get_associations(self, i_frame, mot=False):
        """Get associations from frame `iframe`"""
        if i_frame == -1:
            i_frame = len(self.detection_list) - 1
        detection_list = []
        for detection in self.detection_list[i_frame]:
            if mot:
                if detection.cls != 1:
                    continue
                bbox = detection.box[:-1].tolist()
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                detection_list.append([i_frame + 1, detection.obj_id + 1, bbox[0], bbox[1], bbox_width,
                    bbox_height, 1 if detection.conf_prev > 0 else 0, -1, -1])
            else:
                detection_list.append([i_frame, detection.obj_id, detection.new, detection.box,
                    detection.conf_prev, detection.cls, detection.segm, detection.kps])
        return detection_list

    def get_all_associations(self, mot=False):
        """Get associations from all frames"""
        detection_list = []
        for i in xrange(len(self.detection_list)):
            detection_list.extend(self.get_associations(i, mot))
        return detection_list

    def get_trace(self):
        """Get tracking traces for visualizations"""
        trace = { det.obj_id: [] for det in self.detection_list[-1]}
        obj_ids = trace.keys()
        for detections in self.detection_list:
            for obj_id in obj_ids:
                det = next((det for det in detections if det.obj_id == obj_id), None)
                bbox_center_bottom = None
                if det is not None:
                    bbox_center_bottom = (int((det.box[0] + det.box[2] + 1.) / 2), int(det.box[3] + 0.5))
                trace[obj_id].append(bbox_center_bottom)
        return trace

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    classes = classes
    return boxes, segms, keyps, classes

class Detection(object):
    """Class containing various information for an object detection
    E.g. instance segments, human poses, object associations etc.
    """
    detection_prev = None
    detection_next = None
    conf_next = 0
    conf_prev = 1
    new = True

    cls = None
    box = None
    segm = None
    kps = None
    i_frame = None
    obj_id = None
    assign_ind = None

    def __init__(self, cls, box, segm, kps, i_frame, assign_ind, obj_id=None):
        self.cls = cls
        self.box = box
        self.segm = segm
        self.kps = kps
        self.i_frame = i_frame
        self.assign_ind = assign_ind
        self.obj_id = obj_id

    def associate_next(self, detection, conf):
        """Associate to the detection from the next frame"""
        self.detection_next = detection
        self.conf_next = conf

    def associate_prev(self, detection, conf):
        """Associate to the detection from the previous frame"""
        assert detection.obj_id is not None
        self.obj_id = detection.obj_id
        self.detection_prev = detection
        self.conf_prev = conf
        self.new = False


def infer_track_sequence(model, im_dir, tracking, proposals=None, vis=None, det_file=None, mot=True):
    """Image sequence inference method. Includes optional visualizations
    and tracking detection output file creation"""
    im_names = os.listdir(im_dir)
    assert len(im_names) > 1, "Sequence must contain > 1 images"
    im_names.sort()
    im_names.sort(key=lambda el: int(el.split('-')[1])) # alternative sorting
    im_paths = [os.path.join(im_dir, im_name) for im_name in im_names]
    im_names = [".".join(im_name.split(".")[:-1]) for im_name in im_names]

    dets_full = []

    if vis is not None:
        if not os.path.exists(vis['output-dir']):
            os.makedirs(vis['output-dir'])

    with open(det_file, "w+") if det_file is not None else None as output_file:
        for i, im_path in enumerate(im_paths):
            logger_flush.info("Processing {}".format(im_path))
            im = cv2.imread(im_path)
            with c2_utils.NamedCudaScope(0):
                proposal_boxes = proposals and proposals['boxes'][i]
                # Single image detections on first frame
                if i == 0:
                    cls_boxes_list, cls_segms_list, cls_keyps_list, track_mat, extras = multi_im_detect_all(
                        model,
                        [im],
                        [proposal_boxes],
                        tracking=False,
                    )
                    extras = [l[0] for l in extras]
                    cls_boxes = cls_boxes_list[0]
                    cls_segms = cls_segms_list[0]
                    cls_keyps = cls_keyps_list[0]
                # Sequential detections after the first frame
                else:
                    timers = defaultdict(Timer)
                    timers = None
                    t = time.time()
                    cls_boxes, cls_segms, cls_keyps, track_mat, extras = im_detect_all_seq(
                        model,
                        im,
                        proposal_boxes,
                        (cls_boxes_prev, fpn_res_sum_prev, boxes_prev, im_scale_prev),
                        timers=timers,
                    )
                    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                    if timers is not None:
                        for k, v in timers.items():
                            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
                im_scale, boxes, fpn_res_sum = extras
            cls_boxes_prev = cls_boxes
            cls_segms_prev = cls_segms
            cls_keyps_prev = cls_keyps
            im_scale_prev = im_scale
            boxes_prev = boxes
            fpn_res_sum_prev = fpn_res_sum

            tracking.accumulate(cls_boxes, cls_segms, cls_keyps, extras, track_mat)
            back_track(model, tracking)

            if vis is not None:
                im_pred = vis_utils.vis_detections_one_image_opencv(
                    im,
                    detections=tracking.detection_list[-1],
                    detections_prev=tracking.detection_list[-2] if i > 0 else [],
                    trace=tracking.get_trace(),
                    dataset=vis.get('dummy-dataset'),
                    show_class=vis.get('show-class'),
                    show_track=vis.get('show-track'),
                    show_box=True,
                    thresh=vis.get('thresh'),
                    kp_thresh=vis.get('kp-thresh'),
                    track_thresh=vis.get('track-thresh'),
                    n_colors=vis.get('n-colors'),
                    track_mat=track_mat,
                )
                cv2.imwrite("{}/{}_pred.png".format(vis['output-dir'], im_names[i]), im_pred)

            if output_file is not None:
                if mot:
                    for association in tracking.get_associations(-1, True):
                        output_file.write(",".join([str(x) for x in association]) + "\n")
                else:
                    dets_full.append(tracking.get_associations(-1, False))

        if not mot and output_file is not None:
            pickle.dump(dets_full, output_file)


def back_track(model, tracking):
    """Back tracking method for object re-identification"""
    detections = tracking.new_detections

    if not len(detections):
        return

    # reverse list and begin with second last
    lost_detections_list = list(tracking.lost_detections_deque)[-2::-1]
    # reverse list and begin with third last
    extras_list = list(tracking.extras_deque)[-3::-1]

    im_scale, boxes_raw, fpn_res_sum = tracking.extras_deque[-1]
    # Filter out new detections
    assign_inds = [det.assign_ind for det in detections]
    boxes = boxes_raw[assign_inds]
    classes = [det.cls for det in detections]
    m_rois = len(assign_inds)

    # Search for matching pairs in previously lost detections
    for i, lost_detections in enumerate(lost_detections_list):
        if not len(lost_detections):
            continue

        # Filter out detections
        im_scale_lost, boxes_lost, fpn_res_sum_lost = extras_list[i]
        assign_inds_lost = [det.assign_ind for det in lost_detections]
        boxes_lost = boxes_lost[assign_inds_lost]
        classes_lost = [det.cls for det in lost_detections]
        n_rois = len(assign_inds_lost)

        # Merge fpn_res_sums
        for blob_name, fpn_res_sum_lost_val in fpn_res_sum_lost.items():
            workspace.FeedBlob(core.ScopedName(blob_name), np.concatenate((
                fpn_res_sum_lost_val,
                fpn_res_sum[blob_name]
            )))
        # Compute matches
        with c2_utils.NamedCudaScope(0):
            track = im_detect_track(model, [im_scale_lost, im_scale], [boxes_lost, boxes], [fpn_res_sum_lost, fpn_res_sum])
            track_mat = track.reshape((n_rois, m_rois))
            track_mat = np.where(
                np.bitwise_and(
                    np.array([[cls_lost == cls for cls in classes] for cls_lost in classes_lost]),
                    track_mat >= cfg.TRCNN.DETECTION_THRESH),
                track_mat, np.zeros((n_rois, m_rois)))
        assigned_inds_lost, assigned_inds = tracking.assign(lost_detections, detections, track_mat)

        logger.debug("Back tracking level {}:".format(i), [det.obj_id for j, det in enumerate(detections) if j in assigned_inds])

        # Filter out newly assigned detections
        assign_inds = [j for j, det in enumerate(detections) if j not in assigned_inds]
        detections = [det for j, det in enumerate(detections) if j in assign_inds]
        # Assign back
        tracking.new_detections = detections
        boxes = boxes_raw[assign_inds]
        classes = [det.cls for det in detections]
        m_rois = len(assign_inds)

        # Filter out newly assigned lost detections
        assign_inds_lost = [j for j, det in enumerate(lost_detections) if j not in assigned_inds_lost]
        lost_detections = [det for j, det in enumerate(lost_detections) if j in assign_inds_lost]
        # Assign back starting from second last
        tracking.lost_detections_deque[-(i + 2)] = lost_detections

        if not len(detections):
            break

def get_matlab_engine(devkit_path):
    """Get the matlab engine."""
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(devkit_path)
    return eng

def eval_detections_matlab(eng, seq_map, res_dir, gt_data_dir, benchmark):
    """Evaluate detections with the matlab MOT devkit."""
    try:
        eng.evaluateTracking(seq_map, res_dir, gt_data_dir, benchmark)
    except:
        pass
    # m = eng.evaluateTracking(seq_map, res_dir, gt_data_dir, benchmark)["m"]
    # return m


if __name__ == '__main__':
    eng = get_matlab_engine("~/repositories/motchallenge-devkit")
    eval_detections_matlab(eng, 'c10-train.txt', '~/repositories/detectron/outputs/', '~/datasets/MOT17/train/', 'MOT17')
