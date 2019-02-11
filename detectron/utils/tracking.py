from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from collections import deque
import sys, os
import cv2
import pickle


from scipy.optimize import linear_sum_assignment
import numpy as np
from caffe2.python import workspace
from caffe2.python import core

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

        keep_idx = []
        if boxes is None:
            boxes = np.array([])
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
            lost_detections = [det_prev for det_prev in detections_prev if det_prev.obj_id not in [det.obj_id for det in detections]]
            self.lost_detections_deque.append(lost_detections)
            logger.debug("New:", [det.obj_id for det in self.new_detections])
            logger.debug("Lost:", [det.obj_id for det in lost_detections])
        else:
            detections = [Detection(classes[i], box, segms[i], keypoints[i], self.i_frame, i, i) for i, box in enumerate(boxes)]
            for det in detections:
                det.conf_prev = 1
            self.obj_id_counter = len(detections)
            self.new_detections = detections
            logger.debug("New:", [det.obj_id for det in self.new_detections])
        
        self.extras_deque.append(extras)
        self.detection_list.append(detections)
        self.keep_indices.append(keep_idx)
        self.i_frame += 1

    def assign(self, detections_prev, detections, track_mat):
        # calculate optimal assignments
        assign_inds_prev, assign_inds = linear_sum_assignment(-track_mat)

        # associate detections
        assigned_inds = []
        assigned_inds_prev = []
        for i, assign_i in enumerate(assign_inds):
            assign_i_prev = assign_inds_prev[i]
            detection = detections[assign_i]
            detection_prev = detections_prev[assign_i_prev]
            conf = track_mat[assign_i_prev, assign_i]
            if conf:
                assigned_inds_prev.append(assign_i_prev)
                assigned_inds.append(assign_i)
                detection.associate_prev(detection_prev, conf)
                detection_prev.associate_next(detection, conf)
        return assigned_inds_prev, assigned_inds

    def get_associations(self, i_frame, mot=False):
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
                detection_list.append([i_frame + 1, detection.obj_id + 1, bbox[0], bbox[1], bbox_width, bbox_height, 1 if detection.conf_prev > 0 else 0, -1, -1])
            else:
                detection_list.append([i_frame, detection.obj_id, detection.new, detection.box, detection.conf_prev, detection.cls, detection.segm, detection.kps])
        return detection_list

    def get_all_associations(self, mot=False):
        detection_list = []
        for i in xrange(len(self.detection_list)):
            detection_list.extend(self.get_associations(i, mot))
        return detection_list

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
    detection_prev = None
    detection_next = None
    conf_next = 0
    conf_prev = 0
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
        self.detection_next = detection
        self.conf_next = conf

    def associate_prev(self, detection, conf):
        # assert self.obj_id is None
        assert detection.obj_id is not None
        self.obj_id = detection.obj_id
        self.detection_prev = detection
        self.conf_prev = conf
        self.new = False


def infer_track_sequence(model, im_dir, tracking, proposals=None, vis=None, det_file=None, mot=True):
    im_names = os.listdir(im_dir)
    assert len(im_names) > 1, "Sequence must contain > 1 images"
    im_names.sort()
    # im_names.sort(key=lambda el: int(el.split('-')[1]))
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
                else:
                    cls_boxes, cls_segms, cls_keyps, track_mat, extras = im_detect_all_seq(
                        model,
                        im,
                        proposal_boxes,
                        (cls_boxes_prev, fpn_res_sum_prev, boxes_prev, im_scale_prev)
                    )
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
                    dataset=vis['dummy-dataset'],
                    show_class=vis['show-class'],
                    show_track=vis['show-track'],
                    show_box=True,
                    thresh=vis['thresh'],
                    kp_thresh=vis['kp-thresh'],
                    track_thresh=vis['track-thresh'],
                    n_colors=vis['n-colors'],
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
            track_mat = np.where(np.bitwise_and(np.array([[cls_prev == cls for cls in classes] for cls_prev in classes_prev]), track_mat > cfg.TRCNN.DETECTION_THRESH), track_mat, np.zeros((n_rois, m_rois)))
        assigned_inds_prev, assigned_inds = tracking.assign(detections_prev, detections, track_mat)

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
        assign_inds_prev = [j for j, det in enumerate(detections_prev) if j not in assigned_inds_prev]
        detections_prev = [det for j, det in enumerate(detections_prev) if j in assign_inds_prev]
        # Assign back starting from second last
        tracking.lost_detections_deque[-(i + 2)] = detections_prev

        if not len(detections):
            break

def get_matlab_engine(devkit_path):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(devkit_path)
    return eng

def eval_detections_matlab(eng, seq_map, res_dir, gt_data_dir, benchmark):
    m = eng.evaluateTracking(seq_map, res_dir, gt_data_dir, benchmark)['m']
    return m


if __name__ == '__main__':
    eng = get_matlab_engine("~/repositories/motchallenge-devkit")
    eval_detections_matlab(eng, 'c10-train.txt', '~/repositories/detectron/outputs/', '~/datasets/MOT17/train/', 'MOT17')
