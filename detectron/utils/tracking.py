from collections import deque

from scipy.optimize import linear_sum_assignment
import numpy as np


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
            print("New:", [det.obj_id for det in self.new_detections])
            print("Lost:", [det.obj_id for det in lost_detections])
        else:
            detections = [Detection(classes[i], box, segms[i], keypoints[i], self.i_frame, i, i) for i, box in enumerate(boxes)]
            for det in detections:
                det.conf_prev = 1
            self.obj_id_counter = len(detections)
            self.new_detections = detections
            print("New:", [det.obj_id for det in self.new_detections])
        
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
                detection_list.append([i_frame, detection.obj_id, detection.new, detection.box, detection.conf_prev])
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
