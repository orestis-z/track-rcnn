"""
This script converts detections provided by the MOT Benchmark to COCO compatible
object proposals.
"""

import argparse
import os
import sys
import pickle
import numpy as np
from itertools import groupby


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir',
        dest='dataset_dir',
        help="Directory of the MOT dataset",
        default=None,
        type=str
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


def mot_detections_to_proposals(det_path):
    """Converts MOT detections to proposals format"""
    n_images = len([name for name in os.listdir(os.path.join(det_path, "img1")) \
            if os.path.isfile(os.path.join(det_path, "img1", name))])
    empty_ids = [np.empty([]) for _ in xrange(n_images)]
    empty_scores = [np.array([]) for _ in xrange(n_images)]
    empty_boxes = [np.empty((0, 4)) for _ in xrange(n_images)]

    # proposals['ids][i] - id for the i-th image
    # proposals['scores'][i] - ndarray (N_i,) containing scores for the proposals in the i-th image
    # proposals['boxes'][i]  - ndarray (N_i, 4) containing coordinates (x1, y1, x2, y2) for the proposals in the i-th image
    proposals = {
        "ids": empty_ids,
        "scores": empty_scores,
        "boxes": empty_boxes,
    }
    
    # In the following we extract the detections from the detection file
    # and convert it to the COCO compatible format described above
    lines = [line.strip().split(",") for line in open(os.path.join(det_path, "det/det.txt"))]
    temp = [{
        "image_id": int(fields[0]),
        "bbox": [float(fields[2]), float(fields[3]), float(fields[2]) + float(fields[4]),
            float(fields[3]) + float(fields[5])],
        "conf": float(fields[6]),
        "id": i,
    } for i, fields in enumerate(lines)]
    temp = [(key, tuple(group)) for key, group in groupby(temp, key=lambda x: x['image_id'])]
    temp = { key: {
        "ids": np.array([item["id"] for item in group]),
        "scores": np.array([item["conf"] for item in group]),
        "boxes": np.array([item["bbox"] for item in group]),
        } for key, group in temp }
    for key, val in temp.items():
        for met in ["ids", "scores", "boxes"]:
            proposals[met][key - 1] = val[met]

    dest_path = os.path.join(det_path, "det", "proposals.pkl")

    # Write proposals to COCO compatible proposals file
    with open(dest_path, 'w') as fp:
        pickle.dump(proposals, fp)


if __name__ == '__main__':
    from detectron.datasets.dataset_catalog import _mot_test_sequence_idx, _mot_detectors

    args = parse_args()
    # Iterate through all detectors and sequences
    for detector in _mot_detectors:
        for seq in _mot_train_sequence_idx:
            mot_detections_to_proposals(os.path.join(args.dataset_dir, "train/MOT17-{}-{}".format(seq, detector)))
        for seq in _mot_test_sequence_idx:
            mot_detections_to_proposals(os.path.join(args.dataset_dir, "test/MOT17-{}-{}".format(seq, detector)))
