import argparse
import os
import sys
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        help="data dir",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


PERS = False

def mot_detections_to_proposals(det_path):
    proposals = {
        'ids': [None for _ in xrange(2000)],
        'scores': [None for _ in xrange(2000)],
        'boxes': [None for _ in xrange(2000)],
    }

    with open(det_path, 'r') as fp:      
        ids = []
        scores = []
        boxes = []

        init = True
        for i, line in enumerate(fp):
            fields = line.strip().split(",")
            image_id = int(fields[0])
            if i == 0:
                image_id_old = image_id
            bbox = [float(field) for field in fields[2:6]]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            conf = float(fields[6])
            if image_id != image_id_old:
                assert(len(boxes))
                proposals['ids'][image_id - 1] = np.array(ids)
                proposals['scores'][image_id - 1] = np.array(scores)
                proposals['boxes'][image_id - 1] = np.array(boxes)
                ids = []
                scores = []
                boxes = []
                image_id_old = image_id
            ids.append(i)
            scores.append(conf)
            boxes.append(bbox)

    with open(os.path.join("/".join(det_path.split("/")[:-1]), "proposals.pkl"), 'w') as fp:
        pickle.dump(proposals, fp)

if __name__ == '__main__':
    from detectron.datasets.dataset_catalog import _mot_train_sequence_idx, _mot_detectors

    args = parse_args()
    for detector in _mot_detectors:
        for seq in _mot_train_sequence_idx:
            mot_detections_to_proposals("{}/MOT17-{}-{}/det/det.txt".format(args.datadir, seq, detector))
