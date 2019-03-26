"""
Script to convert the MOT ground truth annotations to a COCO
compatible format
"""

import argparse
import os
import sys
from configparser import ConfigParser
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset-dir',
        dest='dataset_dir',
        help="Directory of the MOT dataset",
        default=None,
        type=str
    )
    parser.add_argument(
        '--min_vis',
        dest='min_vis',
        help="Minimum visibility of a ground truth detection",
        default=0.4,
        type=float
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def mot_to_json(ini_path, min_vis):
    gt = {}

    # Compose sequence info
    ini_dir = "/".join(ini_path.split("/")[:-1])
    config = ConfigParser()
    config.read(ini_path)
    seq = config["Sequence"]
    gt["info"] = {
        "name": seq["name"],
        "type": "track",
        "frame_rate": seq["frameRate"],
        "seq_length": seq["seqLength"],
    }

    # Class id to name map
    gt["categories"] = [{
        'id': 1,
        'name': 'ped',
    }, {
        'id': 2,
        'name': 'person_on_vhcl',
    }, {
        'id': 3,
        'name': 'car',
    }, {
        'id': 4,
        'name': 'bicycle',
    }, {
        'id': 5,
        'name': 'mbike',
    }, {
        'id': 6,
        'name': 'non_mot_vhcl',
    }, {
        'id': 7,
        'name': 'static_person',
    }, {
        'id': 8,
        'name': 'distractor',
    }, {
        'id': 9,
        'name': 'occluder',
    }, {
        'id': 10,
        'name': 'occluder_on_grnd',
    }, {
        'id': 11,
        'name': 'occluder_full',
    }, {
        'id': 12,
        'name': 'reflection',
    }, {
        'id': 13,
        'name': 'crowd',
    }]

    # Compose image infos
    gt["images"] = []
    for img_file in os.listdir(os.path.join(ini_dir, seq["imDir"])):
        img_info = {
           "file_name": img_file,
           "height": int(seq["imHeight"]), 
           "width": int(seq["imWidth"]),
           "id": int(img_file.split(".")[0]),
        }
        gt["images"].append(img_info)

    # Compose tracking annotations
    with open(os.path.join(ini_dir, "gt", "gt.txt"), 'r') as fp:
        gt["annotations"] = []
        for i, line in enumerate(fp):
            fields = line.split(",")
            confidence = int(fields[6])
            visibility = float(fields[8])
            # Skip invalid or low-visible detections
            if confidence == 0 or visibility < min_vis:
                continue

            annotation = {
                "image_id": int(fields[0]),
                "id": i,
                "track_id": int(fields[1]),
                "category_id": int(fields[7]),
                "bbox": [int(field) for field in fields[2:6]],
                "segmentation": [],
                "iscrowd": int(fields[7]) == 13,
                "area": int(fields[4]) * int(fields[5]),
                "visibility": visibility
            }
            gt["annotations"].append(annotation)

    # Write annotation to COCO compatible json file
    with open(os.path.join(ini_dir, "gt", "gt.json"), 'w') as fp:
        json.dump(gt, fp)


if __name__ == '__main__':
    from detectron.datasets.dataset_catalog import _mot_train_sequence_idx, _mot_detectors

    args = parse_args()

    # Iterate through all detectors and training sequences
    for detector in _mot_detectors:
        for seq in _mot_train_sequence_idx:
            mot_to_json("{}/train/MOT17-{}-{}/seqinfo.ini".format(args.dataset_dir, seq, detector),
                args.min_vis)
