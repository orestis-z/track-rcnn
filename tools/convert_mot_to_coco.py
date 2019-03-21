"""Script to convert the MOT ground truth annotations to a COCO compatible format
"""

import argparse
import os
import sys
from configparser import ConfigParser
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--datadir', help="data dir",
        default=None, type=str)
    parser.add_argument(
        '--min_vis',
        dest='min_vis',
        help="data dir",
        default=0.4, type=float)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


PERS = False

def mot_to_json(ini_path, min_vis):
    ini_dir = "/".join(ini_path.split("/")[:-1])
    gt = {}

    config = ConfigParser()
    config.read(ini_path)
    seq = config["Sequence"]

    gt["info"] = {
        "name": seq["name"],
        "type": "track",
        "frame_rate": seq["frameRate"],
        "seq_length": seq["seqLength"],
    }
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
    gt["images"] = []
    gt["annotations"] = []

    for img_file in os.listdir(os.path.join(ini_dir, seq["imDir"])):
        height = int(seq["imHeight"])
        img_info = {
           "file_name": img_file,
           "height": int(seq["imHeight"]), 
           "width": int(seq["imWidth"]),
           "id": int(img_file.split(".")[0]),
        }
        gt["images"].append(img_info)

    with open(os.path.join(ini_dir, "gt", "gt.txt"), 'r') as fp:
        gt_txt = ""
        for i, line in enumerate(fp):
            fields = line.split(",")
            confidence = int(fields[6])
            visibility = float(fields[8])
            if confidence == 0 or visibility < min_vis:
                continue
            cls = int(fields[7])

            if not PERS or cls in [1, 2, 7]:
                annotation = {
                    "image_id": int(fields[0]),
                    "id": i,
                    "track_id": int(fields[1]),
                    "category_id": 1 if PERS else cls,
                    "bbox": [int(field) for field in fields[2:6]],
                    "segmentation": [],
                    "iscrowd": int(fields[7]) == 13,
                    "area": int(fields[4]) * int(fields[5]),
                    "visibility": visibility
                }
                gt["annotations"].append(annotation)
                gt_txt += ",".join(fields[:7] + ["1"] + fields[8:])

    gt_str  = "gt"
    if PERS:
        gt_str += "_pers"
    with open(os.path.join(ini_dir, "gt", gt_str + ".json"), 'w') as fp:
        json.dump(gt, fp)
    if PERS:
        with open(os.path.join(ini_dir, "gt", "gt_pers.txt"), 'w') as fp:
            fp.write(gt_txt)


if __name__ == '__main__':
    from detectron.datasets.dataset_catalog import _mot_train_sequence_idx, _mot_detectors

    args = parse_args()
    for detector in _mot_detectors:
        for seq in _mot_train_sequence_idx:
            mot_to_json("{}/MOT17-{}-{}/seqinfo.ini".format(args.datadir, seq, detector), args.min_vis)
