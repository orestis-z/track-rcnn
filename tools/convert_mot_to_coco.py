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
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def mot_to_json(ini_path):
    ini_dir = "/".join(ini_path.split("/")[:-1])
    gt = {}

    config = ConfigParser()
    config.read(ini_path)
    seq = config["Sequence"]

    gt["info"] = {"name": seq["name"], "type": "track"}
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
        for i, line in enumerate(fp):
            fields = line.split(",")
            confidence = int(fields[6])
            visibility = float(fields[8])
            if confidence == 0 or visibility < 0.4:
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

    with open(os.path.join(ini_dir, "gt", "gt.json"), 'w') as fp:
        json.dump(gt, fp)


if __name__ == '__main__':
    from detectron.datasets.dataset_catalog import _mot_train_sequence_idx, _mot_detectors

    args = parse_args()
    for detector in _mot_detectors:
        for seq in _mot_train_sequence_idx:
            mot_to_json("{}/MOT17-{}-{}/seqinfo.ini".format(args.datadir, seq, detector))

