# Tracking Extension

This file documents the tracking extension and multi-task framework developed as part of Orestis Zambounis' Master Thesis "_Multitask CNN Architecture for Online 3D Human Pose Estimation and Multi-person Tracking_".

In the following we provide important command line snippets for training, validation and inference as a starting point to familiarize with the extension to the original Detectron framework.

## Training

### Dataset

The MOT17 Benchmark dataset (or a symlink to it) has to be placed under `detectron/datasets/data`

Next, we have to convert the MOT ground truth annotations to a COCO compatible format:
```
python tools/convert_mot_to_coco.py --datadir path/to/MOT17/train
```

### Run training

Finally, run the training with
```
python tools/train_net.py --cfg configs/tracking/tracking_cfg.yaml
```

### Inspection

Inspect the training loss using
```
tensorboard --logdir outputs
```

### Pre-compute blobs

Save selected blobs to storage to possibly speed up training time.

Adapted from `scripts/save_tracking_blobs.sh`:

```
for seq in "02" "04" "05" "09" "10" "11" "13"; do
    ipython2 --pdb tools/save_blobs.py -- --wts path/to/weights/file --cfg path/to/cfg --blobs [blob-list] --output-dir path/to/output/${seq}/ --dataset mot17_train_frcnn_${seq}
done;
```

## Validation

### Full sequence validation using MOT metrics

Run validation on the full sequences to calculate MOT metrics on all saved models from a specific configuration:
```
python tools/test_tracking.py ---cfg path/to/config.yaml proposals eval
```

### MOT metrics visualization

The previous command will write validation results to the model directory which can be visualized using:
```
python --pdb tools/visualize_mot_val.py --model-dir outputs/test/.../generalized_rcnn/
```

## Inference

### Custom image sequence

Basic:
```
python tools/infer_track_sequence.py --wts path/to/weights --cfg path/to/config --im-dir path/to/image/sequence --n-colors 10 show-track
```

Merging weights from mulitple files for multi-task inference:

```
python2 tools/infer_track_sequence.py --wts path/to/weights/tracking path/to/weights/kps --cfg configs/tracking/siamese-cfg.yaml --preffixes "" sia --im-dir path/to/image/sequence --n-colors 10 show-track
```

### Evaluate MOT sequence

First, we convert proposals provided by the MOT benchmark to a COCO compatible format using:
```
python tools/convert_mot_detetections_to_proposals.py --datadir path/to/parent/directory/of/MOT/detections
```

Custom proposals (`FASTER_RCNN` must be set to `False` in the config): 
```
python tools/test_tracking.py --cfg path/to/config.yaml --model model_iterX.pkl proposals eval
```

### 3D Keypoints

Run inference on custom image sequence:
```
python tools/infer_track_sequence.py --wts path/to/weights/tracking path/to/weights/kps --cfg configs/tracking/cfg-siamese.yaml --preffixes "" sia --im-dir .../Princeton\ Tracking\ Benchmark/EvaluationSet/${folder}/rgb --n-colors 2 --output-dir .../Princeton\ Tracking\ Benchmark/EvaluationSet/${folder}/dets --output-file .../Princeton\ Tracking\ Benchmark/EvaluationSet/${folder}/detections.pkl all-dets show-track
```

Map keypoints to the depth and transform to world coordinates:
```
python tools/3D_inference/vis_rgbd.py --datadir .../Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/ --dataset princeton --mode 1 --shrink-factor 1 --k-size 1 --kps-3d .../Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/kps_3d.npy auto-play record-kps
```

Filter keypoints:
```
python2 tools/3D_inference/filter_kps.py --kps-3d .../Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/kps_3d.npy --output-dir .../Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/ --filter median --filter-var 5
```

Visualize using the filtered keypoints:
```
python tools/3D_inference/vis_rgbd.py --datadir .../Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/ --dataset princeton --kps-3d ~/datasets/Princeton\ Tracking\ Benchmark/EvaluationSet/three_people/kps_3d.npy
```
