# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-13.yaml --start-at 540000 eval; ipython2 --pdb tools/visualize_mot_val.py -- --model-dir outputs/test/mot17_train_frcnn_13/generalized_rcnn/ --smooth-sigma 2

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN.yaml show-track proposals vis eval

# ; ipython2 --pdb tools/visualize_mot_val.py -- --model-dir outputs/test/mot17_train_dpm_09:mot17_train_dpm_10_B=256_F=512/generalized_rcnn/ --smooth-sigma 2

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-09-10.yaml --skip 4 --start-at 105000 proposals eval ; ipython2 --pdb tools/visualize_mot_val.py -- --model-dir outputs/test/mot17_train_dpm_09:mot17_train_dpm_10/generalized_rcnn/ --smooth-sigma 2

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-09-10.yaml --model model_iter99999.pkl --model-suffix _B=256_F=512 proposals eval ; ipython2 --pdb tools/visualize_mot_val.py -- --model-dir outputs/test/mot17_train_dpm_09:mot17_train_dpm_10/generalized_rcnn/detection_thresh --smooth-sigma 2

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN.yaml --model model_iter99999.pkl --model-suffix _B=256_F=512 proposals vis show-track

# ipython2 --pdb tools/visualize_mot_val.py -- --model-dir outputs/test/mot17_train_dpm_09:mot17_train_dpm_10_B=256_F=512/generalized_rcnn/  outputs/test/mot17_train_dpm_09:mot17_train_dpm_10/generalized_rcnn/   --smooth-sigma 2 --iter-max 600000

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-09-10.yaml --model-suffix _B=256_F=512 --start-at 48000 --skip 9 proposals eval

ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-09-10.yaml --model model_iter49999.pkl --skip 9 --start-at 30000 proposals eval

# ipython --pdb tools/test_tracking.py -- --cfg configs/tracking/track_rcnn_R-50-FPN-09-10.yaml --model model_iter49999.pkl --skip 9 --model-suffix _B=256_F=512 proposals eval
