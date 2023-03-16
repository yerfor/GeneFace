export Video_ID=May
# binarize the dataset
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/${Video_ID}/lm3d_radnerf.yaml
# train Head NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/${Video_ID}/lm3d_radnerf.yaml --exp_name=${Video_ID}/lm3d_radnerf --reset
# train Torso NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/${Video_ID}/lm3d_radnerf_torso.yaml --exp_name=${Video_ID}/lm3d_radnerf_torso --reset