export Video_ID=May
# binarize the dataset
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/${Video_ID}/lm3d_nerf.yaml
# train Head NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/${Video_ID}/lm3d_nerf.yaml --exp_name=${Video_ID}/lm3d_nerf --reset
# train Torso NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/${Video_ID}/lm3d_nerf_torso.yaml --exp_name=${Video_ID}/lm3d_nerf_torso --reset