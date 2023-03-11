VIDEO_ID=$1
GPU_ID=$2

CUDA_VISIBLE_DEVICES=$GPU_ID python tasks/run.py --config=egs/datasets/videos/${VIDEO_ID}/lm3d_radnerf.yaml --exp_name=${VIDEO_ID}/lm3d_radnerf --reset
CUDA_VISIBLE_DEVICES=$GPU_ID python tasks/run.py --config=egs/datasets/videos/${VIDEO_ID}/lm3d_radnerf_torso.yaml --exp_name=${VIDEO_ID}/lm3d_radnerf_torso --reset
