export Video_ID=May
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/${Video_ID}/lm3d_postnet_sync.yaml --exp_name=${Video_ID}/lm3d_postnet_sync --reset
