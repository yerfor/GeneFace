export CUDA_VISIBLE_DEVICES=0
export Video_ID=May

python tasks/run.py --config=egs/datasets/videos/${Video_ID}/audio2pose.yaml \
    --exp_name=${Video_ID}/audio2pose \
    --reset
