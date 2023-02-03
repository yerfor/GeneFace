  
CUDA_VISIBLE_DEVICES=0 python inference/nerfs/lm3d_nerf_infer.py \
    --config=checkpoints/May/lm3d_nerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/zozo.wav,infer_cond_name=infer_out/May/pred_lm3d/zozo.npy,infer_out_video_name=infer_out/May/pred_video/zozo.mp4 \
    --reset
