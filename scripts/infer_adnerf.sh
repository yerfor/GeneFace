CUDA_VISIBLE_DEVICES=0 python inference/nerfs/adnerf_infer.py \
    --config=checkpoints/May/adnerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/May_train.wav,infer_out_video_name=infer_out/ADNeRF/May_train.mp4
