export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # now we support multi-gpu inference!
export Video_ID=May
export Wav_ID=zozo

python inference/nerfs/adnerf_infer.py \
    --config=checkpoints/${Video_ID}/adnerf_torso/config.yaml \
    # --config=checkpoints/${Video_ID}/adnerf/config.yaml \ # you can also only render the head part
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,infer_out_video_name=infer_out/ADNeRF/${Video_ID}/pred_video/${Wav_ID}.mp4 \
    --reset
