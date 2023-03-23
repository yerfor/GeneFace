export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1 # now we support multi-gpu inference!
export Video_ID=May
export Wav_ID=zozo # the .wav file should locate at `data/raw/val_wavs/<wav_id>.wav`

python inference/nerfs/lm3d_radnerf_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_radnerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}_radnerf_torso_smo.mp4\
    --infer

