export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # now we support multi-gpu inference!
export Video_ID=May
export Wav_ID=zozo # the .wav file should locate at `data/raw/val_wavs/<wav_id>.wav`
export n_samples_per_ray=64 # during training 64
# export n_samples_per_ray=16 # you can use a smaller value (e.g., 16) to accelerate the inference
export n_samples_per_ray_fine=128 # during training 128
# export n_samples_per_ray_fine=32 # you can use a smaller value (e.g., 32) to accelerate the inference


# use head pose from the dataset
python inference/nerfs/lm3d_nerf_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_nerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}.mp4,\
n_samples_per_ray=${n_samples_per_ray},n_samples_per_ray_fine=${n_samples_per_ray_fine} \
    --reset


# use the head pose predicted by audio2pose model
# python inference/nerfs/lm3d_nerf_infer.py \
#     --config=checkpoints/${Video_ID}/lm3d_nerf_torso/config.yaml \
#     --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
# infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
# infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}.mp4,\
# n_samples_per_ray=${n_samples_per_ray},n_samples_per_ray_fine=${n_samples_per_ray_fine},\
# infer_c2w_name=infer_out/${Video_ID}/pred_c2w/${Wav_ID}.npy \
#     --reset

