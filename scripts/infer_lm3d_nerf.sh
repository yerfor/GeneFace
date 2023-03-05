export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1 # now we support multi-gpu inference!
export Video_ID=May
export Wav_ID=zozo # the .wav file should locate at `data/raw/val_wavs/<wav_id>.wav`
export n_samples_per_ray=32 # during training 64
# export n_samples_per_ray=16 # you can use a smaller value (e.g., 16) to accelerate the inference
export n_samples_per_ray_fine=128 # during training 128
# export n_samples_per_ray_fine=32 # you can use a smaller value (e.g., 32) to accelerate the inference
export infer_scale_factor=1.0 # scale of output resolution, defautlt 1.0 -> 512x512 image

# postprocessing params
export infer_lm3d_clamp_std=3.0 # typically 1.~5., reduce it when blurry or bad cases occurs
export infer_lm3d_lle_percent=0. # 0.~1., enlarge it when blurry or bad cases occurs
export infer_lm3d_smooth_sigma=0. # typically 0.~3., enlarge it when blurry or bad cases occurs

# use head pose from the dataset
python inference/nerfs/lm3d_nerf_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_nerf_torso/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}_infer_torso_nerf.mp4,\
n_samples_per_ray=${n_samples_per_ray},n_samples_per_ray_fine=${n_samples_per_ray_fine},\
infer_scale_factor=${infer_scale_factor},\
infer_lm3d_clamp_std=${infer_lm3d_clamp_std},\
infer_lm3d_lle_percent=${infer_lm3d_lle_percent},\
infer_lm3d_smooth_sigma=${infer_lm3d_smooth_sigma} \
    --infer

# use the head pose predicted by audio2pose model
# python inference/nerfs/lm3d_nerf_infer.py \
#     --config=checkpoints/${Video_ID}/lm3d_nerf_torso/config.yaml \
#     --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
# infer_cond_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
# infer_out_video_name=infer_out/${Video_ID}/pred_video/${Wav_ID}_pred_pose.mp4,\
# n_samples_per_ray=${n_samples_per_ray},n_samples_per_ray_fine=${n_samples_per_ray_fine},\
# infer_scale_factor=${infer_scale_factor},\
# infer_c2w_name=infer_out/${Video_ID}/pred_c2w/${Wav_ID}.npy \
#     --infer
