export CUDA_VISIBLE_DEVICES=3
export Video_ID=May
export Wav_ID=zozo
# export Audio2motion_Steps=2000 # please reach to `docs/train_models.md` to get some tips about how to select an approprate ckpt_steps!
for Audio2motion_Steps in 2000 4000 8000 16000 20000 24000 28000 32000 36000 40000
do
    python inference/audio2motion/audio2motion_infer.py \
        --config=checkpoints/lrs3/lm3d_vae/config.yaml \
        --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/audio2motion/pred_lm3d/step${Audio2motion_Steps}_${Wav_ID}.npy,\
infer_ckpt_steps=${Audio2motion_Steps} \
        --reset
    
    python utils/visualization/lm_visualizer.py --npy_name=infer_out/audio2motion/pred_lm3d/step${Audio2motion_Steps}_${Wav_ID}.npy \
--audio_name=data/raw/val_wavs/${Wav_ID}.wav --out_path=infer_out/audio2motion/visualizd_lm3d/step${Audio2motion_Steps}_${Wav_ID}.mp4
done