export CUDA_VISIBLE_DEVICES=0
export Video_ID=May
export Wav_ID=zozo
export Postnet_Ckpt_Steps=4000 # please reach to `docs/train_models.md` to get some tips about how to select an approprate ckpt_steps!

python inference/postnet/postnet_infer.py \
    --config=checkpoints/${Video_ID}/lm3d_postnet_sync/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_ckpt_steps=${Postnet_Ckpt_Steps} \
    --reset