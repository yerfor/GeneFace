export CUDA_VISIBLE_DEVICES=0
export Video_ID=Zhang2
export Wav_ID=how_i_want_to_say_goodbye
export Postnet_Ckpt_Steps=12000 # please reach to `docs/train_models.md` to get some tips about how to select an approprate ckpt_steps!

python inference/postnet/postnet_infer.py \
    --config=checkpoints/${Video_ID}/postnet/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/${Video_ID}/pred_lm3d/${Wav_ID}.npy,\
infer_ckpt_steps=${Postnet_Ckpt_Steps} \
    --reset