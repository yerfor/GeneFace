export CUDA_VISIBLE_DEVICES=3
export Video_ID=Zhang2
export Wav_ID=how_i_want_to_say_goodbye

python inference/audio2pose/audio2pose_infer.py \
    --config=checkpoints/${Video_ID}/audio2pose/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/${Wav_ID}.wav,\
infer_out_npy_name=infer_out/${Video_ID}/pred_c2w/${Wav_ID}.npy \
    --reset