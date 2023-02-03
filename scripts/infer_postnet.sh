CUDA_VISIBLE_DEVICES=0 python inference/postnet/postnet_infer.py \
    --config=checkpoints/May/postnet/config.yaml \
    --hparams=infer_audio_source_name=data/raw/val_wavs/zozo.wav,infer_out_npy_name=infer_out/May/pred_lm3d/zozo.npy,infer_ckpt_steps=6000
