# Train GeneFace!

GeneFace consists of three models: 1) an generic `audio2motion` model trained on LRS3 dataset; 2) a person-specific `postnet` trained on LRS3 and the target person video; 3) a person-specific `nerf` renderer trained on the target person video.

To train GeneFace, please first follow the docs in `docs/prepare_env` and `docs/process_data`.

We also provide pre-trained models at [this link](https://drive.google.com/drive/folders/1RyN6BpqTACH0RRf6tY6_yW_qA3MM0Qun?usp=share_link).

## Step1. Train the SyncNet Model

NOTE: We provide the pre-trained SyncNet model at [this link](https://drive.google.com/drive/folders/1UFOnON4PzaIinTj5d0m_KJsOjOXl2-zt?usp=share_link), you can download it and place it into the directory `checkpoints/lrs3/syncnet`

If you want to train a SyncNet from scratch, please run the following commandlines:

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/lrs3/lm3d_syncnet.yaml --exp_name=checkpoints/lrs3/syncnet
```

## Step2. Train the Audio2motion model

NOTE: We provide the pre-trained Audio2motion model at [this link](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link), you can download it and place it into the directory `checkpoints/lrs3/lm3d_vae`

If you want to train a audio2motion model from scratch, please run the following commandlines:

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/lrs3/lm3d_vae_sync.yaml --exp_name=checkpoints/lrs3/lm3d_vae
```

## Step3. Train the Postnet

NOTE: We provide the pre-trained Post-net model for ` data/raw/videos/May.mp4` at [this link](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link), you can download it and place it into the directory  `checkpoints/May/postnet`

Note that Audio2motion and Syncnet are generic models and only need to be trained once. By contrast, Post-net and NeRF-based renderer are person-specific models, so for each target person video, you need train a new post-net and NeRF-based renderer.

If you want to train a postnet model from scratch, please run the following commandlines:

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_postnet_sync.yaml --exp_name=checkpoints/May/postnet
```

## Step4. Train the NeRF-based Render

NOTE: We provide the pre-trained NeRF model for ` data/raw/videos/May.mp4` at [this link](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link), you can download it and place it into the directory  `checkpoints/May/lm3d_nerf` and `checkpoints/May/lm3d_nerf_torso`

If you want to train a NeRF model from scratch, please run the following commandlines:

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf.yaml --exp_name=checkpoints/May/lm3d_nerf
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf_torso.yaml --exp_name=checkpoints/May/lm3d_nerf_torso
```

## Step5. Inference!

You can infer the GeneFace with the following commandlines:

```
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_nerf.sh
```
