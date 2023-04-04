# Train GeneFace!

[中文文档](./train_models-zh.md)

GeneFace consists of three models: 1) an generic `audio2motion` model trained on LRS3 dataset; 2) a person-specific `postnet` trained on LRS3 and the target person video; 3) a person-specific `nerf` renderer trained on the target person video.

To train GeneFace, please first follow the docs in `docs/prepare_env` and `docs/process_data` to build the environment and prepare the datasets, respectively.

We also provide pre-trained models at [this link](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0), in which:

* `lrs3.zip` includes the models trained on LRS3-ted dataset (a `lm3d_vae_sync` to perform the audio2motion transform and a `syncnet` for measuring the lip-sync), which are generic for all possible target person videos.
* `May.zip` includes the models trained on the `May.mp4` target person video (a `postnet` for refining the predicted 3d landmark, a `lm3d_radnerf` for rendering the head image, and a `lm3d_radnerf_torso` for rendering the torso part). For each target person video, you need to train these three models.

## Step1. Train the SyncNet Model

NOTE: We provide the pre-trained SyncNet model in `lrs3.zip` at [this link](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0), you can download it and extract the `syncnet` folder and place it into the path `checkpoints/lrs3/syncnet`.

If you want to train a SyncNet from scratch, please run the following commandlines (The processed LRS3 dataset is required):

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/lrs3/lm3d_syncnet.yaml --exp_name=lrs3/syncnet
```

Note that SyncNet is a generic model for all possible target person videos, so you only to train it once!

## Step2. Train the Audio2motion model

NOTE: We provide the pre-trained Audio2motion model in `lrs3.zip` at [this link](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0), you can download it and extract the `lm3d_vae_sync` folder and place it into the path `checkpoints/lrs3/lm3d_vae_sync`.

If you want to train a audio2motion model from scratch, please run the following commandlines (The processed LRS3 dataset is required):

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/lrs3/lm3d_vae_sync.yaml --exp_name=lrs3/lm3d_vae_sync
```

Note that the Audio2motion model named `lm3d_vae_sync` is a generic model for all possible target person videos, so you only to train it once!

## Step3. Train the Postnet

NOTE: We provide the pre-trained Post-net model for the target person video named `data/raw/videos/May.mp4` in `May.zip` at [this link](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0), you can download it and extract the `postnet` folder and place it into the path `checkpoints/May/postnet`.

If you want to train a postnet model from scratch, please run the following commandlines (The processed LRS3 dataset and the target person video is required):

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_postnet_sync.yaml --exp_name=May/postnet
```

Note that the Post-net is person-specific, so for each target person video, you need to train a new Post-net.

#### tips: choosing the appropriate checkpoint

Since our postnet belongs to **Adversarial** Domain Adaptation, whose training process is widely considered to be unstable. For example, training the model for too many steps may lead to model collapse. For example, when mode collapse occurs, the postnet may map abitrary input landmark into the same landmark in the target person domain (which results in rises in validation sync/mse loss). Therefore, to avoid degradation of the lip-sync performance, we should make an early stop, i.e., select a checkpoint trained with a small number of iterations. However, at the same time, if the number of iterations is too small, postnet may be underfitting and cannot successfully map the landmarks into the target person domain (which means the adversarial loss is not converged).

Therefore, in practice, we choose the checkpoint with the appropriate number of iterations according to three principles: (1) validation sync/mse loss should be as low as possible; (2) the adversarial loss should be converged. (3) a small number of iterations is desirable.

The following figure shows an example of the process of selecting the appropriate postnet checkpoint when training `May.mp4`. We found that `val/mse` and `val/sync` are relatively low at 6k steps. Besides, `tr/disc_neg_conf` and `tr/disc_pos_conf` are both about 0.5 (which means that the discriminator cannot distinguish between the (GT) positive samples and the (postnet-generated) negative samples), so we choose the checkpoint at 6k steps.

<p align="center">
    <br>
    <img src="../../assets/tips_to_select_postnet_ckpt.png" width="1000"/>
    <br>
</p>

Finally, to quickly verify the lip-sync performance of the selected postnet checkpoint, we also provide a script to visualize the predicted 3D landmark. Run the following script (you may need to modify the path names in the following `.sh` and `.py` files):

```
conda activate geneface
bash infer_postnet.sh # use the selected postnet checkpoint to predict the 3D landmark sequence.
python utils/visualization/lm_visualizer.py # visualize the 3D landmark sequence.
```

You can see the visualized 3d landmark video in the output path.

## Step4. Train the RAD-NeRF-based Render

RAD-NeRF uses instant-ngp to improve the training and inference speed of NeRF. We recommend using RAD-NeRF as the backend of NeRF-based renderer. RAD-NeRF is 6x faster than NeRF in training, and could infer in real-time, with similar rendering quality to the vanilla NeRF.

NOTE: We provide the pre-trained RAD-NeRF model for the target person video named `data/raw/videos/May.mp4` in `May.zip` at [this link](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0), you can download it and extract the `lm3d_radnerf` and `lm3d_radnerf_torso` folder, then place it into the path `checkpoints/May/lm3d_radnerf` and `checkpoints/May/lm3d_radnerf_torso`, respectively.

If you want to train a RAD-NeRF model from scratch, please run the following commandlines (The processed target person video dataset is required). It takes about 10 hours on a RTX3090Ti.

```
conda activate geneface
export PYTHONPATH=./
# Train the head rad_nerf, it takes about 6 hours in one RTX3090Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_radnerf.yaml --exp_name=May/lm3d_radnerf
# Train the torso rad_nerf, it takes about 4 hours in one RTX3090Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_radnerf_torso.yaml --exp_name=May/lm3d_radnerf_torso
```

Note that the NeRF-based renderer is person-specific, so for each target person video, you need to train a new NeRF-based renderer.

### Legacy: Train the vanilla NeRF-based renderer in the GeneFace paper

Although the use of RAD-NeRF is recommended, we still support the vanilla NeRF-based renderer for completeness. You can train it with the following command line. It takes about 60 hours on a RTX3090Ti.

Note: As pointed out in [this issue](https://github.com/yerfor/GeneFace/issues/18), since NeRF requires good initilization, you may need to run the commandline several times, until the loss converges normally.

```
conda activate geneface
export PYTHONPATH=./
# Train the head nerf, it takes about 30 hours on a RTX3090Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf.yaml --exp_name=May/lm3d_nerf
# Train the torso nerf, it takes about 36 hours on a RTX3090Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf_torso.yaml --exp_name=May/lm3d_nerf_torso
```

The inference process for the vanilla Nerf-based renderer is very slow (it takes about 2 hours to render 250 frames of 512x512 resolution images using RTX2080Ti). This problem can be partially mitigated by setting `--n_samples_per_ray` and `--n_samples_per_ray_fine` to lower values. However, with the implementation of RAD-NeRF, inference speed is no longer an issue for GeneFace.

## Step5. Inference!

You can infer the GeneFace with the following commandlines:

```
# By default we use the data/raw/val_wavs/zozo.wav as the driving audio.
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_radnerf.sh
```
