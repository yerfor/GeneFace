# 训练 GeneFace!
GeneFace 包含三个模块：1）一个训练于LRS3数据集并通用于所有说话人的`语音转动作`模块；2）一个适用于特定说话人的`动作后处理`网络，它被训练于LRS3数据集和对应说话人的视频数据；3）一个适用于特定说话人的`基于NeRF的渲染器`，它被训练于对应说话人的视频数据。

要训练GeneFace，请首先按照我们在`docs/prepare_env`文档和`docs/process_data`文档中的步骤，分别完成搭建环境和准备数据集。

在[这个链接](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)中，我们还准备了GeneFace的预训练模型，其中：

* `lrs3.zip` 包含了在LRS3数据集上训练的模型 (包括一个`lm3d_vae`模型以实现语音转动作的变换，和一个`syncnet`以实现对语音-嘴形对齐程度的衡量)，这些模型是通用于所有说话人视频的。
* `May.zip` 包含了我们在`May.mp4`视频上训练的所有模型（包括一个`postnet`以对`lm3d_vae`产生的3D landmark进行后处理，以及一个`lm3d_nerf`和`lm3d_nerf_torso`分别渲染说话人的头部和躯干部位。）对每个说话人视频，你都需要新训练这三个模型。

## 步骤1. 训练SyncNet模型
注意：我们在[这个链接](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)的`lrs3.zip`文件中提供了预训练好的SyncNet,你可以将其下载并提取出其中的`syncnet`文件夹，并将它放到`checkpoints/lrs3/syncnet`路径中。

如果你想要从头训练SyncNet，请执行以下命令行（你需要首先准备好LRS3数据集）：

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/lrs3/lm3d_syncnet.yaml --exp_name=checkpoints/lrs3/syncnet
```

注意SyncNet模型适用于所有说话人视频，所以你只需要训练它一次！

## 步骤2. 训练Audio2Motion模型

注意：我们在[这个链接](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)的`lrs3.zip`文件中提供了预训练好的audio2motion模型,你可以将其下载并提取出其中的`lm3d_vae`文件夹，并将它放到`checkpoints/lrs3/lm3d_vae`路径中。

如果你想要从头训练audio2motion模型，请执行以下命令行（你需要首先准备好LRS3数据集）：

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/lrs3/lm3d_vae_sync.yaml --exp_name=checkpoints/lrs3/lm3d_vae
```

注意名为`lm3d_vae`的audio2motion模型适用于所有说话人视频，所以你只需要训练它一次！

## 步骤3. 训练PostNet模型


注意：我们在[这个链接](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)的`May.zip`文件中提供了专用于`data/raw/videos/May.mp4`视频的预训练好的Postnet模型,你可以将其下载并提取出其中的`postnet`文件夹，并将它放到`checkpoints/May/postnet`路径中。

如果你想要从头训练postnet模型，请执行以下命令行（你需要首先准备好LRS3数据集和对应的说话人视频数据集）：

```
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_postnet_sync.yaml --exp_name=checkpoints/May/postnet
```

注意postnet模型仅适用于对应的说话人视频，所以对每个新的说话人视频你都需要训练一个新的postnet。

## 步骤4. 训练基于NeRF的渲染器

注意：我们在[这个链接](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)的`May.zip`文件中提供了专用于`data/raw/videos/May.mp4`视频的预训练好的NeRF模型,你可以将其下载并提取出其中的`lm3d_nerf`和`lm3d_nerf_torso`文件夹，并将它放到`checkpoints/May/lm3d_nerf`和`checkpoints/May/lm3d_nerf_torso`路径中。

如果你想要从头训练NeRF模型，请执行以下命令行（你需要首先准备好LRS3数据集和对应的说话人视频数据集）：

```
conda activate geneface
export PYTHONPATH=./
# Train the head nerf, it takes about 1 day in one RTX2080Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf.yaml --exp_name=checkpoints/May/lm3d_nerf
# Train the torso nerf, it takes about 1.5 days in one RTX2080Ti
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/videos/May/lm3d_nerf_torso.yaml --exp_name=checkpoints/May/lm3d_nerf_torso
```

注意NeRF模型仅适用于对应的说话人视频，所以对每个新的说话人视频你都需要训练一个新的NeRF模型。

## 步骤5. 使用GeneFace生成说话人视频

你可以执行以下命令行，以运行训练好的GeneFace生成说话人视频。

```
# By default we use the data/raw/val_wavs/zozo.wav as the driving audio.
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_nerf.sh
```

注意：基于NeRF的图像渲染器的推理过程相对较慢(使用RTX2080Ti渲染250帧512x512分辨率的图像需要大约2个小时)。目前，我们可以通过将'——n_samples_per_ray '和'——n_samples_per_ray_fine '设置为较低的值来部分缓解这个问题。在未来，我们将添加加速NeRF推理的技术。
