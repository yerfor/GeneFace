# GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis | ICLR'23

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2301.13430)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/GeneFace)](https://github.com/yerfor/GeneFace) | [![downloads](https://img.shields.io/github/downloads/yerfor/GeneFace/total.svg)](https://github.com/yerfor/GeneFace/releases) | ![visitors](https://visitor-badge.glitch.me/badge?page_id=yerfor/GeneFace)

这个仓库是我们[ICLR-2023论文](https://arxiv.org/abs/2301.13430)的官方PyTorch实现，我们在其中提出了**GeneFace** 算法，用于高泛化高保真的音频驱动的虚拟人视频合成。

<p align="center">
    <br>
    <img src="assets/GeneFace.png" width="1000"/>
    <br>
</p>

我们的GeneFace对域外音频（如不同说话人、不同语种的音频）实现了更好的嘴唇同步和表现力。推荐您观看[此视频](https://geneface.github.io/GeneFace/example_show_improvement.mp4)，以了解GeneFace与之前基于NeRF的虚拟人合成方法的口型同步能力对比。您也可以访问我们的[项目页面](https://geneface.github.io/)以了解更多详细信息。


## 🔥新闻:
- `2023.3.16` 我们在[这个release](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0)做出了重大的更新，我们提供了一个[demo](assets/zozo_radnerf_torso_smo.mp4)。本次更新包括：1) 基于RAD-NeRF的渲染器，它可以做到实时渲染，并且训练时间缩短到10小时。2) 基于pytorch的`deep3d_recon`模块,相比起之前使用的Tensorflow版本，它更容易安装，并且推理速度快8倍。 3) 音高感知的`audio2motion`模块，相比原先的版本可以生成更加准确的唇形。4) 解决了一些导致过多内存占用的bug。5)我们会在四月上传最新的论文。

- `2023.2.22` 我们发布了一段一分钟的[Demo视频](https://geneface.github.io/GeneFace/how_i_want_to_say_goodbye.mp4)，在其中GeneFace由[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)生成的一段中文歌曲所驱动，并能够产生准确的嘴形。
- `2023.2.20` 我们发布了一个稳定版本的3D landmark后处理逻辑，位于 `inference/nerfs/lm3d_nerf_infer.py`，它大大提升了最终合成的视频的稳定性和质量。

## Quick Started!

在[这个release](https://github.com/yerfor/GeneFace/releases/tag/v1.1.0)中，我们提供了预训练的GeneFace模型和处理好的数据集，以便您能快速上手。在本小节的剩余部分我们将介绍如何分4个步骤运行这些模型。如果您想在您自己的目标人物视频上训练GeneFace，请遵循 `docs/prepare_env`、`docs/process_data` 、`docs/train_models` 中的步骤。

步骤1：根据我们在`docs/prepare_env/install_guide.md`中的步骤，新建一个名为`geneface`的Python环境。

步骤2：下载`lrs3.zip`和`May.zip`文件，并将其解压在`checkpoints`文件夹中。

步骤3：根据`docs/process_data/zh/process_target_person_video-zh.md`的指引，处理`May.mp4`文件，得到数据集文件`data/binary/videos/May/trainval_dataset.npy`。

做完上面的步骤后，您的 `checkpoints`和`data` 文件夹的结构应该是这样的：

```
> checkpoints
    > lrs3
        > lm3d_vae_sync
        > syncnet
    > May
        > lm3d_postnet_sync
        > lm3d_radnerf
        > lm3d_radnerf_torso
> data
    > binary
        > videos
            > May
                trainval_dataset.npy
```

Step4. 在终端中执行以下命令：

```
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_radnerf.sh
# bash scripts/infer_radnerf_gui.sh # 你也可以利用RADNeRF提供的GUI进行交互体验
```

你能在以下路径找到输出的视频 `infer_out/May/pred_video/zozo.mp4`.

## 搭建环境

请参照`docs/prepare_env`文件夹中的步骤 .

## 准备数据
请参照`docs/process_data`文件夹中的步骤.

## 训练模型

请参照`docs/train_models`文件夹中的步骤.

# 在其他目标人物视频上训练GeneFace

除了本仓库中提供的 `May.mp4`，我们还提供了8个实验中使用的目标人物视频。你可以从[这个链接](https://drive.google.com/drive/folders/1FwQoBd1ZrBJMrJE3ZzlNhK8xAe1OYGjX?usp=share_link)下载。

要训练一个名为 <`video_id>.mp4`的新视频，你应该把它放在 `data/raw/videos/`目录下，然后在 `egs/datasets/videos/<video_id>`目录下创建一个新文件夹，并根据提供的示例文件夹 `egs/datasets/videos/May`添加对应的yaml配置文件。

除了使用我们提供的视频进行训练外，您还可以自己录制视频，为自己训练一个独一无二的GeneFace虚拟人模型！

## 引用我们的论文

```
@article{ye2023geneface,
  title={GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis},
  author={Ye, Zhenhui and Jiang, Ziyue and Ren, Yi and Liu, Jinglin and He, Jinzheng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.13430},
  year={2023}
}
```

## 致谢

本工作受到以下仓库的影响：

* [NATSpeech](https://github.com/NATSpeech/NATSpeech) (参考了其中的代码框架)
* [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) (参考了NeRF相关的数据提取和原始NeRF的代码实现)
* [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF) (参考了RAD-NeRF的代码实现)
* [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) (参考了基于Pytorch的3DMM参数提取)
