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
- `2023.3.7` 我们计划于5月1日释出GeneFace的升级版：`GeneFace-S`. 相比起当前版本的GeneFace，升级版能够提供更加准确和稳定的嘴形；训练时间能够缩短4倍；并且能够在RTX2080Ti上实现实时推理（25fps）！欢迎大家保持关注。
- `2023.2.22` 我们发布了一段一分钟的[Demo视频](https://geneface.github.io/GeneFace/how_i_want_to_say_goodbye.mp4)，在其中GeneFace由[DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)生成的一段中文歌曲所驱动，并能够产生准确的嘴形。
- `2023.2.20` 我们发布了一个稳定版本的3D landmark后处理逻辑，位于 `inference/ners/lm3d_nerf_infer.py`，它大大提升了最终合成的视频的稳定性和质量。

## Quick Started!

在[这个release](https://github.com/yerfor/GeneFace/releases/tag/v1.0.0)中，我们提供了预训练的GeneFace模型和处理好的数据集，以便您能快速上手。在本小节的剩余部分我们将介绍如何分4个步骤运行这些模型。如果您想在您自己的目标人物视频上训练GeneFace，请遵循 `docs/prepare_env`、`docs/process_data` 、`docs/train_models` 中的步骤。

步骤1：根据我们在`docs/prepare_env/install_guide_nerf.md`中的步骤，新建一个名为`geneface`的Python环境。通过[这个链接](https://drive.google.com/drive/folders/1YCxXKJFfo1w01PzayhnxWSZZK5k7spSH?usp=share_link)下载`BFM_model_front.mat`，将其复制到`./deep_3drecon/BFM` 和 `./data_util/BFM_models` 目录下。

步骤2：下载`lrs3.zip`和`May.zip`文件，并将其解压在`checkpoints`文件夹中。

步骤3：下载我们处理好的`May.mp4`的数据集文件`trainval_dataset.npy`（大约3.5GB），[链接](https://drive.google.com/drive/folders/1QmXCLjVw4D1wMnrXIuH3ATnCYivSNxoB?usp=share_link)，并将其移动到`data/binary/videos/May/trainval_dataset.npy`路径中。

做完上面的步骤后，您的 `checkpoints`和`data` 文件夹的结构应该是这样的：

```
> checkpoints
    > lrs3
        > lm3d_vae
        > syncnet
    > May
        > postnet
        > lm3d_nerf
        > lm3d_nerf_torso
> data
    > binary
        > videos
            > May
                trainval_dataset.npy
```

Step4. 在终端中执行以下命令：

```
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_nerf.sh
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

# 待办事项

- GeneFace使用3D人脸关键点作为语音转动作模块和运动转图像模块之间的中介变量。但是，由Post-net生成的3D人脸关键点序列有时会出现不好的情况（如时序上的抖动，或超大的嘴巴），进而影响NeRF渲染的视频质量。目前，我们通过对预测的人脸关键点序列进行后处理，部分缓解了这一问题。但是目前的后处理方法还是略显简易，不能完美解决所有bad case。因此我们鼓励大家提出更好的后处理方法。
- 基于NeRF的图像渲染器的推理过程相对较慢(使用RTX2080Ti渲染250帧512x512分辨率的图像需要大约2个小时)。目前，我们可以通过多GPU加速或者将'——n_samples_per_ray '和'——n_samples_per_ray_fine' 设置为较低的值来部分缓解这个问题。在未来，我们将添加加速NeRF推理的技术。
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
* [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) (参考了NeRF相关的代码实现)
* [style_avatar](https://github.com/wuhaozhe/style_avatar) (参考了3DMM相关的代码实现)
