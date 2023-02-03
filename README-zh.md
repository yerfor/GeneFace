# GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis | ICLR'23

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2301.13430)| [![GitHub Stars](https://img.shields.io/github/stars/yerfor/GeneFace)](https://github.com/yerfor/SyntaSpeech) | ![visitors](https://visitor-badge.glitch.me/badge?page_id=yerfor/GeneFace)

这个仓库是我们[ICLR-2023论文](https://arxiv.org/abs/2301.13430)的官方PyTorch实现，我们在其中提出了**GeneFace** 算法，用于通用和高保真的音频驱动的虚拟人视频合成。

<p align="center">
    <br>
    <img src="assets/GeneFace.png" width="1000"/>
    <br>
</p>

我们的GeneFace对域外音频（如不同说话人、不同语种的音频）实现了更好的嘴唇同步和表现力。推荐您观看[此视频](https://geneface.github.io/GeneFace/example_show_improvement.mp4)，以了解GeneFace与之前基于NeRF的虚拟人合成方法的口型同步能力对比。您也可以访问我们的[项目页面](https://geneface.github.io/)以了解更多详细信息。

## Quick Started!

我们提供[预训练的GeneFace模型](https://drive.google.com/drive/folders/1L87ZuvC3BOPdWZ7fALdUKYcIt4pWXtDz?usp=share_link)，以便您能快速上手。如果您想在您自己的目标人物视频上训练GeneFace，请遵循 `docs/prepare_env`、`docs/process_data` 、`docs/train_models` 中的步骤。

步骤1：我们在[这个链接](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link)上提供了预先训练好的Audio2motion模型(上图中的Variational Motion Generator)，您可以下载它并将其放在 `checkpoints/lrs3/lm3d_vae` 。

步骤2：我们在[这个链接](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link)上提供了预先训练好的Post-net (上图中的Domain Adaptative Post-net )，这个模型在 ` data/raw/videos/May.mp4` 上预训练。 您可以下载它并将其放在  `checkpoints/May/postnet` 。

Step3. 我们在[这个链接](https://drive.google.com/drive/folders/1qsYYWmyiDnf0v5AAF9EplAaoO6DLxjFd?usp=share_link)上提供了预先训练好的NeRF (上图中的3DMM NeRF Renderer) ，这个模型在 ` data/raw/videos/May.mp4` 上预训练。您可以下载它并将其放在 `checkpoints/May/lm3d_nerf` and `checkpoints/May/lm3d_nerf_torso` 。

做完上面的步骤后，您的 `checkpoints`文件夹的结构应该是这样的：

```
> checkpoints
    > lrs3
        > lm3d_vae
        > syncnet
    > May
        > postnet
        > lm3d_nerf
        > lm3d_nerf_torso
  
```

Step4. 在终端中执行以下命令：

```
bash scripts/infer_postnet.sh
bash scripts/infer_lm3d_nerf.sh
```

你能在以下路径找到输出的视频 `infer_out/May/pred_video/zozo.mp4`.

## 搭建环境

请参照该文件夹中的步骤 `docs/prepare_env`.

## 准备数据

请参照该文件夹中的步骤 `docs/process_data`.

## 训练模型

请参照该文件夹中的步骤 `docs/train_models`.

# 在其他目标人物视频上训练GeneFace

除了本仓库中提供的 `May.mp4`，我们还提供了8个实验中使用的目标人物视频。你可以从[这个链接](https://drive.google.com/drive/folders/1FwQoBd1ZrBJMrJE3ZzlNhK8xAe1OYGjX?usp=share_link)下载。

要训练一个名为 <`video_id>.mp4`的新视频，你应该把它放在 `data/raw/videos/`目录下，然后在 `egs/datasets/videos/<video_id>`目录下创建一个新文件夹，并根据提供的示例文件夹 `egs/datasets/videos/May`添加对应的yaml配置文件。

除了使用我们提供的视频进行训练完，您还可以自己录制视频，为自己训练一个独一无二的GeneFace虚拟人模型！

# 待办事项

GeneFace使用3D人脸关键点作为语音转动作模块和运动转图像模块之间的中介。但是，由Post-net生成的3D人脸关键点序列有时会出现不好的情况（如时序上的抖动，或超大的嘴巴），进而影响NeRF渲染的视频质量。目前，我们通过对预测的人脸关键点序列进行后处理，部分缓解了这一问题。但是目前的后处理方法还是略显简易，不能完美解决所有bad case。因此我们鼓励大家提出更好的后处理方法。

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
