# 处理LRS3-TED数据集

我们利用LRS3数据集来训练一个鲁棒的语音转动作的映射，这也是GeneFace能够实现高泛化能力的核心所在。除了audio2motion模型外，LRS3数据集还被用来训练postnet和syncnet。

## 处理完毕的LRS3数据集文件可用

🔥注意：由于我们转向新的3DMM提取器，我们更新了提供的lrs3数据集。您可能需要下载最新的数据集文件以与最新的代码兼容。

由于LRS3数据集数据量较大（500小时+），其处理过程非常消耗计算资源，因此我们在百度云盘提供了处理好的LRS3数据集（总共约26GB）。如果你使用我们处理好的数据集，可以跳过下面的步骤1-3，直接进入步骤4，验证安装成功。

- 谷歌网盘下载链接：[分区1](https://drive.google.com/drive/folders/1QK_ikLKUzGYiqHBzvKz0s5zKWeH-sm3L?usp=sharing)，[分区2](https://drive.google.com/drive/folders/1WbECLfpxAZ0D7PcrlZxV-fCObT-TnfD8?usp=share_link)。
- 百度云盘下载链接：[链接](https://pan.baidu.com/s/1JsvEz58c9ItSI73ls43tTw?pwd=lrs3)，提取码:`lrs3`
- 如何使用:

  - 步骤1：将拆分的子文件还原成压缩包 `cat lrs3.zip.part_* > lrs3.zip` 。
  - 步骤2：将压缩包解压，并将其移动到 `data/binary/lrs3` 目录下。
- 免责声明：我们提供的文件仅包含了经过数据脱敏处理的特征（比如HuBET作为音频的表征），没有侵犯LRS3中视频的版权。

## 步骤1. 申请并下载LRS3-TED数据集

由于License的原因，我们不能在这里提供下载链接。请您通过[这个链接](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)向LRS3-TED数据集的所有者提交申请。

## 步骤2. 处理LRS3数据集

在处理LRS3之前，请确保您按照 `dosc/prepare_env/install_guide.md` 的步骤正确安装了处理LRS3数据集的 `geneface`环境。

接着执行以下命令行（你可能需要修改下面 `.py`文件里面的路径名）：

```
conda activate 
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_video.py # extract 3dmm motion representations
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_mel_f0.py # extract mel spectrogram
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_hubert.py # extract hubert audio representations
```

由于LRS3-TED数据集比较大，您可能需要同时开多个python进程，以利用多个gpu来加速数据预处理。

## 步骤3. 将数据集打包

执行以下命令行（你可能需要修改下面 `.py`文件里面的路径名）

```
conda activate procerss_lrs3
python data_gen/process_lrs3/binarizer.py 
```

## 步骤4. 验证

如果上述步骤都顺利完成的话，您将能在 `data/binary/lrs3`路径看到处理好的LRS3数据集。理想状态下，你的 `data`文件夹内部结构应该是类似这样的：

```
> data
    > binary
        > lrs3
            sizes_train.npy
            sizes_val.npy
            spk_id2ispk_idx.npy
            stats.npy
            train.data
            train.idx
            val.data
            val.idx
```
