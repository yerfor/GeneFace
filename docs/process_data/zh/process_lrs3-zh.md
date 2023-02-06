# 处理LRS3-TED数据集

我们利用LRS3数据集来训练一个鲁棒的语音转动作的映射，这也是GeneFace能够实现高泛化能力的核心所在。除了audio2motion模型外，LRS3数据集还被用来训练postnet和syncnet。

## 步骤1. 申请并下载LRS3-TED数据集

由于License的原因，我们不能在这里提供下载链接。请您通过[这个链接](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)向LRS3-TED数据集的所有者提交申请。

## 步骤2. 处理LRS3数据集

在处理LRS3之前，请确保您按照 `dosc/prepare_env/install_guide_lrs3.md` 的步骤正确安装了处理LRS3数据集的 `process_lrs3`环境。

接着执行以下命令行（你可能需要修改下面 `.py`文件里面的路径名）：

```
conda activate procerss_lrs3
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_video.py # extract 3dmm motion representations
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_mel.py # extract mel spectrogram
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_hubert.py # extract hubert audio representations
```

由于LRS3-TED数据集比较大，您可能需要同时开多个python进程，以利用多个gpu来加速数据预处理。

## 步骤3. 将数据集打包

执行以下命令行（你可能需要修改下面 `.py`文件里面的路径名）

```
conda activate procerss_lrs3
python data_gen/process_lrs3/binarizer.py 
```

如果上述步骤都顺利完成的话，您将能在 `data/binary/lrs3`路径看到处理好的LRS3数据集。
