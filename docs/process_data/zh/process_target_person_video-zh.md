# 处理说话人视频数据集

你需要一个大约3分钟的目标人物视频来训练特定人物的postnet和基于NeRF的渲染器。（视频长度越长越好）

我们在  `data/raw/videos/May.mp4` 路径下提供了一个示例视频

## 步骤1. 提取NeRF所需的特征

运行如下命令行：

```
conda activate geneface
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 data_gen/nerf/process_data.sh $VIDEO_ID
```

## 步骤2. 从视频中提取3DMM特征

这个特征是训练postnet和NeRF所必须的。
运行如下命令行：

```
conda activate process_lrs3
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 python data_gen/nerf/extract_3dmm.py --video_id=$VIDEO_ID
```

## 步骤3. 将处理好的训练数据打包

运行如下命令行：

```
conda activate geneface
export PYTHONPATH=./
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/May/lm3d_nerf.yaml

```

如果上面的步骤都顺利完成，你可以在 `data/binary/videos/May`路径下看到处理好的目标说话人视频的数据集。
