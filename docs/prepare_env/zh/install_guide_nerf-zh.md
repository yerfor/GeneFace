本指南介绍了如何构建一个python环境，这是处理NeRF的目标说话人视频数据集和训练GeneFace所必需的。

以下安装流程在RTX2080，Ubuntu 16.04得到验证。在更新的环境（如RTX3090Ti+）经过细微的修改（如使用更新的CUDA版本）后可能也适用。

# 1. 安装Python库

```
conda create -n geneface python=3.8
conda activate geneface
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
# conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

pip install -r requirements_nerf.txt

```

# 2. 准备3DMM模型

到这个网站去申请：[链接](https://faces.dmi.unibas.ch/bfm/)

你能得到一个 `BaselFaceModel.tgz`，将其解压，解压后获得其中 `01_MorphableModel.mat`保存到 `./data_util/face_tracking/3DMM`文件夹

接着执行：

```
cd data_util/face_tracking
conda activate 
python convert_BFM.py
```

# 3. 下载DeepSpeech（AD-NeRF需要）

```bash
cd data_util/deepspeech_features
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/deepspeech-0.9.2-models.pbmm
```
