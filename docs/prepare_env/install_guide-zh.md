# 搭建环境
本指南介绍了如何构建一个用于GeneFace的python环境。以下安装流程在RTX3090，Ubuntu 18.04得到验证。

# 1. 安装Python库

```
conda create -n geneface python=3.9 -y
conda activate geneface
# install pytorch with conda, older versions also work
conda install pytorch=1.12 torchvision cudatoolkit=11.3.1 -c pytorch -c nvidia -y
# install pytorch-3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y # 0.7.2 recommended
# other dependencies, including tensorflow-gpu=2.x
sudo apt-get install libasound2-dev portaudio19-dev # dependency for pyaudio
pip install -r docs/prepare_env/requirements.txt 

# install torch-ngp cuda extensions
# NOTE: you need to manually install CUDA with the same version of pytorch (in this case, v11.3.1)
# make sure your cuda path (typically /usr/local/cuda) points to a installed `/usr/local/cuda-11.3`
# then run:
bash docs/prepare_env/install_ext.sh 
```

如果你在上述安装过程中遇到兼容性问题，可以参考`docs/prepare_env/geneface_*.yaml`文件，其中记录了我在不同型号GPU下安装成功的详细环境配置。

# 2. 准备 3DMM 模型

## 2.1 下载 3DMM model

在[这个链接](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)申请BFM2009 model.

你能得到一个 `BaselFaceModel.tgz`，将其解压，解压后获得其中 `01_MorphableModel.mat`保存到`./deep_3drecon/BFM/`文件夹

## 2.2 Download PCA Basis

通过这个链接下载：[链接](https://drive.google.com/drive/folders/1iTopSpZucEmjWiWZIErLYiMBlZYwzil2?usp=share_link)

获得其中的 `Exp_Pca.bin`存到 `./deep_3drecon/BFM` 路径

## 2.3 Download BFM Model Front

通过这个链接下载：[链接](https://drive.google.com/drive/folders/1YCxXKJFfo1w01PzayhnxWSZZK5k7spSH?usp=share_link)

获得其中的 ` BFM_model_front.mat` 存到 `./deep_3drecon/BFM` 路径

## 2.4 Download FaceRecon Model

通过这个链接下载：[链接](https://drive.google.com/drive/folders/18VRcygXYOKPYvJWsl9lrF0J9PoFPk77y?usp=sharing)

获得其中的 `epoch_20.pth` 存到 `./deep_3drecon/checkpoints/facerecon` 路径

## 2.5 生成face_tracking需要的文件
在GeneFace的root路径执行以下命令行：

```
cd data_util/face_tracking
conda activate geneface
python convert_BFM.py
```
这将在以下路径生成文件：`data_util/face_tracking/3DMM/3DMM_info.npy`.


# 3. 验证安装成功

```
# 跑通 deep_3drecon_pytorch 项目的原始example
cd <root_dir>
conda activate geneface
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python deep_3drecon/test.py 

# 验证与GeneFace之间的桥梁
# 生成deep_3drecon的config文件（默认已生成）
python deep_3drecon/generate_reconstructor_opt_for_geneface.py 
CUDA_VISIBLE_DEVICES=0 python
# 以下几行在python中执行
> import deep_3drecon
> face_reconstructor = deep_3drecon.Reconstructor()
```


