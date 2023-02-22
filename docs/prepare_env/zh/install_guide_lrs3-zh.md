本指南介绍了如何构建一个python env，来处理LRS3-TED数据集。

以下安装流程在RTX2080，Ubuntu 16.04得到验证。在更新的环境（如RTX3090Ti+）经过细微的修改（如使用更新的CUDA版本）后可能也适用。

# 1. 下载3D人脸模型

## 1.1  BFM09 Model

到这个网站去申请：[链接](https://faces.dmi.unibas.ch/bfm/)

你能得到一个 `BaselFaceModel.tgz`，将其解压，解压后获得其中 `01_MorphableModel.mat`保存到 `./deep_3drecon/BFM`文件夹

## 1.2 PCA Basis

通过这个链接下载：[链接](https://drive.google.com/drive/folders/1iTopSpZucEmjWiWZIErLYiMBlZYwzil2?usp=share_link)

获得其中的 `Exp_Pca.bin`存到 `./deep_3drecon/BFM` 路径

## 1.3 BFM Model Front

通过这个链接下载：[链接](https://drive.google.com/drive/folders/1YCxXKJFfo1w01PzayhnxWSZZK5k7spSH?usp=share_link)

获得其中的 ` BFM_model_front.mat` 存到 `./deep_3drecon/BFM` 路径

获得其中的 ` BFM_model_front.mat` 存到 `./data_util/BFM_models` 路径

# 2. 安装python库


```
conda create -n process_lrs3 python=3.7.11
conda activate process_lrs3
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.1 # RTX2080 
# conda install -c pytorch pytorch=1.11.0 torchvision cudatoolkit=11.3 # RTX3090 
conda install tensorflow-gpu=1.14.0 # RTX2080 supports TF114 gpu version
# conda install tensorflow=1.14.0 # RTX3090 only supports TF114 cpu version
pip install -r docs/prepare_env/requirements_lrs3.txt
```

# 3. 本地编译mesh renderer

## 3.1 首先安装bazel

```
sudo apt install curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
sudo apt update && sudo apt full-upgrade
sudo apt install bazel-1.0.0
sudo ln -s /usr/bin/bazel-1.0.0 /usr/bin/bazel
bazel --version  # 1.0.0

```

## 3.2 接着编译mesh renderer

```
cd deep_3drecon/mesh_renderer/
touch WORKSPACE
bazel build //kernels:rasterize_triangles_kernel 
mv ./bazel-bin/kernels/rasterize_triangles_kernel.so ./kernels/
```

### 3.2.1 编译常见的BUG

* 找不到tensorflow_framework，解决方法是到python环境的**lib/python3.7/site-package/tensorflow** (或tensorflow_core)里创建一个**tensorflow_framework.so.x**指向**tensorflow_framework.so**的软链接：

  ```
  ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
  ```
* 删除 `BUILD` 文件第30行的  `-D_GLIBCXX_USE_CXX11_ABI=0`

# 4. 验证安装成功

```
cd <root_dir>
conda activate process_lrs3
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python 
# below are run in python console
> import face_alignment
> face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')
> import deep_3drecon
> deep_3drecon.Reconstructor()
```
