[中文文档](./zh/install_guide_lrs3-zh.md)

This guide is about building a python env to process the LRS3-TED dataset.

The following installation process is verified in RTX2080, Ubuntu 16.04. It may also work in newer environments (such as RTX3090 + ) with small modifications (such as changing a newer CUDA version).

# 1. Download 3D morphable face models

## 1.1  BFM09 Model

Apply in [this link](https://faces.dmi.unibas.ch/bfm/).

You can get a `BaselFaceModel.tgz`, unzip it, unzip it and get `01_MorphableModel.mat` and save it to the `./deep_3drecon/BFM` folder

## 1.2 PCA Basis

Download at [this link](https://drive.google.com/drive/folders/1iTopSpZucEmjWiWZIErLYiMBlZYwzil2?usp=share_link)

Extract the `Exp_Pca.bin` and place it to the `./deep_3drecon/BFM` directory.

## 1.3 BFM Model Front

Download at [this link](https://drive.google.com/drive/folders/1YCxXKJFfo1w01PzayhnxWSZZK5k7spSH?usp=share_link)

Extract the `BFM_model_front.mat` and place it to the `./deep_3drecon/BFM` directory.

Extract the `BFM_model_front.mat` and place it to the `./deep_util/BFM_models` directory.

# 2. Install the python libraries

Note: Please ensure that tensorflow-gpu is successfully installed with CUDA.

```
conda create -n process_lrs3 python=3.7.11
conda activate process_lrs3
conda install tensorflow-gpu=1.14.0 cudatoolkit=10.1
conda install pytorch=1.7.1 torchvision -c pytorch
pip install -r docs/prepare_env/requirements_lrs3.txt
```

# 3. Locally compile the Tensorflow-based mesh renderer

## 3.1 Install the bazel comiler

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

## 3.2 Then compile the mesh renderer

```
cd deep_3drecon/mesh_renderer/
touch WORKSPACE
bazel build //kernels:rasterize_triangles_kernel 
mv ./bazel-bin/kernels/rasterize_triangles_kernel.so ./kernels/
```

### 3.2.1 Frequent BUG

* cannot found tensorflow_framework: the solution is to build a soft link that connects the tensorflow_framework.so.x to tensorflow_framework.so in the TF directory: **lib/python3.7/site-package/tensorflow** (or tensorflow_core)

  ```
  cd xxx/lib/python3.7/site-package/tensorflow
  ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
  ```
* Compilation failed: delete the 30 line `-D_GLIBCXX_USE_CXX11_ABI=0` in the  `BUILD` file.

# 4. Verification of the Installation

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
