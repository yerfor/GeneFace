[中文文档](./zh/install_guide_nerf-zh.md)

This guide is about building a python environment, which is necessary to process the dataset for NeRF and train the GeneFace.

The following installation process is verified in RTX2080, Ubuntu 16.04. It may also work in newer environments (such as RTX3090 + ) with small modifications (such as changing a newer CUDA version).

# 1. Install the python libraries

```
conda create -n geneface python=3.8
conda activate geneface
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
# conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0 # RTX3090 etc.
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install -r docs/prepare_env/requirements_nerf.txt

```

# 2. Prepare the 3DMM Model

Apply in [this link](https://faces.dmi.unibas.ch/bfm/).

You can obtain a file named `BaselFaceModel.tgz`, extract a file named `01_MorphableModel.mat` from it and save it into the directory `./data_util/face_tracking/3DMM`

Then run the following commandlines:

```
cd data_util/face_tracking
conda activate 
python convert_BFM.py
```

# 3. Download DeepSpeech pretrained model（Required by AD-NeRF）

```bash
cd data_util/deepspeech_features
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.2/deepspeech-0.9.2-models.pbmm
```
