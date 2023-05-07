# Docker Env for GeneFace

## Build image

```shell
cd docker/
docker build -t geneface:latest -f dockerfile .
cd ..
```

## Download weights (in parallel while building image)

```shell
# [DATA]
wget ? -O ./deep_3drecon/BFM/BaselFaceModel.tgz
cd ./deep_3drecon/BFM
tar -xvf BaselFaceModel.tgz PublicMM1/01_MorphableModel.mat --strip-components 1 
rm BaselFaceModel.tgz
cd ../../

mkdir -p ./deep_3drecon/checkpoints/facerecon/
wget ? -O ./deep_3drecon/BFM/Exp_Pca.bin
wget ? -O ./deep_3drecon/BFM/BFM_model_front.mat
wget ? -O ./deep_3drecon/checkpoints/facerecon/epoch_20.pth

# [PRETRAIN WEIGHTS]
wget https://github.com/yerfor/GeneFace/releases/download/v1.1.0/lrs3.zip -P checkpoints/
wget https://github.com/yerfor/GeneFace/releases/download/v1.1.0/May.zip -P checkpoints/
unzip checkpoints/lrs3.zip -d checkpoints/ && rm checkpoints/lrs3.zip
unzip checkpoints/May.zip -d checkpoints/ && rm checkpoints/May.zip
```

## Run Container

```shell
docker run -itd \
--name geneface \
--gpus all \
-v $(realpath .):/workspace/GeneFace \
-w /workspace/GeneFace \
--network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
geneface:latest bash

docker exec -it geneface bash
```

Prepare 3DMM for data preprocssing

```shell
cd data_util/face_tracking
python convert_BFM.py
cd ../../
```

Run preprocessing

```shell
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 data_gen/nerf/process_data.sh $VIDEO_ID
```

There are several weights to be downloaded

For training and inference please checkout `README.md`.

(by [xk-huang](https://github.com/xk-huang/))