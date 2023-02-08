[中文文档](./zh/process_target_person_video-zh.md)

# Process the Target Person Video

You need a about 3-minute-long videos of the target person to train the person-specific postnet and NeRF-based renderer. The video is the longer the better.

We provide a example video at the path: `data/raw/videos/May.mp4`

## Step1. extract features required by NeRF

```
conda activate geneface
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 data_gen/nerf/process_data.sh $VIDEO_ID


```

## Step2. extract hubert/3dmm from the video

These features are required to train the postnet.

run the following commandlines:

```
conda activate process_lrs3
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 python data_gen/nerf/extract_hubert.py --video_id=$VIDEO_ID
CUDA_VISIBLE_DEVICES=0 python data_gen/nerf/extract_3dmm.py --video_id=$VIDEO_ID
```

## Step3. binarize the dataset

```
conda activate geneface
export PYTHONPATH=./
python data_gen/nerf/binarizer.py --config=egs/datasets/videos/May/lm3d_nerf.yaml

```

Then you can find a directory at the path `data/binary/videos/May`
