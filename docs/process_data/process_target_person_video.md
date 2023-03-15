# Process the Target Person Video

[中文文档](./zh/process_target_person_video-zh.md)

You need a about 3-minute-long videos of the target person to train the person-specific postnet and NeRF-based renderer. The video is the longer the better.

We provide a example video at the path: `data/raw/videos/May.mp4`

## Only 1 step: extract all required features and binarize it
```
conda activate geneface
export PYTHONPATH=./
export VIDEO_ID=May
CUDA_VISIBLE_DEVICES=0 data_gen/nerf/process_data.sh $VIDEO_ID
```

Then you can find a directory at the path `data/binary/videos/May`
