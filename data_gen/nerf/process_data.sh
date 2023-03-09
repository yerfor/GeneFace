export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=3
# 1. extrac 16khz wav
python data_util/process.py --video_id=$1 --task=1
# 2. extrac deepspeech and esperanto; 3.extract image frames 
python data_util/process.py --video_id=$1 --task=2 &
python data_util/process.py --video_id=$1 --task=3
# 7.detect landmarks
python data_util/process.py --video_id=$1 --task=7
# 4.face segmentation parsing; 8.estimate head pose
python data_util/process.py --video_id=$1 --task=4 &
python data_util/process.py --video_id=$1 --task=8
# 4. extract background image
python data_util/process.py --video_id=$1 --task=5
# Optional: Once the background image is extracted before running step 5,
# you could use a image inpainting tool (such as Inpaint on MacOS)
# to edit the backgroud image, so it could be more realistic.
# 5. save head, torso, gt imgs
python data_util/process.py --video_id=$1 --task=6
wait
# 7. integrate the results into meta
python data_util/process.py --video_id=$1 --task=9
# 8. calculate audio features
python data_gen/nerf/extract_hubert_mel_f0.py --video_id=$1
# 9. calculate 3DMM (you should run this with `process_lrs3`` python interpreter)
/home/yezhenhui/anaconda3/envs/process_lrs3/bin/python data_gen/nerf/extract_3dmm.py --video_id=$VIDEO_ID

python data_gen/nerf/binarizer.py --config=egs/datasets/videos/${VIDEO_ID}/lm3d_nerf.yaml
