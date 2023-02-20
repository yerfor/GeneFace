export PYTHONPATH=./
# 0. extrac deepspeech; 1.extract image frames 
python data_util/process_data.py --id=$1 --step=0 &
python data_util/process_data.py --id=$1 --step=1
# 2.detect landmarks
python data_util/process_data.py --id=$1 --step=2
# 3.face segmentation parsing; 6.estimate head pose
python data_util/process_data.py --id=$1 --step=6 &
python data_util/process_data.py --id=$1 --step=3
# 4. extract background image
python data_util/process_data.py --id=$1 --step=4
# Optional: Once the background image is extracted before running step 5,
# you could use a image inpainting tool (such as Inpaint on MacOS)
# to edit the backgroud image, so it could be more realistic.
# 5. save head and com imgs
python data_util/process_data.py --id=$1 --step=5
wait
# 7. integrate the results into meta
python data_util/process_data.py --id=$1 --step=7
