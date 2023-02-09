[中文文档](./zh/process_lrs3-zh.md)

# Process the Target Person Video

You need a about 3-minute-long videos of the target person to train the person-specific postnet and NeRF-based renderer

# Process the LRS3 dataset

We use LRS3 dataset to learn a robust audio2motion generator. It is also required for training a post-net and syncnet.

## Step1. Apply and Download the LRS3-TED dataset

Due to the License, we cannot provide a download link here. You can apply for LRS3-TED at [this link]().

## Step2. Process the LRS3

You should first install the python env for process lrs3, following the docs in `dosc/prepare_env/install_guide_lrs3.md`

Then run these commandlines: (You may need to modify the directory name in the .py files)

```
conda activate procerss_lrs3
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_video.py # extract 3dmm motion representations
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_mel.py # extract mel spectrogram
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_hubert.py # extract hubert audio representations

```

Since the LRS3-TED dataset is relatively big, you may need run multiple process in several GPUs to accelerate the data preprocessing.

## Step3. Binarize the dataset

run the following commandline to binarize the dataset. (You may need to modify the directory name in the .py files)

```
conda activate procerss_lrs3
python data_gen/process_lrs3/binarizer.py
```

Then you may find a directory at the path  `data/binary/lrs3/`
