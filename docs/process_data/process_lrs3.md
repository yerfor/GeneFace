[中文文档](./zh/process_lrs3-zh.md)

# Process the LRS3 dataset

We use LRS3 dataset to learn a robust audio2motion generator. It is also required for training a post-net and syncnet.

🔥 Update: Since LRS3 is quite big (500 hours+), it is expensive to process this dataset (e.g., four RTX2080ti GPUs for a week). For your convenience, we provide the binarized LRS3 dataset file (about 35 GB) on Baidu Drive. 
- Download at [this link](https://pan.baidu.com/s/1fLu7c0lYv3FhGLH6YJsZbw?pwd=lrs3) with the password `lrs3`. 
- How to use: 
    - step1. Integrate the segments `cat lrs3_0722.zip.part_* > lrs3_0722.zip` .
    - step2. Unzip `lrs3_0722.zip` and place it into the `data/binary/lrs3` folder.
- Disclaimer: the provided binarized dataset file only contains data-masked features (such as HuBERT for audio representations), so it does not viloate the copyright of LRS3.

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
