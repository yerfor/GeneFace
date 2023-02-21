[中文文档](./zh/process_lrs3-zh.md)

# Process the LRS3 dataset

We use LRS3 dataset to learn a robust audio2motion generator. It is also required for training a post-net and syncnet.

## Processed LRS3 Dataset available
🔥 Update: Since LRS3 is quite big (500 hours+), it is expensive to process this dataset (e.g., four RTX2080ti GPUs for a week). For your convenience, we provide the binarized LRS3 dataset file (about 25 GB) on Google Drive. If you use the processed dataset, you can skip `step 1-3` below and go directly to `step 4` to verify the installation.
- Download link: [Partition 1](https://drive.google.com/file/d/1ScyB4DeKNCcyMNvVx6Gz39tOmN3h4Gsk/view?usp=share_link), [Partition 2](https://drive.google.com/file/d/1treFjXaWgYom3dcj8p_NyNvWktE25wTK/view?usp=share_link). 
- How to use: 
    - step1. Integrate the segments `cat processed_lrs3_.zip.part_* > processed_lrs3.zip` .
    - step2. Unzip `processed_lrs3.zip` and place it into the `data/binary/lrs3` folder.
    - step3. Move to `Step4. Verification` to verify installation.
- Disclaimer: the provided binarized dataset file only contains data-masked features (such as HuBERT for audio representations), so it does not viloate the copyright of LRS3.

If you ue our processed lrs3 dataset, you can skip the first 3 steps, and directly go to `step4`

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

# Step4. Verification
Then you may find a directory at the path  `data/binary/lrs3/`
After the above steps, the structure of your `data` directory should look like this:

```
> data
    > binary
        > lrs3
            sizes_train.npy
            sizes_val.npy
            spk_id2ispk_idx.npy
            stats.npy
            train.data
            train.idx
            val.data
            val.idx
```