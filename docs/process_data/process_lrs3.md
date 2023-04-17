# Process the LRS3 dataset

[中文文档](./zh/process_lrs3-zh.md)

We use LRS3 dataset to learn a robust audio2motion generator. It is also required for training a post-net and syncnet.

## Processed LRS3 Dataset available

🔥 Note: Since we turn to a new 3DMM extractor, we update the provided lrs3 dataset. You may need download the newest dataset file to be compatible with the latest code.

Since LRS3 is quite big (500 hours+), it is expensive to process this dataset. For your convenience, we provide the binarized LRS3 dataset file (about 26 GB) on Google Drive. If you use the processed dataset, you can skip `step 1-3` below and go directly to `step 4` to verify the installation.

- Download Link on Google Drive: [Partition 1](https://drive.google.com/drive/folders/1QK_ikLKUzGYiqHBzvKz0s5zKWeH-sm3L?usp=share_link), [Partition 2](https://drive.google.com/drive/folders/1WbECLfpxAZ0D7PcrlZxV-fCObT-TnfD8?usp=share_link).
- Download Link on Baiduyun Disk: [link](https://pan.baidu.com/s/1JsvEz58c9ItSI73ls43tTw?pwd=lrs3), passward: `lrs3`
- How to use:
  - step1. Integrate the segments `cat lrs3.zip.part_* > lrs3.zip` .
  - step2. Unzip `processed_lrs3.zip` and place it into the `data/binary/lrs3` folder.
  - step3. Move to `Step4. Verification` to verify installation.
- Disclaimer: the provided binarized dataset file only contains data-masked features (such as HuBERT for audio representations), so it does not viloate the copyright of LRS3.

If you ue our processed lrs3 dataset, you can skip the first 3 steps, and directly go to `step4`

## Step1. Apply and Download the LRS3-TED dataset

🔥 Note: It seems that the raw dataset of LRS3 is no longer provided by the official website.

Due to the License, we cannot provide a download link here. You can apply for LRS3-TED at [this link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/).

## Step2. Process the LRS3

For process lrs3, you should first install the python env `geneface` following the docs in  `dosc/prepare_env/install_guide.md`

Then run these commandlines: (You may need to modify the directory name of raw lrs3 in the .py files)

```
conda activate geneface
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_video_3dmm.py # extract 3dmm motion representations
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_mel_f0.py # extract mel spectrogram
CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_audio_hubert.py # extract hubert audio representations

```

Since the LRS3-TED dataset is relatively big, you may need run multiple processes in several GPUs to accelerate the data preprocessing, for instance:

```
# run on two GPUs

CUDA_VISIBLE_DEVICES=0 python data_gen/process_lrs3/process_video_3dmm.py --process_id=0 --total_process=2 \
& CUDA_VISIBLE_DEVICES=1 python data_gen/process_lrs3/process_video_3dmm.py --process_id=1 --total_process=2
```

## Step3. Binarize the dataset

run the following commandline to binarize the dataset. (You may need to modify the directory name of raw lrs3 in the .py files)

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
            val.data
```
