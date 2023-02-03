import numpy as np
import torch
import glob
import os
import tqdm
import librosa

import deep_3drecon
import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')
face_reconstructor = deep_3drecon.Reconstructor()
fa.get_landmarks(np.ones([224,224,3],dtype=np.uint8)) # 识别图片中的人脸，获得角点, shape=[68,2]
del fa
torch.cuda.empty_cache()


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2

def get_wav_features_from_fname(wav_path,
                      fft_size=512,
                      hop_size=320,
                      win_length=512,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=16000,
                      min_level_db=-100):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=False)
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    mel = mel.T
    # f0 = get_pitch(wav, mel)

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)

    wav = wav[:mel.shape[1] * hop_size]
    return wav.T, mel

def process_audio(fname, out_name=None):
    assert fname.endswith(".wav")
    tmp_name = fname[:-4] + '.doi'
    if os.path.exists(tmp_name):
        print("tmp exist, skip")
        return
    if out_name is None:
        out_name = fname[:-4] + '_audio.npy'
    if os.path.exists(out_name):
        print("out exisit, skip")
        return False
    os.system(f"touch {tmp_name}")

    wav, mel = get_wav_features_from_fname(fname)
    out_dict = {
        "mel": mel, # [T, 80]
    }
    np.save(out_name, out_dict)
    os.system(f"rm {tmp_name}")
    return True

if __name__ == '__main__':
    import os, glob
    lrs3_dir = "/home/yezhenhui/datasets/raw/lrs3_raw"
    wav_name_pattern = os.path.join(lrs3_dir, "*/*.wav")
    wav_names = glob.glob(wav_name_pattern)
    wav_names = sorted(wav_names)
    # you can run multiple processes in different GPUS to accelerate the deepspeech extraction. 
    # import random
    # random.shuffle(wav_names)
    # for wav_name in tqdm.tqdm(wav_names, desc='extracting Mel and Deepspeech'):
    #     doi_name = wav_name[:-4]+ '.doi'
    #     if os.path.exists(doi_name):
    #         os.remove(doi_name)
    #     doi_name = wav_name[:-4]+ '_audio.npy'
    #     if os.path.exists(doi_name):
    #         os.remove(doi_name)
    #     doi_name = wav_name[:-4]+ '_deepspeech.npy'
    #     if os.path.exists(doi_name):
    #         os.remove(doi_name)
    for wav_name in tqdm.tqdm(wav_names, desc='extracting Mel and Deepspeech'):
        process_audio(wav_name)