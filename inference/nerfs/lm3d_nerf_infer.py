import os
import numpy as np
import torch
from inference.nerfs.base_nerf_infer import BaseNeRFInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda
import tqdm
import cv2
import importlib
import math
from utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans
from scipy.ndimage import gaussian_filter1d



def postprocess_lm3d(lm3d):
    from scipy.ndimage import gaussian_filter1d
    lm3d = lm3d.reshape([-1, 68, 3])
    is_bad_mask = lm3d[:, 66, 1] <= -1. # detect the outlier landmark
    is_bad_idices = np.where(is_bad_mask)[0]
    bad_slices = [] # list of (start_idx, dur)
    start_idx = None
    end_idx = None
    for idx in is_bad_idices:
        if start_idx is None:
            start_idx = idx
            end_idx = start_idx
        else:
            if idx - end_idx == 1:
                end_idx = idx
            else:
                bad_slices.append([start_idx, end_idx])
                start_idx = idx
                end_idx = start_idx
    bad_slices.append([start_idx, end_idx]) # the list of start,end index of outliers

    def local_smooth(arr, start_idx, end_idx, pad=5, smo_std=1.0):
        # smooth the segment of outliers with gaussian filter
        pad_start_idx = max(0, start_idx - pad)
        left_pad_length = start_idx - pad_start_idx
        pad_end_idx = min(len(arr), end_idx + 1 + pad)
        arr_slice = arr[pad_start_idx: pad_end_idx]
        smo_slice = gaussian_filter1d(arr_slice.reshape([-1, 68*3]), smo_std, axis=0).reshape([-1,68,3])
        smo_slice = smo_slice[left_pad_length:left_pad_length+(end_idx-start_idx+1)]
        if isinstance(arr, torch.Tensor):
            smo_slice = torch.from_numpy(smo_slice).float()
        arr[start_idx:end_idx+1] = smo_slice
        return arr
    # postprocess the outliers
    for (start,end) in bad_slices:
        lm3d = local_smooth(lm3d, start, end, pad=5, smo_std=1.5)
    return lm3d

class LM3dNeRFInfer(BaseNeRFInfer):

    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str), 'cond_name': (str, optional)}
        :return: a list that contains the condition feature of NeRF
        """
        self.save_wav16k(inp)

        # load the lm3d as the condition for lm3d head nerf
        assert inp['cond_name'].endswith('.npy')
        lm3d_arr = np.load(inp['cond_name'])[0] # [T, w=16, c=29]
        idexp_lm3d = torch.from_numpy(lm3d_arr).float()
        print(f"Loaded pre-extracted 3D landmark sequence from {inp['cond_name']}!")
        
        # load the deepspeech features as the condition for lm3d torso nerf
        wav16k_name = self.wav16k_name
        deepspeech_name = wav16k_name[:-4] + '_deepspeech.npy'
        if not os.path.exists(deepspeech_name):
            print(f"Try to extract deepspeech from {wav16k_name}...")
            deepspeech_python = '/home/yezhenhui/anaconda3/envs/geneface/bin/python' # the path of your python interpreter that has installed DeepSpeech
            extract_deepspeech_cmd = f'{deepspeech_python} data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
            os.system(extract_deepspeech_cmd)
            print(f"Saved deepspeech features of {wav16k_name} to {deepspeech_name}.")
        else:
            print(f"Try to load pre-extracted deepspeech from {deepspeech_name}...")
        deepspeech_arr = np.load(deepspeech_name) # [T, w=16, c=29]
        print(f"Loaded deepspeech features from {deepspeech_name}.")
        # get window condition of deepspeech
        from data_gen.nerf.binarizer import get_win_conds
        num_samples = min(len(lm3d_arr), len(deepspeech_arr), self.infer_max_length)
        samples = [{} for _ in range(num_samples)]
        for idx, sample in enumerate(samples):
            sample['deepspeech_win'] = torch.from_numpy(deepspeech_arr[idx]).float().unsqueeze(0) # [B=1, w=16, C=29]
            sample['deepspeech_wins'] = torch.from_numpy(get_win_conds(deepspeech_arr, idx, smo_win_size=8)).float() # [W=8, w=16, C=29]

        # get condition of idexp_lm3d
        idexp_lm3d = postprocess_lm3d(idexp_lm3d) # optional, we use it to reduce the bad case caused by outlier landmark.
        idexp_lm3d_mean = self.dataset.idexp_lm3d_mean
        idexp_lm3d_std = self.dataset.idexp_lm3d_std
        idexp_lm3d_normalized = (idexp_lm3d.reshape([-1,68,3]) - idexp_lm3d_mean)/idexp_lm3d_std * hparams['lm3d_scale_factor']
        idexp_lm3d_normalized = idexp_lm3d_normalized.reshape([-1, 68*3])

        idexp_lm3d_normalized_numpy = idexp_lm3d_normalized.cpu().numpy()
        idexp_lm3d_normalized_win_numpy = np.stack([get_win_conds(idexp_lm3d_normalized_numpy, i, smo_win_size=hparams['cond_win_size'], pad_option='edge') for i in range(idexp_lm3d_normalized_numpy.shape[0])])
        idexp_lm3d_normalized_win = torch.from_numpy(idexp_lm3d_normalized_win_numpy)

        for idx, sample in enumerate(samples):
            sample['cond'] = idexp_lm3d_normalized[idx].unsqueeze(0)
            if hparams['use_window_cond']:
                sample['cond_win'] = idexp_lm3d_normalized_win[idx]
                sample['cond_wins'] = torch.from_numpy(get_win_conds(idexp_lm3d_normalized_win_numpy, idx, hparams['smo_win_size'], 'edge'))
        return samples


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.hparams import hparams as hp
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'cond_name': 'infer_out/May/pred_lm3d/zozo.npy',
            'out_video_name': 'infer_out/May/pred_video/zozo.mp4',
            }

    LM3dNeRFInfer.example_run(inp)