import os
import numpy as np
import torch
from inference.nerfs.base_nerf_infer import BaseNeRFInfer
from data_gen.nerf.binarizer import get_win_conds


class AdNeRFInfer(BaseNeRFInfer):
    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str), 'cond_name': (str, optional)}
        :return: a list that contains the condition feature of NeRF
        """
        self.save_wav16k(inp)
        if inp.get('cond_name', None) is not None:
            assert inp['cond_name'].endswith('.npy')
            deepspeech_arr = np.load(inp['cond_name']) # [T, w=16, c=29]
            print(f"I have Loaded pre-extracted deepspeech from {inp['cond_name']}!")
        else:
            wav16k_name = self.wav16k_name
            print(f"Trying to extract deepspeech from {wav16k_name}...")
            deepspeech_name = wav16k_name[:-4] + '_deepspeech.npy'
            if not os.path.exists(deepspeech_name):
                extract_deepspeech_cmd = f'python data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
                os.system(extract_deepspeech_cmd)
                print(f"I have extracted deepspeech features from {wav16k_name} to {deepspeech_name}.")
            else:
                print(f"I have Loaded pre-extracted deepspeech from {deepspeech_name}!")
            deepspeech_arr = np.load(deepspeech_name) # [T, w=16, c=29]
        
        num_samples = min(len(deepspeech_arr), self.infer_max_length)
        samples = [{} for _ in range(num_samples)]
        for idx, sample in enumerate(samples):
            sample['cond_win'] = torch.from_numpy(deepspeech_arr[idx]).float().unsqueeze(0) # [B=1, w=16, C=29]
            sample['cond_wins'] = torch.from_numpy(get_win_conds(deepspeech_arr, idx, smo_win_size=8)).float() #.unsqueeze(0) # [B=1,W=8, w=16, C=29]
        return samples

if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.hparams import hparams as hp
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_video_name': 'infer_outs/out.mp4',
            }
    AdNeRFInfer.example_run(inp)