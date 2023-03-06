import torch
import numpy as np

from utils.commons.hparams import hparams

from tasks.radnerfs.dataset_utils import RADNeRFDataset
from inference.nerfs.lm3d_nerf_infer import LM3dNeRFInfer

class LM3d_RADNeRFInfer(LM3dNeRFInfer):
    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)
        self.dataset_cls = RADNeRFDataset # the dataset only provides head pose 
        self.dataset = self.dataset_cls('trainval')
        self.dataset.training = False

    def get_pose_from_ds(self, samples):
        """
        process the item into torch.tensor batch
        """
        for i, sample in enumerate(samples):
            ds_sample = self.dataset[i]
            sample['rays_o'] = ds_sample['rays_o']
            sample['rays_d'] = ds_sample['rays_d']
            sample['bg_coords'] = ds_sample['bg_coords']
            sample['pose'] = ds_sample['pose']
            sample['idx'] = ds_sample['idx']
            sample['bg_img'] = ds_sample['bg_img']
            sample['H'] = ds_sample['H']
            sample['W'] = ds_sample['W']
        return samples


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.hparams import hparams as hp
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'cond_name': 'infer_out/May/pred_lm3d/zozo.npy',
            'out_video_name': 'infer_out/May/pred_video/zozo.mp4',
            }

    LM3d_RADNeRFInfer.example_run(inp)