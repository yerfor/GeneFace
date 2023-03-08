import os
import tqdm
import torch
import cv2
import numpy as np

from scipy.spatial.transform import Rotation

from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import convert_to_tensor
from utils.commons.image_utils import load_image_as_uint8_tensor

from modules.radnerfs.utils import get_audio_features, get_rays, get_bg_coords, convert_poses, nerf_matrix_to_ngp


def smooth_camera_path(poses, kernel_size=7):
    # smooth the camera trajectory (i.e., translation)...
    # poses: [N, 4, 4], numpy array
    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        try:
            poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()
        except:
            if i == 0:
                poses[i, :3, :3] = rots[i]
            else:
                poses[i, :3, :3] = poses[i-1, :3, :3]
    return poses


class RADNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, prefix, data_dir=None, training=True):
        super().__init__()
        self.data_dir = os.path.join(hparams['binary_data_dir'], hparams['video_id']) if data_dir is None else data_dir
        binary_file_name = os.path.join(self.data_dir, "trainval_dataset.npy")
        ds_dict = np.load(binary_file_name, allow_pickle=True).tolist()
        if prefix == 'train':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']]
        elif prefix == 'val':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        elif prefix == 'trainval':
            self.samples = [convert_to_tensor(sample) for sample in ds_dict['train_samples']] + [convert_to_tensor(sample) for sample in ds_dict['val_samples']]
        else:
            raise ValueError("prefix should in train/val !")
        self.prefix = prefix
        self.cond_type = hparams['cond_type']
        self.H = ds_dict['H']
        self.W = ds_dict['W']
        self.focal = ds_dict['focal']
        self.cx = ds_dict['cx']
        self.cy = ds_dict['cy']
        self.near = hparams['near'] # follow AD-NeRF, we dont use near-far in ds_dict
        self.far = hparams['far'] # follow AD-NeRF, we dont use near-far in ds_dict
        if hparams['infer_bg_img_fname'] == '':
            # use the default bg_img from dataset
            bg_img = torch.from_numpy(ds_dict['bg_img']).float() / 255.
        elif hparams['infer_bg_img_fname'] == 'white': # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif hparams['infer_bg_img_fname'] == 'black': # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else: # load from a specificfile
            bg_img = cv2.imread(hparams['infer_bg_img_fname'], cv2.IMREAD_UNCHANGED) # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255 # [H, W, 3/4]
        self.bg_img = convert_to_tensor(bg_img)

        self.idexp_lm3d_mean = torch.from_numpy(ds_dict['idexp_lm3d_mean']).float()
        self.idexp_lm3d_std = torch.from_numpy(ds_dict['idexp_lm3d_std']).float()

        fl_x = fl_y = self.focal
        self.intrinsics = np.array([fl_x, fl_y, self.cx, self.cy])
        self.poses = torch.from_numpy(np.stack([nerf_matrix_to_ngp(s['c2w'], scale=hparams['camera_scale'], offset=hparams['camera_offset']) for s in self.samples]))
        if not training and hparams['infer_smooth_camera_path']:
            smo_poses = smooth_camera_path(self.poses.numpy(), kernel_size=hparams['infer_smooth_camera_path_kernel_size'])
            self.poses = torch.from_numpy(smo_poses)
            print(f"Smooth head trajectory (rotation and translation) with a window size of {hparams['infer_smooth_camera_path_kernel_size']}")
        self.bg_coords = get_bg_coords(self.H, self.W, 'cpu') # [1, H*W, 2] in [-1, 1]

        if self.cond_type == 'deepspeech':
            self.conds = torch.stack([s['deepspeech_win'] for s in self.samples]) # [B=1, T=16, C=29]
        elif self.cond_type == 'esperanto':
            self.conds = torch.stack([s['esperanto_win'] for s in self.samples]) # [B=1, T=16, C=44]
        elif self.cond_type == 'idexp_lm3d_normalized':
            self.conds = torch.stack([s['idexp_lm3d_normalized_win'].reshape([hparams['cond_win_size'], 204]) for s in self.samples]) # [B=1, T=1, C=204]
        else:
            raise NotImplementedError
        
        self.finetune_lip_flag = False
        self.lips_rect = []
        for sample in self.samples:
            img_id = sample['idx']
            lms = np.loadtxt(os.path.join(hparams['processed_data_dir'],hparams['video_id'], 'ori_imgs', str(img_id) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = max(0, cx - l)
            xmax = min(self.H, cx + l)
            ymin = max(0, cy - l)
            ymax = min(self.W, cy + l)
            self.lips_rect.append([xmin, xmax, ymin, ymax])

        self.training = training
        self.global_step = 0

    @property
    def num_rays(self):
        return hparams['n_rays'] if self.training else -1

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        
        if 'torso_img' not in self.samples[idx].keys():
            self.samples[idx]['torso_img'] = load_image_as_uint8_tensor(self.samples[idx]['torso_img_fname'])
        if 'gt_img' not in self.samples[idx].keys():
            self.samples[idx]['gt_img'] = load_image_as_uint8_tensor(self.samples[idx]['gt_img_fname'])

        sample = {
            'H': self.H,
            'W': self.W,
            'focal': self.focal,
            'cx': self.cx,
            'cy': self.cy,
            'near': self.near,
            'far': self.far,
            'idx': raw_sample['idx'],
            'face_rect': raw_sample['face_rect'],
            'lip_rect': self.lips_rect[idx],
            'bg_img': self.bg_img,
        }

        if self.cond_type == 'deepspeech':
            sample.update({
                'cond_win': raw_sample['deepspeech_win'].unsqueeze(0), # [B=1, T=16, C=29]
                'cond_wins': raw_sample['deepspeech_wins'], # [Win=8, T=16, C=29]
            })
        elif self.cond_type == 'esperanto':
            sample.update({
                'cond_win': raw_sample['esperanto_win'].unsqueeze(0), # [B=1, T=16, C=29]
                'cond_wins': raw_sample['esperanto_wins'], # [Win=8, T=16, C=29]
            })
        elif self.cond_type == 'idexp_lm3d_normalized':
            sample['cond'] = raw_sample['idexp_lm3d_normalized'].reshape([1,-1]) # [1, 204]
            sample['cond_win'] = raw_sample['idexp_lm3d_normalized_win'].reshape([1, hparams['cond_win_size'],-1]) # [1, T_win, 204]
            sample['cond_wins'] = raw_sample['idexp_lm3d_normalized_wins'].reshape([hparams['smo_win_size'], hparams['cond_win_size'],-1]) # [smo_win, T_win, 204]
        else:
            raise NotImplementedError
        
        ngp_pose = self.poses[idx].unsqueeze(0)
        sample['pose'] = convert_poses(ngp_pose) # [B, 6]
        sample['pose_matrix'] = ngp_pose # [B, 4, 4]

        sample.update({
            'torso_img': raw_sample['torso_img'].float() / 255.,
            'gt_img': raw_sample['gt_img'].float() / 255.,
        })
        
        if self.training and self.finetune_lip_flag:
            # the finetune_lip_flag is controlled by the task that use this dataset 
            rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=-1, rect=sample['lip_rect'])
        else:
            rays = get_rays(ngp_pose.cuda(), self.intrinsics, self.H, self.W, N=self.num_rays)
        sample['rays_o'] = rays['rays_o']
        sample['rays_d'] = rays['rays_d']

        xmin, xmax, ymin, ymax = raw_sample['face_rect']
        face_mask = (rays['j'] >= xmin) & (rays['j'] < xmax) & (rays['i'] >= ymin) & (rays['i'] < ymax) # [B, N]
        sample['face_mask'] = face_mask

        bg_torso_img = sample['torso_img']
        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.view(1, -1, 3) # treat torso as a part of background
        bg_img =  self.bg_img.view(1, -1, 3)
        
        C = sample['gt_img'].shape[-1]
        if self.training:
            bg_img = torch.gather(bg_img.cuda(), 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            bg_torso_img = torch.gather(bg_torso_img.cuda(), 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            gt_img = torch.gather(sample['gt_img'].reshape(1, -1, C).cuda(), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            sample['gt_img'] = gt_img
        else:
            sample['gt_img'] = sample['gt_img'].reshape([1,-1,C])
        sample['bg_img'] = bg_img
        sample['bg_torso_img'] = bg_torso_img

        if self.training:
            bg_coords = torch.gather(self.bg_coords.cuda(), 1, torch.stack(2 * [rays['inds']], -1)) # [1, N, 2]
        else:
            bg_coords = self.bg_coords # [1, N, 2]
        sample['bg_coords'] = bg_coords


        return sample
    
    def __len__(self):
        return len(self.samples)

    def collater(self, samples):
        assert len(samples) == 1 # NeRF only take 1 image for each iteration
        return samples[0]
 
if __name__ == '__main__':
    set_hparams()
    ds = RADNeRFDataset('trainval', data_dir='data/binary/videos/May')
    ds[0]
    print("done!")