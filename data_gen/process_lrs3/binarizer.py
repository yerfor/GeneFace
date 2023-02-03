import os
import numpy as np
from scipy.misc import face
import torch
from tqdm import trange
import pickle
from copy import deepcopy
from data_util.face3d_helper import Face3DHelper

class IndexedDataset:
    def __init__(self, path, num_cache=10):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})


def load_video_npy(fn):
    assert fn.endswith(".npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    video_dict = {
        'coeff': ret_dict['coeff'], # [T, h]
        'lm68': ret_dict['lm68'], # [T, 68, 2]
        'lm5': ret_dict['lm5'], # [T, 5, 2]
    }
    return video_dict

def cal_lm3d_in_video_dict(video_dict, face3d_helper):
    coeff = torch.from_numpy(video_dict['coeff']).float()
    identity = coeff[:, 0:80]
    exp = coeff[:, 80:144]
    idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d(identity, exp).cpu().numpy()
    video_dict['idexp_lm3d'] = idexp_lm3d

def load_audio_npy(fn):
    assert fn.endswith(".npy")
    ret_dict = np.load(fn,allow_pickle=True).item()
    audio_dict = {
        "mel": ret_dict['mel'], # [T, 80]
        "energy": ret_dict['energy'], # [T,1]
    }
    return audio_dict


if __name__ == '__main__':
    face3d_helper = Face3DHelper(use_gpu=False)
    
    import glob,tqdm
    prefixs = ['val', 'train']
    binarized_ds_path = "/home/yezhenhui/datasets/binary/lrs3_tmp"
    os.makedirs(binarized_ds_path, exist_ok=True)
    for prefix in prefixs:
        databuilder = IndexedDatasetBuilder(os.path.join(binarized_ds_path, prefix))
        raw_base_dir =  '/home/yezhenhui/datasets/raw/lrs3_raw'
        spk_ids = sorted([dir_name.split("/")[-1] for dir_name in glob.glob(raw_base_dir + "/*")])
        spk_id2spk_idx = {spk_id : i for i,spk_id in enumerate(spk_ids) }
        np.save(os.path.join(binarized_ds_path, "spk_id2spk_idx.npy"), spk_id2spk_idx, allow_pickle=True)
        mp4_names = glob.glob(raw_base_dir + "/*/*.mp4")
        cnt = 0
        for i, mp4_name in tqdm.tqdm(enumerate(mp4_names), total=len(mp4_names)):
            if prefix == 'train':
                if i % 100 == 0:
                    continue
            else:
                if i % 100 != 0:
                    continue
            lst = mp4_name.split("/")
            spk_id = lst[-2]
            clip_id = lst[-1][:-4]
            audio_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+"_audio.npy")
            hubert_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+"_hubert.npy")
            video_npy_name = os.path.join(raw_base_dir, spk_id, clip_id+".npy")
            if (not os.path.exists(audio_npy_name)) or (not os.path.exists(video_npy_name)):
                print(f"Skip item for not found.")
                continue
            if (not os.path.exists(hubert_npy_name)):
                print(f"Skip item for hubert_npy not found.")
                continue
            audio_dict = load_audio_npy(audio_npy_name)
            hubert = np.load(hubert_npy_name)
            video_dict = load_video_npy(video_npy_name)
            cal_lm3d_in_video_dict(video_dict, face3d_helper)
            mel = audio_dict['mel']
            if mel.shape[0] < 64:
                print(f"Skip item for too short.")
                continue
            audio_dict.update(video_dict)
            audio_dict['spk_id'] = spk_id
            audio_dict['spk_idx'] = spk_id2spk_idx[spk_id]
            audio_dict['item_id'] = spk_id + "_" + clip_id
            
            audio_dict['hubert'] = hubert # [T_x, hid=1024]
            databuilder.add_item(audio_dict)
            cnt += 1
        databuilder.finalize()
        print(f"{prefix} set has {cnt} samples!")