import os
import numpy as np
import math
import json
import imageio
import torch
from data_util.face3d_helper import Face3DHelper

from utils.commons.euler2rot import euler_trans_2_c2w, c2w_to_euler_trans
from tasks.audio2motion.dataset_utils.euler2quaterion import euler2quaterion, quaterion2euler
import tqdm

from utils.commons.hparams import hparams, set_hparams
set_hparams()

face3d_helper = Face3DHelper()

audio_cond_win_size = 16
audio_smo_win_size = 8
exp_cond_win_size = hparams['cond_win_size']
exp_smo_win_size = hparams['smo_win_size']


def get_win_conds(conds, idx, smo_win_size=8, pad_option='zero'):
    """
    conds: [b, t=16, h=29]
    idx: long, time index of the selected frame
    """
    idx = max(0, idx)
    idx = min(idx, conds.shape[0]-1)
    smo_half_win_size = smo_win_size//2
    left_i = idx - smo_half_win_size
    right_i = idx + (smo_win_size - smo_half_win_size)
    pad_left, pad_right = 0, 0
    if left_i < 0:
        pad_left = -left_i
        left_i = 0
    if right_i > conds.shape[0]:
        pad_right = right_i - conds.shape[0]
        right_i = conds.shape[0]
    conds_win = conds[left_i:right_i]
    if pad_left > 0:
        if pad_option == 'zero':
            conds_win = np.concatenate([np.zeros_like(conds_win)[:pad_left], conds_win], axis=0)
        elif pad_option == 'edge':
            edge_value = conds[0][np.newaxis, ...]
            conds_win = np.concatenate([edge_value] * pad_left + [conds_win], axis=0)
        else: 
            raise NotImplementedError
    if pad_right > 0:
        if pad_option == 'zero':
            conds_win = np.concatenate([conds_win, np.zeros_like(conds_win)[:pad_right]], axis=0)
        elif pad_option == 'edge':
            edge_value = conds[-1][np.newaxis, ...]
            conds_win = np.concatenate([conds_win] + [edge_value] * pad_right , axis=0)
        else: 
            raise NotImplementedError
    assert conds_win.shape[0] == smo_win_size
    return conds_win


def load_data(processed_dir):    
    com_img_dir = os.path.join(processed_dir, "com_imgs")
    head_img_dir = os.path.join(processed_dir, "head_imgs")
    ori_img_dir = os.path.join(processed_dir, "ori_imgs")
    parsing_dir = os.path.join(processed_dir, "parsing")
    background_img_name = os.path.join(processed_dir, "bc.jpg")
    train_json_name = os.path.join(processed_dir, "transforms_train.json")
    val_json_name = os.path.join(processed_dir, "transforms_val.json")
    track_params_name = os.path.join(processed_dir, "track_params.pt")
    deepspeech_npy_name = os.path.join(processed_dir, "aud.npy")
    config_txt_name = os.path.join(processed_dir, "HeadNeRF_config.txt")
    coeff_npy_name = os.path.join(processed_dir, "coeff.npy")
    hubert_npy_name = os.path.join(processed_dir, "hubert.npy")
    
    ret_dict = {}

    print("loading deepspeech ...")
    deepspeech_features = np.load(deepspeech_npy_name)
    print("loading hubert ...")
    hubert_features = np.load(hubert_npy_name)
    ret_dict['hubert'] = hubert_features

    print("loading 3dmm coeff ...")
    coeff_dict = np.load(coeff_npy_name, allow_pickle=True).tolist()
    coeff_arr = coeff_dict['coeff'][:]
    
    identity_arr = coeff_arr[:, 0:80]
    exp_arr = coeff_arr[:, 80:144]

    print("calculating lm3d ...")
    idexp_lm3d_arr = face3d_helper.reconstruct_idexp_lm3d(torch.from_numpy(identity_arr), torch.from_numpy(exp_arr)).cpu().numpy()
    
    video_idexp_lm3d_mean = idexp_lm3d_arr.mean(axis=0).reshape([1,68,3])
    video_idexp_lm3d_std = idexp_lm3d_arr.std(axis=0).reshape([1,68,3])
    ret_dict['idexp_lm3d_mean'] = video_idexp_lm3d_mean
    ret_dict['idexp_lm3d_std'] = video_idexp_lm3d_std
    idexp_lm3d_arr_normalized = (idexp_lm3d_arr - video_idexp_lm3d_mean) / video_idexp_lm3d_std

    if deepspeech_features.shape[0] < coeff_arr.shape[0]:
        num_to_pad = coeff_arr.shape[0] - deepspeech_features.shape[0]
        tmp = np.zeros([num_to_pad, 16, 29])
        deepspeech_features = np.concatenate([deepspeech_features, tmp], axis=0)
    elif deepspeech_features.shape[0] > coeff_arr.shape[0]:
        deepspeech_features = deepspeech_features[:coeff_arr.shape[0]]

    translation = coeff_arr[:, 254:257] # [T_y, c=3]
    angles = euler2quaterion(coeff_arr[:, 224:227]) # # [T_y, c=4]
    pose_deep3drecon = np.concatenate([translation, angles], axis=1)

    print("loading train_val.json ...")
    with open(train_json_name) as f:
        train_meta = json.load(f)
    with open(val_json_name) as f:
        val_meta = json.load(f)
    bc_img = imageio.imread(background_img_name)
    ret_dict['bc_img'] = bc_img
    ret_dict['H'], ret_dict['W'] = bc_img.shape[:2]
    ret_dict['focal'], ret_dict['cx'], ret_dict['cy'] = float(train_meta['focal_len']), float(train_meta['cx']), float(train_meta['cy'])

    print("loading config.txt ...")
    with open(config_txt_name) as f:
        config_txt = f.readlines()
        for line in config_txt:
            if line.startswith("near"):
                ret_dict['near'] = eval(line.split("=")[-1])
            elif line.startswith("far"):
                ret_dict['far'] = eval(line.split("=")[-1])

    idexp_lm3d_normalized_win_lst = []
    hubert_win_lst = []
    for frame in train_meta['frames'] + val_meta['frames'] :
        idx = frame['aud_id']
        idexp_lm3d_normalized_win = get_win_conds(idexp_lm3d_arr_normalized, idx, smo_win_size=exp_cond_win_size, pad_option='zero')
        idexp_lm3d_normalized_win_lst.append(idexp_lm3d_normalized_win)
        hubert_win = get_win_conds(hubert_features, idx, smo_win_size=16)
        hubert_win_lst.append(hubert_win)
    idexp_lm3d_normalized_wins_arr = np.stack(idexp_lm3d_normalized_win_lst, axis=0) # [T, t_w, 204]
    hubert_win_arr = np.stack(hubert_win_lst, axis=0) # [T, t_w, 204]
   
    # obtaining train samples
    train_samples = []
    for i_frame, frame in tqdm.tqdm(enumerate(train_meta['frames']), desc="Binarizing train set", total=len(train_meta['frames'])):
        assert frame['aud_id'] == frame['img_id']
        idx = frame['aud_id']
        ori_img_fname = os.path.join(ori_img_dir,f"{idx}.jpg")
        head_img_fname = os.path.join(head_img_dir,f"{idx}.jpg")
        com_img_fname = os.path.join(com_img_dir,f"{idx}.jpg")
        parsing_fname = os.path.join(parsing_dir,f"{idx}.png")
        
        camera2world_matrix = np.array(frame['transform_matrix'])
        euler, trans = c2w_to_euler_trans(camera2world_matrix)
        face_rect = np.array(frame['face_rect'])
        deepspeech_wins = get_win_conds(deepspeech_features, idx, smo_win_size=audio_smo_win_size, pad_option='zero')

        idexp_lm3d_normalized_win = get_win_conds(idexp_lm3d_arr_normalized, idx, smo_win_size=exp_cond_win_size, pad_option='zero') # [cond_win_size, 68, 3]
        idexp_lm3d_normalized_wins = get_win_conds(idexp_lm3d_normalized_wins_arr, idx, smo_win_size=exp_smo_win_size, pad_option='zero') # [smo_win_size, cond_win_size, 68, 3]

        hubert_win = hubert_win_arr[idx]
        hubert_wins = get_win_conds(hubert_win_arr, idx, smo_win_size=8, pad_option='zero')

        sample = {
            'idx': idx,
            'face_rect': face_rect,
            'ori_img_fname': ori_img_fname,
            'head_img_fname': head_img_fname,
            'com_img_fname': com_img_fname,
            'parsing_fname': parsing_fname,
            'c2w': camera2world_matrix,
            'euler': euler,
            'trans': trans,
            'exp': exp_arr[idx],
            'identity': identity_arr[idx],
            'pose_deep3drecon': pose_deep3drecon[idx],
            'idexp_lm3d': idexp_lm3d_arr[idx],
            'idexp_lm3d_normalized': idexp_lm3d_arr_normalized[idx],
            'idexp_lm3d_normalized_win': idexp_lm3d_normalized_win,
            'idexp_lm3d_normalized_wins': idexp_lm3d_normalized_wins,
            'deepspeech_win': deepspeech_features[idx],
            'hubert_win': hubert_win,
            'deepspeech_wins': deepspeech_wins,
            'hubert_wins': hubert_wins,
        }
        train_samples.append(sample)
    ret_dict['train_samples'] = train_samples
    
    # obtaining val samples
    val_samples = []
    for i_frame, frame in tqdm.tqdm(enumerate(val_meta['frames']), desc="Binarizing val set", total=len(val_meta['frames'])):
        assert frame['aud_id'] == frame['img_id']
        idx = frame['aud_id']
        ori_img_fname = os.path.join(ori_img_dir,f"{idx}.jpg")
        head_img_fname = os.path.join(head_img_dir,f"{idx}.jpg")
        com_img_fname = os.path.join(com_img_dir,f"{idx}.jpg")
        parsing_fname = os.path.join(parsing_dir,f"{idx}.png")
        
        face_rect = np.array(frame['face_rect'])
        camera2world_matrix = np.array(frame['transform_matrix'])
        euler, trans = c2w_to_euler_trans(camera2world_matrix)
        deepspeech_wins = get_win_conds(deepspeech_features, idx, smo_win_size=audio_smo_win_size, pad_option='zero')
        
        idexp_lm3d_normalized_win = get_win_conds(idexp_lm3d_arr_normalized, idx, smo_win_size=exp_cond_win_size, pad_option='zero')
        idexp_lm3d_normalized_wins = get_win_conds(idexp_lm3d_normalized_wins_arr, idx, smo_win_size=exp_smo_win_size, pad_option='zero')

        hubert_win = hubert_win_arr[idx]
        hubert_wins = get_win_conds(hubert_win_arr, idx, smo_win_size=8, pad_option='zero')

        sample = {
            'idx': idx,
            'face_rect': face_rect,
            'ori_img_fname': ori_img_fname,
            'head_img_fname': head_img_fname,
            'com_img_fname': com_img_fname,
            'parsing_fname': parsing_fname,
            'c2w': camera2world_matrix,
            'euler': euler,
            'trans': trans,
            'exp':  exp_arr[idx], # [64]
            'identity': identity_arr[idx],
            'pose_deep3drecon': pose_deep3drecon[idx],
            'idexp_lm3d': idexp_lm3d_arr[idx],
            'idexp_lm3d_normalized': idexp_lm3d_arr_normalized[idx],
            'idexp_lm3d_normalized_win': idexp_lm3d_normalized_win,
            'idexp_lm3d_normalized_wins': idexp_lm3d_normalized_wins,
            'deepspeech_win': deepspeech_features[idx], # [16,29]
            'hubert_win': hubert_win,
            'deepspeech_wins': deepspeech_wins,
            'hubert_wins': hubert_wins,
        }

        val_samples.append(sample)    
    ret_dict['val_samples'] = val_samples

    return ret_dict


def load_adnerf_data(processed_dir):    
    com_img_dir = os.path.join(processed_dir, "com_imgs")
    head_img_dir = os.path.join(processed_dir, "head_imgs")
    ori_img_dir = os.path.join(processed_dir, "ori_imgs")
    parsing_dir = os.path.join(processed_dir, "parsing")
    background_img_name = os.path.join(processed_dir, "bc.jpg")
    train_json_name = os.path.join(processed_dir, "transforms_train.json")
    val_json_name = os.path.join(processed_dir, "transforms_val.json")
    track_params_name = os.path.join(processed_dir, "track_params.pt")
    deepspeech_npy_name = os.path.join(processed_dir, "aud.npy")
    config_txt_name = os.path.join(processed_dir, "HeadNeRF_config.txt")
    
    ret_dict = {}

    deepspeech_features = np.load(deepspeech_npy_name)

    with open(train_json_name) as f:
        train_meta = json.load(f)
    with open(val_json_name) as f:
        val_meta = json.load(f)
    bc_img = imageio.imread(background_img_name)
    ret_dict['bc_img'] = bc_img
    ret_dict['H'], ret_dict['W'] = bc_img.shape[:2]
    ret_dict['focal'], ret_dict['cx'], ret_dict['cy'] = float(train_meta['focal_len']), float(train_meta['cx']), float(train_meta['cy'])

    with open(config_txt_name) as f:
        config_txt = f.readlines()
        for line in config_txt:
            if line.startswith("near"):
                ret_dict['near'] = eval(line.split("=")[-1])
            elif line.startswith("far"):
                ret_dict['far'] = eval(line.split("=")[-1])

    train_samples = []

    for i_frame, frame in tqdm.tqdm(enumerate(train_meta['frames']), desc="Binarizing train set", total=len(train_meta['frames'])):
        assert frame['aud_id'] == frame['img_id']
        idx = frame['aud_id']
        ori_img_fname = os.path.join(ori_img_dir,f"{idx}.jpg")
        head_img_fname = os.path.join(head_img_dir,f"{idx}.jpg")
        com_img_fname = os.path.join(com_img_dir,f"{idx}.jpg")
        parsing_fname = os.path.join(parsing_dir,f"{idx}.png")
        
        camera2world_matrix = np.array(frame['transform_matrix'])
        euler, trans = c2w_to_euler_trans(camera2world_matrix)
        face_rect = np.array(frame['face_rect'])
        deepspeech_wins = get_win_conds(deepspeech_features, idx, smo_win_size=audio_smo_win_size)
        sample = {
            'idx': idx,
            'face_rect': face_rect,
            'ori_img_fname': ori_img_fname,
            'head_img_fname': head_img_fname,
            'com_img_fname': com_img_fname,
            'parsing_fname': parsing_fname,
            'c2w': camera2world_matrix,
            'euler': euler,
            'trans': trans,
            'deepspeech_win': deepspeech_features[idx],
            'deepspeech_wins': deepspeech_wins,
        }
        train_samples.append(sample)
    ret_dict['train_samples'] = train_samples

    val_samples = []
    for i_frame, frame in tqdm.tqdm(enumerate(val_meta['frames']), desc="Binarizing val set", total=len(val_meta['frames'])):
        assert frame['aud_id'] == frame['img_id']
        idx = frame['aud_id']
        ori_img_fname = os.path.join(ori_img_dir,f"{idx}.jpg")
        head_img_fname = os.path.join(head_img_dir,f"{idx}.jpg")
        com_img_fname = os.path.join(com_img_dir,f"{idx}.jpg")
        parsing_fname = os.path.join(parsing_dir,f"{idx}.png")
        
        face_rect = np.array(frame['face_rect'])
        camera2world_matrix = np.array(frame['transform_matrix'])
        euler, trans = c2w_to_euler_trans(camera2world_matrix)
        deepspeech_wins = get_win_conds(deepspeech_features, idx, smo_win_size=audio_smo_win_size, pad_option='zero')
        sample = {
            'idx': idx,
            'face_rect': face_rect,
            'ori_img_fname': ori_img_fname,
            'head_img_fname': head_img_fname,
            'com_img_fname': com_img_fname,
            'parsing_fname': parsing_fname,
            'c2w': camera2world_matrix,
            'euler': euler,
            'trans': trans,
            'deepspeech_win': deepspeech_features[idx], # [16,29]
            'deepspeech_wins': deepspeech_wins,
        }

        val_samples.append(sample)    
    ret_dict['val_samples'] = val_samples

    return ret_dict



class Binarizer:
    def __init__(self):
        self.data_dir = 'data/'
        
    def parse(self, video_id):
        processed_dir = os.path.join(self.data_dir, 'processed/videos', video_id)
        binary_dir = os.path.join(self.data_dir, 'binary/videos', video_id)
        out_fname = os.path.join(binary_dir, "trainval_dataset.npy")
        os.makedirs(binary_dir, exist_ok=True)
        ret = load_data(processed_dir)
        # ret = load_adnerf_data(processed_dir)
        mel_name = os.path.join(processed_dir, 'mel.npy')
        mel = np.load(mel_name, allow_pickle=True)
        ret['mel'] = mel
        # ret.update(mel_dict)
        np.save(out_fname, ret, allow_pickle=True)



if __name__ == '__main__':
    binarizer = Binarizer()
    binarizer.parse(hparams['video_id'])
    print(f"Binarization for {hparams['video_id']} Done!")
