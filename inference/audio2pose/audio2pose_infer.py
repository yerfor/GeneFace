import os
import torch
import librosa
import numpy as np
import importlib
import tqdm

from utils.commons.tensor_utils import move_to_cuda
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.commons.hparams import hparams, set_hparams
from utils.commons.euler2rot import euler_trans_2_c2w

from tasks.audio2pose.dataset_utils import Audio2PoseDataset


class Audio2PoseInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.infer_max_length = hparams.get('infer_max_length', 500000)
        self.device = device
        self.audio2pose_task = self.build_audio2pose_task()
        self.audio2pose_task.eval()
        self.audio2pose_task.to(self.device)
        dataset = Audio2PoseDataset()
        self.mean_trans = dataset.mean_trans.unsqueeze(0).numpy()
        self.init_pose = torch.cat([dataset.euler_lst[0], dataset.trans_lst[0]], dim=0).numpy()

    def build_audio2pose_task(self):
        assert hparams['task_cls'] != ''
        pkg = ".".join(hparams["task_cls"].split(".")[:-1])
        cls_name = hparams["task_cls"].split(".")[-1]
        task_cls = getattr(importlib.import_module(pkg), cls_name)
        task = task_cls()
        task.build_model()
        task.eval()
        # steps = hparams.get('infer_ckpt_steps', 5000)
        steps = None
        load_ckpt(task.model, hparams['work_dir'], 'model', steps=steps)
        ckpt, _ = get_last_checkpoint(hparams['work_dir'], steps=steps)
        task.global_step = ckpt['global_step']
        return task

    def infer_once(self, inp):
        self.inp = inp
        samples = self.get_cond_from_input(inp)
        out_name = self.forward_system(samples, inp)
        print(f"The predicted 3D landmark sequence is saved at {out_name}")

    def get_cond_from_input(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a list that contains the condition feature of NeRF
        """

        self.save_wav16k(inp)
        # from data_gen.process_lrs3.process_audio_hubert import get_hubert_from_16k_wav
        # hubert = get_hubert_from_16k_wav(self.wav16k_name).detach().numpy()
        # len_mel = hubert.shape[0]
        # x_multiply = 8
        # if len_mel % x_multiply == 0:
        #     num_to_pad = 0
        # else:
        #     num_to_pad = x_multiply - len_mel % x_multiply
        # hubert = np.pad(hubert, pad_width=((0,num_to_pad), (0,0))) # [t_x, 1024]
        # t_x = hubert.shape[0]
        # hubert = hubert.reshape([t_x//2, 1024*2])
        # sample = {
        #     'hubert': torch.from_numpy(hubert).float().unsqueeze(0), # [1, T, 2048]
        #     }

        # load the deepspeech features as the condition for lm3d torso nerf
        wav16k_name = self.wav16k_name
        deepspeech_name = wav16k_name[:-4] + '_deepspeech.npy'
        if not os.path.exists(deepspeech_name):
            print(f"Try to extract deepspeech from {wav16k_name}...")
            # deepspeech_python = '/home/yezhenhui/anaconda3/envs/geneface/bin/python' # the path of your python interpreter that has installed DeepSpeech
            # extract_deepspeech_cmd = f'{deepspeech_python} data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
            extract_deepspeech_cmd = f'python data_util/deepspeech_features/extract_ds_features.py --input={wav16k_name} --output={deepspeech_name}'
            os.system(extract_deepspeech_cmd)
            print(f"Saved deepspeech features of {wav16k_name} to {deepspeech_name}.")
        else:
            print(f"Try to load pre-extracted deepspeech from {deepspeech_name}...")
        deepspeech_arr = np.load(deepspeech_name) # [T, w=16, c=29]
        print(f"Loaded deepspeech features from {deepspeech_name}.")
        # get window condition of deepspeech
        sample = {}
        # sample['deepspeech'] = torch.from_numpy(deepspeech_arr).float().reshape([-1, 16*29])
        sample['deepspeech'] = torch.from_numpy(deepspeech_arr[:, 7:9,:]).float().reshape([-1, 2*29])
        return [sample]

    def forward_system(self, batches, inp):
        out_dir = self._forward_audio2pose_task(batches, inp)
        return out_dir

    def _forward_audio2pose_task(self, batches, inp):
        with torch.no_grad():
            pred_lst = []            
            for idx, batch in tqdm.tqdm(enumerate(batches), total=len(batches),
                                desc=f"Now Audio2Pose model is predicting the head pose (camera2world matrix) into {inp['out_npy_name']}"):
                if self.device == 'cuda':
                    batch = move_to_cuda(batch)

                # smo_pred_pose = self.audio2pose_task.model.autoregressive_infer(batch['hubert'].squeeze(), self.init_pose)
                smo_pred_pose = self.audio2pose_task.model.autoregressive_infer(batch['deepspeech'].squeeze(), self.init_pose)
                smo_pred_pose = smo_pred_pose.squeeze().cpu().numpy()
                euler, trans = smo_pred_pose[:,:3], smo_pred_pose[:,3:6]
                trans = trans + self.mean_trans
                c2w = euler_trans_2_c2w(euler, trans).numpy()
                pred_lst.append(c2w)
        np.save(inp['out_npy_name'], pred_lst)
        return inp['out_npy_name']

    @classmethod
    def example_run(cls, inp=None):
        inp_tmp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_npy_name': 'infer_outs/May/pred_c2w/zozo.npy'
            }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp
        if hparams.get("infer_audio_source_name", '') != '':
            inp['audio_source_name'] = hparams['infer_audio_source_name'] 
        if hparams.get("infer_out_npy_name", '') != '':
            inp['out_npy_name'] = hparams['infer_out_npy_name']
        out_dir = os.path.dirname(inp['out_npy_name'])

        os.makedirs(out_dir, exist_ok=True)
        infer_ins = cls(hparams)
        infer_ins.infer_once(inp)

    ##############
    # IO-related
    ##############
    def save_wav16k(self, inp):
        source_name = inp['audio_source_name']
        supported_types = ('.wav', '.mp3', '.mp4', '.avi')
        assert source_name.endswith(supported_types), f"Now we only support {','.join(supported_types)} as audio source!"
        wav16k_name = source_name[:-4] + '_16k.wav'
        self.wav16k_name = wav16k_name
        extract_wav_cmd = f"ffmpeg -i {source_name} -f wav -ar 16000 -v quiet {wav16k_name} -y"
        os.system(extract_wav_cmd)
        print(f"Extracted wav file (16khz) from {source_name} to {wav16k_name}.")

if __name__ == '__main__':
    set_hparams()
    inp = {
            'audio_source_name': 'data/raw/val_wavs/zozo.wav',
            'out_npy_name': 'infer_outs/May/pred_c2w/zozo.npy',
            }
    Audio2PoseInfer.example_run(inp)