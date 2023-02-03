import os
import numpy as np
import torch
from scipy.io import loadmat


class Face3DHelper:
    def __init__(self, bfm_dir='data_util/BFM_models', use_gpu=True):
        self.bfm_dir = bfm_dir
        self.device = 'cuda' if use_gpu else 'cpu'
        self.load_3dmm()

    def load_3dmm(self):
        model = loadmat(os.path.join(self.bfm_dir, "BFM_model_front.mat"))
        self.mean_shape = torch.from_numpy(model['meanshape'].transpose()).float().to(self.device) # mean face shape. [3*N, 1], N=35709, xyz=3, ==> 3*N=107127
        self.id_base = torch.from_numpy(model['idBase']).float().to(self.device) # identity basis. [3*N,80], we have 80 eigen faces for identity
        self.exp_base = torch.from_numpy(model['exBase']).float().to(self.device) # expression basis. [3*N,64], we have 64 eigen faces for expression

        self.mean_texure = torch.from_numpy(model['meantex'].transpose()).float().to(self.device) # mean face texture. [3*N,1] (0-255)
        self.tex_base = torch.from_numpy(model['texBase']).float().to(self.device) # texture basis. [3*N,80], rgb=3

        self.point_buf = torch.from_numpy(model['point_buf']).float().to(self.device) # triangle indices for each vertex that lies in. starts from 1. [N,8] (1-F)
        self.face_buf = torch.from_numpy(model['tri']).float().to(self.device) # vertex indices in each triangle. starts from 1. [F,3] (1-N)
        self.key_points = torch.from_numpy(model['keypoints'].squeeze().astype(np.long)).long().to(self.device) # vertex indices of 68 facial landmarks. starts from 1. [68,1]

        self.key_mean_shape = self.mean_shape.reshape([-1,3])[self.key_points,:].to(self.device)
        self.key_id_base = self.id_base.reshape([-1,3,80])[self.key_points, :, :].reshape([-1,80]).to(self.device)
        self.key_exp_base = self.exp_base.reshape([-1,3,64])[self.key_points, :, :].reshape([-1,64]).to(self.device)

    def split_coeff(self, coeff):
        """
        coeff: Tensor[B, T, c=257] or [T, c=257]
        """
        ret_dict = {
            'identity': coeff[..., :80],  # identity, [b, t, c=80] 
            'expression': coeff[..., 80:144],  # expression, [b, t, c=80]
            'texture': coeff[..., 144:224],  # texture, [b, t, c=80]
            'angles': coeff[..., 224:227],  # euler angles for pose, [b, t, c=3]
            'translation':  coeff[..., 254:257], # translation, [b, t, c=3]
            'gamma': coeff[..., 227:254] # lighting, [b, t, c=27]
        }
        return ret_dict
    
    def reconstruct_face3d(self, id_coeff, exp_coeff):
        """
        Generate a pose-independent 3D face mesh!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.device)
        exp_coeff = exp_coeff.to(self.device)
        mean_face = self.mean_shape.squeeze().reshape([1, -1]) # [3N, 1] ==> [1, 3N]
        id_base, exp_base = self.id_base, self.exp_base # [3*N, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3N] ==> [t,3N]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3N] ==> [t,3N]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        mean_xyz = self.mean_shape.squeeze().reshape([-1,3]).mean(dim=0) # [1, 3]
        face3d = face - mean_xyz.unsqueeze(0) # [t,N,3]
        return face3d

    def reconstruct_lm3d(self, id_coeff, exp_coeff):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.device)
        exp_coeff = exp_coeff.to(self.device)
        mean_face = self.key_mean_shape.squeeze().reshape([1, -1]) # [3*68, 1] ==> [1, 3*68]
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = mean_face + identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        # re-centering the face with mean_xyz, so the face will be in [-1, 1]
        mean_xyz = self.key_mean_shape.squeeze().reshape([-1,3]).mean(dim=0) # [1, 3]
        lm3d = face - mean_xyz.unsqueeze(0) # [t,N,3]
        return lm3d

    def reconstruct_idexp_lm3d(self, id_coeff, exp_coeff):
        """
        Generate 3D landmark with keypoint base!
        id_coeff: Tensor[T, c=80]
        exp_coeff: Tensor[T, c=64]
        """
        id_coeff = id_coeff.to(self.device)
        exp_coeff = exp_coeff.to(self.device)
        id_base, exp_base = self.key_id_base, self.key_exp_base # [3*68, C]
        identity_diff_face = torch.matmul(id_coeff, id_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        face = identity_diff_face + expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        lm3d = face * 10
        return lm3d

    def reconstruct_exp_lm3d(self, exp_coeff):
        """
        Generate Exp 3D landmark with keypoint base!
        exp_coeff: Tensor[T, c=64]
        """
        exp_coeff = exp_coeff.to(self.device)
        # mean_face = self.key_mean_shape.squeeze().reshape([1, -1]) # [3*68, 1] ==> [1, 3*68]
        exp_base = self.key_exp_base # [3*68, C]
        expression_diff_face = torch.matmul(exp_coeff, exp_base.transpose(0,1)) # [t,c],[c,3*68] ==> [t,3*68]
        
        # face = mean_face + expression_diff_face # [t,3N]
        face = expression_diff_face # [t,3N]
        face = face.reshape([face.shape[0], -1, 3]) # [t,N,3]
        lm3d = face * 10
        return lm3d

    def get_lm3d_from_face3d(self, face3d):
        """
        get the 3D coordinates of 68 keypoints in the face3D by simply indexing.
        face3d: [t, N, 3]
        """
        lm3d = face3d[:, self.key_points, :] # [t, 68, 3]
        return lm3d
    
    def get_lm3d_from_coeff_seq(self, coeff_arr):
        """
        coeff_arr: [T, 257]
        """
        ret_dict = self.split_coeff(coeff_arr)
        lm3d = self.reconstruct_lm3d(ret_dict['identity'], ret_dict['expression'])
        return lm3d

    def get_eye_mouth_lm_from_lm3d(self, lm3d):
        eye_lm = lm3d[:, 17:48] # [T, 31, 3]
        mouth_lm = lm3d[:, 48:68] # [T, 20, 3]
        return eye_lm, mouth_lm
    
    def get_eye_mouth_lm_from_lm3d_batch(self, lm3d):
        eye_lm = lm3d[:, :, 17:48] # [T, 31, 3]
        mouth_lm = lm3d[:, :, 48:68] # [T, 20, 3]
        return eye_lm, mouth_lm
    
    def get_lm3d_from_identity_exp(self, identity, exp_arr):
        """
        exp: [T, 64]
        identity: [80]
        """
        assert identity.ndim == 1 and exp_arr.ndim == 2
        T = exp_arr.shape[0]
        identity = identity[None, :].repeat([T, 1])
        lm3d = self.reconstruct_lm3d(identity, exp_arr)
        return lm3d

if __name__ == '__main__':
    import cv2
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    face3d_helper = Face3DHelper('data_util/BFM_models')
    coeff_npy = 'data/processed/videos/May/coeff.npy'
    coeff_dict = np.load(coeff_npy, allow_pickle=True).tolist()
    coeff = torch.from_numpy(coeff_dict['coeff']) # [-250:]
    lm3d = face3d_helper.reconstruct_idexp_lm3d(coeff[:, :80], coeff[:, 80:144])

    lm3d_mean = lm3d.mean(dim=0, keepdims=True)
    lm3d_std = lm3d.std(dim=0, keepdims=True)

    WH = 512
    lm3d = (lm3d * WH/2 + WH/2).cpu().int().numpy()
    eye_idx = list(range(36,48))
    mouth_idx = list(range(48,68))
    for i_img in range(len(lm3d)):
        lm2d = lm3d[i_img ,:, :2] # [68, 2]
        img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
        for i in range(len(lm2d)):
            x, y = lm2d[i]
            if i in eye_idx:
                color = (0,0,255)
            elif i in mouth_idx:
                color = (0,255,0)
            else:
                color = (255,0,0)
            img = cv2.circle(img, center=(x,y), radius=3, color=color, thickness=-1)
            img = cv2.putText(img, f"{i}", org=(x,y), fontFace=font, fontScale=0.3, color=(255,0,0))
        img = cv2.flip(img, 0)
        cv2.imwrite(f'infer_outs/tmp_imgs/{format(i_img, "05d")}.png', img)