import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import lpips

from modules.radnerfs.models import RADNeRF

from utils.commons.image_utils import to8b
from utils.commons.base_task import BaseTask
from utils.commons.dataset_utils import data_loader
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars, convert_to_np, move_to_cuda
from utils.nn.model_utils import print_arch, num_params
from utils.nn.schedulers import ExponentialScheduleForRADNeRF
from utils.nn.grad import get_grad_norm

from tasks.radnerfs.dataset_utils import RADNeRFDataset


class RADNeRFTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = RADNeRFDataset
        self.train_dataset = self.dataset_cls(prefix='train', training=True)
        self.val_dataset = self.dataset_cls(prefix='val', training=False)

        self.criterion_lpips = lpips.LPIPS(net='alex')
        self.finetune_lip_flag = False

    def build_model(self):
        self.model = RADNeRF(hparams)
        self.embedders_params = []
        self.embedders_params += [p for k, p in self.model.named_parameters() if p.requires_grad and 'position_embedder' in k]
        self.embedders_params += [p for k, p in self.model.named_parameters() if p.requires_grad and 'ambient_embedder' in k]
        self.network_params = [p for k, p in self.model.named_parameters() if (p.requires_grad and 'position_embedder' not in k and 'ambient_embedder' not in k and 'cond_att_net' not in k)]
        self.att_net_params = [p for k, p in self.model.named_parameters() if p.requires_grad and 'cond_att_net' in k]
        
        self.model.conds = self.train_dataset.conds
        self.model.mark_untrained_grid(self.train_dataset.poses, self.train_dataset.intrinsics)

        return self.model

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
            
    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.network_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            eps=1e-15)
        self.optimizer.add_param_group({
            'params': self.embedders_params,
            'lr': hparams['lr'] * 10,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })
        self.optimizer.add_param_group({
            'params': self.att_net_params,
            'lr': hparams['lr'] * 5,
            'betas': (hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            'eps': 1e-15
        })
        return self.optimizer

    def build_scheduler(self, optimizer):
        return ExponentialScheduleForRADNeRF(optimizer, hparams['lr'], hparams['warmup_updates'])

    @data_loader
    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(self.train_dataset,collate_fn=self.train_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.train_dl

    @data_loader
    def val_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=True, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.val_dl

    @data_loader
    def test_dataloader(self):
        self.val_dl = torch.utils.data.DataLoader(self.val_dataset,collate_fn=self.val_dataset.collater,
                                            batch_size=1, shuffle=False, 
                                            # num_workers=0, pin_memory=True)
                                            num_workers=0, pin_memory=False)
        return self.val_dl
        
    ##########################
    # forward the model
    ##########################
    def run_model(self, sample, infer=False):
        """
        render or train on a single-frame
        :param sample: a batch of data
        :param infer: bool, run in infer mode
        :return:
            if not infer:
                return losses, model_out
            if infer:
                return model_out
        """
        cond = sample['cond_win'] if hparams['use_window_cond'] else sample['cond']
        cond_wins = sample['cond_wins']
        rays_o = sample['rays_o'] # [B, N, 3]
        rays_d = sample['rays_d'] # [B, N, 3]
        bg_coords = sample['bg_coords'] # [1, N, 2]
        poses = sample['pose'] # [B, 6]
        face_mask = sample['face_mask'] # [B, N]
        idx = sample['idx'] # [B]
        gt_rgb = sample['gt_img']
        bg_color = sample['bg_img']
        H, W = sample['H'], sample['W']

        cond_inp = cond_wins if hparams['with_att'] else cond
        start_finetune_lip = hparams['finetune_lips'] and self.global_step > hparams['finetune_lips_start_iter']

        model_out = self.model.render(rays_o, rays_d, cond_inp, bg_coords, poses, index=idx, staged=False, bg_color=bg_color, perturb=not infer, force_all_rays=infer, **hparams)
        pred_rgb = model_out['image']
        losses_out = {}
        losses_out['mse_loss'] = torch.mean((pred_rgb - gt_rgb) ** 2) # [B, N, 3] --> [B, N]

        if not infer:
            alphas = model_out['weights_sum'].clamp(1e-5, 1 - 1e-5)
            losses_out['weights_entropy_loss'] = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
            ambient = model_out['ambient'] # [N], abs sum
            losses_out['ambient_loss'] = (ambient * (~face_mask.view(-1))).mean()

        if infer or (start_finetune_lip and self.finetune_lip_flag):
            xmin, xmax, ymin, ymax = sample['lip_rect']
            if infer:
                # clip lip part from the whole image
                gt_rgb = gt_rgb.view(-1, H, W, 3)[:,xmin:xmax,ymin:ymax,:].permute(0, 3, 1, 2).contiguous()
                pred_rgb = pred_rgb.view(-1, H, W, 3)[:,xmin:xmax,ymin:ymax,:].permute(0, 3, 1, 2).contiguous()
            else:
                # during the training phase of finetuning lip, all rays are from lip part
                gt_rgb = gt_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous()
                pred_rgb = pred_rgb.view(-1, xmax - xmin, ymax - ymin, 3).permute(0, 3, 1, 2).contiguous()
            losses_out['lpips_loss'] = self.criterion_lpips(pred_rgb, gt_rgb).mean()
        
        if start_finetune_lip and not infer:
            # during training, flip in each iteration, to prevent forgetting other facial parts.
            self.finetune_lip_flag = not self.finetune_lip_flag
            self.train_dataset.finetune_lip_flag = self.finetune_lip_flag
        return losses_out, model_out

    ##########################
    # training 
    ##########################
    def _training_step(self, sample, batch_idx, optimizer_idx):
        outputs = {}
        self.train_dataset.global_step = self.global_step
        if self.global_step % hparams['update_extra_interval'] == 0:
            start_finetune_lips = hparams['finetune_lips'] and self.global_step > hparams['finetune_lips_start_iter']
            if not start_finetune_lips:
                # when finetuning lips, we don't update the density grid and bitfield.
                self.model.update_extra_state()

        loss_output, model_out = self.run_model(sample)
        loss_weights = {
            'mse_loss': 1.0,
            'weights_entropy_loss': hparams['lambda_weights_entropy'],
            'lpips_loss': hparams['lambda_lpips_loss'],
            'ambient_loss': min(self.global_step / 250000, 1.0) * hparams['lambda_ambient'], # gradually increase it
        }
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x.device)
        loss_output['head_psnr'] = mse2psnr(loss_output['mse_loss'].detach())
        outputs.update(loss_output)

        if self.global_step % hparams['tb_log_interval'] == 0:
            density_grid_info = {
                "density_grid_info/min_density": self.model.density_grid.min().item(),
                "density_grid_info/max_density": self.model.density_grid.max().item(),
                "density_grid_info/mean_density": self.model.mean_density,
                # "density_grid_info/occupancy_rate": (self.model.density_grid > 0.01).sum() / (128**3 * self.model.cascade), 
                "density_grid_info/occupancy_rate": (self.density_grid > min(self.model.mean_density, self.model.density_thresh)).sum() / (128**3 * self.cascade), 
                "density_grid_info/step_mean_count": self.model.mean_count
            }
            outputs.update(density_grid_info)
        return total_loss, outputs
    
    def on_before_optimization(self, opt_idx):
        prefix = f"grad_norm_opt_idx_{opt_idx}"
        grad_norm_dict = {
            f'{prefix}/cond_att': get_grad_norm(self.att_net_params),
            f'{prefix}/embedders_params': get_grad_norm(self.embedders_params),
            f'{prefix}/network_params': get_grad_norm(self.network_params ),
        }
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
        return grad_norm_dict
        
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    #####################
    # Validation
    #####################
    def validation_start(self):
        if self.global_step % hparams['valid_infer_interval'] == 0:
            self.gen_dir = os.path.join(hparams['work_dir'], f'validation_results/validation_{self.trainer.global_step}')
            os.makedirs(self.gen_dir, exist_ok=True)
            os.makedirs(f'{self.gen_dir}/images', exist_ok=True)
            os.makedirs(f'{self.gen_dir}/depth', exist_ok=True)

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = 1
        outputs = tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            num_val_samples = len(self.val_dataset)
            interval = (num_val_samples-1) // 4
            idx_lst = [i * interval for i in range(5)]
            sample = move_to_cuda(self.val_dataset[idx_lst[batch_idx]])
            _, infer_outputs = self.run_model(sample, infer=True)
            rgb_pred = infer_outputs['image']
            depth_pred = infer_outputs['depth']
            H, W = sample['H'], sample['W']
            img_pred = rgb_pred.reshape([H, W, 3])
            depth_pred = depth_pred.reshape([H, W])
            gen_dir = self.gen_dir
            base_fn = f"frame_{sample['idx']}"
            self.save_rgb_to_fname(img_pred, f"{gen_dir}/images/{base_fn}.png")
            self.save_rgb_to_fname(depth_pred, f"{gen_dir}/depth/{base_fn}.png")
            target = sample['gt_img']
            img_gt = target.reshape([H, W, 3])
            if hparams['save_gt']:
                base_fn = f"frame_{sample['idx']}_gt"
                self.save_rgb_to_fname(img_gt, f"{gen_dir}/images/{base_fn}.png")
        return outputs

    def validation_end(self, outputs):
        return super().validation_end(outputs)

    #####################
    # Testing
    #####################
    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/images', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/depth', exist_ok=True)

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        outputs = self.run_model(sample, infer=True)
        rgb_pred = outputs['rgb_map']
        H, W = sample['H'], sample['W']
        img_pred = rgb_pred.reshape([H, W, 3])
        gen_dir = self.gen_dir
        base_fn = f"frame_{sample['idx']}"
        self.save_rgb_to_fname(img_pred, f"{gen_dir}/images/{base_fn}.png")
        self.save_rgb_to_fname(img_pred, f"{gen_dir}/depth/{base_fn}.png")
        target = sample['gt_img']
        img_gt = target.reshape([H, W, 3])
        if hparams['save_gt']:
            base_fn = f"frame_{sample['idx']}_gt"
            self.save_rgb_to_fname(img_gt, f"{gen_dir}/images/{base_fn}.png")
            
        outputs['losses'] = (img_gt - img_pred).mean()
        return outputs

    def test_end(self, outputs):
        pass

    #####################
    # Visualization utils
    #####################
    @staticmethod
    def save_rgb_to_fname(rgb, fname):
        rgb = convert_to_np(rgb * 255.).astype(np.uint8)
        if rgb.ndim == 3:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{fname}", bgr)
        else:
            # gray image
            cv2.imwrite(f"{fname}", rgb)

