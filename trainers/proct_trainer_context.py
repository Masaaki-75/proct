import os
import sys
sys.path.append('..')
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
from torch.utils.data import DataLoader
from trainers.basic_trainer import BasicTrainer, set_requires_grad
from datasets.lowlevel_ct_dataset import SimpleCTDataset, PHANTOM, RandomContextDataset, AAPM_DIR, DEEPL_DIR
from wrappers.basic_wrapper import BasicWrapper
from utilities.rec_loss import get_metrics, MSSSIMLoss, SSIMLoss
from utilities.shaping import rescale_array
from utilities.misc import ensure_tuple_rep
import einops as E
from itertools import chain, cycle
from PIL import Image


class ProCTTrainerContext(BasicTrainer):
    def __init__(self, opt=None, net=None, unsqueeze=True):
        super().__init__()
        assert opt is not None and net is not None
        self.net:BasicWrapper = net
        self.opt = opt
        self.multigpu = torch.cuda.device_count() > 1
        self.unsqueeze = unsqueeze
        self.img_size = ensure_tuple_rep(opt.img_size, opt.spatial_dims)
        
        self.prepare_dataset()
        self.prepare_log()
        self.prepare_ct_tasks()
        self.prepare_criteria()

    def prepare_dataset(self):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        print('Loading dataset: ', dataset_name)
        if dataset_name == 'deeplesion':
            dataset_root_dir = DEEPL_DIR
        elif dataset_name == 'aapm':
            dataset_root_dir = AAPM_DIR
        else:
            raise NotImplementedError(f'Dataset not implemented: {dataset_name}.')
        
        if not os.path.exists(dataset_root_dir):
            raise ValueError(
                f"Invalid root path for dataset {dataset_name}: {dataset_root_dir}." + \
                    "Please check if it exists (in datasets/lowlevel_ct_dataset.py)")
        
        self.train_dataset = SimpleCTDataset(
            root_dir=dataset_root_dir, img_list_info=opt.train_json, img_shape=self.img_size, mode='train', 
            num_train=opt.num_train, min_hu=opt.min_hu, max_hu=opt.max_hu, clip_hu=opt.clip_hu)
        self.val_dataset = SimpleCTDataset(
            root_dir=dataset_root_dir, img_list_info=opt.val_json, img_shape=self.img_size, mode='val', 
            num_val=opt.num_val, min_hu=opt.min_hu, max_hu=opt.max_hu, clip_hu=opt.clip_hu)
        self.context_dataset = RandomContextDataset(
            root_dir=DEEPL_DIR, img_list_info=opt.train_json, img_shape=self.img_size, mode='train', 
            num_train=opt.num_train, min_hu=opt.min_hu, max_hu=opt.max_hu, clip_hu=opt.clip_hu)
        self.phantom = self.prepare_phantom(PHANTOM)
    
    def prepare_phantom(self, phantom, remove_bone=True):
        if remove_bone:
            phantom[phantom == 1.] = 0.
        
        hu_range = (self.train_dataset.min_hu, self.train_dataset.max_hu)
        hu = rescale_array(phantom, hu_range, old_range=(0, 1))
        if hu.shape != self.train_dataset.img_shape:
            hu = self.train_dataset._resize_image(hu)
        
        hu = self.train_dataset.clip_range(hu)
        mu = self.train_dataset.normalize_hu(torch.from_numpy(hu))
        return mu.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    def prepare_log(self,):
        opt = self.opt
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)
        if opt.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_root = opt.wandb_root if opt.tensorboard_root == '' else opt.tensorboard_root
            tensorboard_dir = opt.wandb_dir if opt.tensorboard_dir == '' else opt.tensorboard_dir
            self.writer = SummaryWriter(os.path.join(tensorboard_root, tensorboard_dir))
            
        if opt.use_wandb:
            if opt.local_rank == 0 or not self.multigpu:  # only on main process
                self.wandb_init2(self.opt)
        self.itlog_intv = opt.log_interval
        self.txt_path = self.checkpoint_path + '.txt'
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.write_txt_file('---------' + str(datetime.now()) + '---------', 'a')
        self.write_txt_file(opt.checkpoint_dir, 'a')
        print('Text path created: ', self.txt_path)
    
    def prepare_criteria(self):
        opt = self.opt
        self.ssim_criterion = SSIMLoss(val_range=1)
        self.msssim_criterion = MSSSIMLoss(val_range=1)
    
    def prepare_ct_tasks(self):
        task_list = [_.lower() for _ in self.opt.ct_task_list]
        self.task_kwargs = dict()
        
        for task in task_list:
            has_sv = 'sparse_view_' in task
            has_la = 'limited_angle_' in task
            
            if has_sv:
                self.task_kwargs.update({'sparse_view': sorted([int(n) for n in task.split('_')[2:]])})
            if has_la:
                self.task_kwargs.update({'limited_angle': [(0, int(n)) for n in task.split('_')[2:]]})
                
        if len(self.task_kwargs.keys()) == 0:
            raise NotImplementedError(f'Unsupported task: {task_list}')
            
        print('Task params:', self.task_kwargs)
    
    ##################### Functions for using CT Tools #####################
    def get_sino_mask(self, task_param, task='sparse_view'):
        net_module = self.net.module if self.multigpu else self.net
        if task == 'sparse_view':
            sinogram_mask = net_module.get_sparse_view_indices_from_full_angles(task_param, return_mask=True)
        elif task == 'limited_angle':
            sinogram_mask = net_module.get_limited_angle_indices_from_full_angles(task_param, return_mask=True)
        # sinogram mask shape: [1, 1, Nv, Nw]
        return sinogram_mask
    
    def mask2prompt(self, sino_mask):
        return 1. - sino_mask[:, :, :, 0]  # 1 for masked view, 0 for existed view
    
    
    ##################### Functions for basic trainer #####################
    def write_txt_file(self, string, write_mode='a', txt_path=None, add_linebreak=True):
        txt_path = self.txt_path if txt_path is None else txt_path
        string = string + '\n' if add_linebreak else string
        with open(self.txt_path, write_mode) as f:
            f.write(string)
    
    def get_fused_loss(self, preds, labels, reduction='mean', labels_sino=None):
        loss_type_list = [_.lower() for _ in self.opt.loss_type_list]
        loss = 0.0
        if 'l1' in loss_type_list:
            loss = loss + F.l1_loss(preds, labels, reduction=reduction)
        if 'sml1' in loss_type_list:
            loss = loss + F.smooth_l1_loss(preds, labels, reduction=reduction)
        if 'l2' in loss_type_list:
            loss = loss + F.mse_loss(preds, labels, reduction=reduction)
        if 'ssim' in loss_type_list:
            ssim_loss = self.ssim_criterion(preds, labels)
            loss = loss + self.opt.loss2_factor * ssim_loss
        if 'msssim' in loss_type_list:
            ssim_loss = self.msssim_criterion(preds, labels)
            loss = loss + self.opt.loss2_factor * ssim_loss
        return loss
    
    def fit(self,):
        opt = self.opt
        self.fix_seed(3407)
        if self.multigpu:
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            torch.cuda.set_device(opt.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.device = torch.device('cuda', opt.local_rank)
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        strings = f"""
        Summary:
            Number of Epochs:      {opt.epochs}
            Warmup steps:          {opt.warmup_steps}
            Batch Size:            {opt.batch_size}
            Initial Learning rate: {opt.lr}
            Loss Type:             {opt.loss_type_list}
            Training Size:         {len(self.train_dataset)}
            Validation Size:       {len(self.val_dataset)}
            Checkpoints Saved:     {opt.checkpoint_dir}
            Checkpoints Loaded:    {opt.net_checkpath}
            Device:                {self.device}
        """
        print(strings)
        self.write_txt_file(strings, 'a')
        
        # Loading checkpoint
        self.net = self.net.to(self.device)

        if opt.load_net:
            self.load_net()
            print('Network loaded!')
        
        self.optimizer = self.get_optimizer(self.net, lr=opt.lr)
        self.scheduler = self.get_scheduler(self.optimizer)
        if opt.load_opt:
            self.load_opt()
        
        batch_size = opt.batch_size
        support_size = opt.support_size  # repeat to batch size
            
        if self.multigpu:
            self.net = nn.parallel.DistributedDataParallel(
                self.net, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = DistributedSampler(self.val_dataset)
            self.context_sampler = DistributedSampler(self.context_dataset)
            self.train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, num_workers=opt.num_workers, sampler=self.train_sampler, pin_memory=True, drop_last=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=opt.num_workers, sampler=self.val_sampler,)
            self.context_loader = DataLoader(self.context_dataset, batch_size=support_size, num_workers=opt.num_workers, sampler=self.context_sampler, pin_memory=True)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=opt.num_workers)
            self.context_loader = DataLoader(self.context_dataset, batch_size=support_size, num_workers=opt.num_workers)
        
        # start training
        start_epoch = self.epoch
        print(f'Starting from epoch {start_epoch} (iter {self.iter}), with {opt.epochs} epochs to run.')
        for self.epoch in range(start_epoch, opt.epochs):
            print('start training epoch:{}'.format(self.epoch))
            if self.multigpu:
                self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.val()
            if self.scheduler is not None:
                self.scheduler.step()
                
            epoch_condition = ((self.epoch+1) % opt.save_epochs == 0) or ((self.epoch+1) == opt.epochs)
            
            if (opt.local_rank == 0 or not self.multigpu):
                self.save_net(save_latest=True)  # restore the latest result anyway
                self.save_opt(save_latest=True)
                if epoch_condition:
                    self.save_net()
                    if not opt.save_net_only:
                        self.save_opt()
    
    @staticmethod
    def get_loss_weight(task, task_param, return_task_ids=False):
        task_ids = [None, None]
        if task == 'sparse_view':
            task_ids[0] = 0
            if task_param < 18:
                w = 1.
                task_ids[1] = 0
            elif 18 <= task_param < 36:
                w = (task_param - 18) / (36 - 18) * (1.6 - 1) + 1.
                task_ids[1] = 1
            elif 36 <= task_param < 72:
                w = (task_param - 36) / (72 - 36) * (1.9 - 1.6) + 1.6
                task_ids[1] = 2
            elif 72 <= task_param < 144:
                w = (task_param - 72) / (144 - 72) * (2. - 1.9) + 1.9
                task_ids[1] = 3
            elif task_param >= 144:
                w = 2.
                task_ids[1] = 4
                
        else:  # limited_angle
            task_ids[0] = 1
            r = max(task_param) - min(task_param)
            if r < 60: 
                w = 1.
                task_ids[1] = 0
            elif 60 <= r < 90:
                w = (r - 60) / (90 - 60) * (1.6 - 1) + 1.
                task_ids[1] = 1
            elif 90 <= r < 120:
                w = (r - 90) / (120 - 90) * (1.9 - 1.6) + 1.6
                task_ids[1] = 2
            elif 120 <= r < 150:
                w = (r - 120) / (150 - 120) * (2. - 1.9) + 1.9
                task_ids[1] = 3
            else:
                w = 2. 
                task_ids[1] = 4
            w /= 2.
        
        return w if not return_task_ids else (w, task_ids)
    
    def forward_net(self, images, context, cond, task_ids):
        net_module = self.net.module if self.multigpu else self.net
        cond_dim = net_module.cond_dim
        prompt = net_module.prompt
        if cond_dim > 0:
            if prompt is not None:
                preds = self.net(images, context=context, cond=cond, task_ids=task_ids)
            else:
                preds = self.net(images, context=context, cond=cond)
        else:
            preds = self.net(images, context=context)
        
        return preds
    
    @torch.no_grad()
    def prepare_ct_batch(self, source: torch.Tensor, context=None):
        batch_size = source.shape[0]  # could be different from opt.batch_size, since with drop_last
        
        if context is None:
            assert self.opt.use_phantom
            context = self.phantom.repeat(batch_size, 1, 1, 1)
        else:
            context = context.repeat(batch_size, 1, 1, 1)
        return torch.cat((source, context), dim=0).to(self.device, non_blocking=True)
    
    @torch.no_grad()
    def get_input_target_pair(self, srcs_w_cons:torch.Tensor, task_param=None, task='sparse_view', prob_flip=0):
        eff_batch_size = srcs_w_cons.shape[0]
        supp_size = self.opt.support_size
        batch_size = eff_batch_size // (supp_size + 1)
        batch_size_supp = eff_batch_size - batch_size
        
        if (prob_flip > 0) and (np.random.rand() <= prob_flip):
            srcs_w_cons = torch.flip(srcs_w_cons, dims=[-1])  #mus[:, :, :, ::-1]
        
        if self.multigpu:
            lqs, gts = self.net.module.generate_input_target_mu(srcs_w_cons, task_param, task=task, return_sinogram=False)
        else:
            lqs, gts = self.net.generate_input_target_mu(srcs_w_cons, task_param, task=task, return_sinogram=False)
        
        lq_src, gt_src = lqs[:batch_size], gts[:batch_size]  # [B, 1, H, W]
        lq_con, gt_con = lqs[-batch_size_supp:], gts[-batch_size_supp:]
        #target_data_sino = sinograms[:batch_size]
        if self.unsqueeze:
            lq_con = E.rearrange(lq_con, "(b s) c h w -> b s c h w", b=batch_size)
            gt_con = E.rearrange(gt_con, "(b s) c h w -> b s c h w", b=batch_size)
            
        context = torch.cat([lq_con, gt_con], dim=-3) # [B, S, 2*C, H, W]
            
        sinogram_mask = self.get_sino_mask(task_param, task=task)
        prompt = self.mask2prompt(sinogram_mask)
        prompt = prompt.float().repeat_interleave(batch_size, dim=0).to(lq_src.device)
        
        return lq_src, gt_src, prompt, context
    
    
    def train(self,):
        self.net = self.net.train()
        opt = self.opt
        psnrs, ssims = [], []
        svct_params = [int(_) for _ in range(9, 288+1)]
        lact_params = [int(_) for _ in range(60, 180+1)]
        
        pbar = tqdm.tqdm(self.train_loader, leave=False, ncols=100) if opt.use_tqdm else self.train_loader
        for i, (sources) in enumerate(pbar):  # one batch contain different images
            contexts = next(iter(self.context_loader))
            for task, task_param_list in self.task_kwargs.items():
                if self.epoch > opt.warmup_steps:
                    if task == 'sparse_view':
                        task_param = np.random.choice(svct_params)
                        if task_param % 2 and task_param > 144:  # deal with torch-radon bug, probability some division issues
                            task_param -= 1
                    else:
                        task_param = (0, np.random.choice(lact_params))
                else:
                    rand_task_index = np.random.choice(len(task_param_list))
                    task_param = task_param_list[rand_task_index]  # np.random.choice cannot deal with list of tuples
                
                srcs_w_cons = self.prepare_ct_batch(sources, contexts)
                images, labels, prompt, context = self.get_input_target_pair(srcs_w_cons, task_param, task=task, prob_flip=opt.prob_flip)
                w, task_ids = self.get_loss_weight(task=task, task_param=task_param, return_task_ids=True)
                w = 1.0 if (not opt.loss_weight) else w
                preds = self.forward_net(images, context, prompt, task_ids)
                
                del context
                loss = self.get_fused_loss(preds, labels, reduction='mean') * w  # [BUG]: check here
                self.optimizer.zero_grad()
                loss.backward()
                if opt.clip_grad > 0:
                    nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                
                rmse, psnr, ssim = self.get_metrics_by_window(preds, labels)
                psnrs.append(psnr)
                ssims.append(ssim)

                if opt.use_tqdm:
                    pbar.set_postfix({'loss': '%.2f' % (loss.item()), 'psnr': '%.2f' % psnr})
                    pbar.update(1)

                # log acc by iteration
                if opt.local_rank == 0 or not self.multigpu:
                    if self.iter !=0 and self.iter % self.itlog_intv == 0:
                        optimizer = self.optimizer
                        log_info = {
                            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                            'psnr': np.mean(psnrs[-self.itlog_intv:]),
                            'ssim': np.mean(ssims[-self.itlog_intv:])
                        }
                        
                        self.write_txt_file(str(log_info))
                        if opt.use_tensorboard:
                            self.tensorboard_scalar(self.writer, 'train/loss', self.iter, **log_info)
                        if opt.use_wandb:
                            self.wandb_logger('train/iter' ,**log_info)
                    self.iter += 1

        # epoch info
        if opt.local_rank == 0 or not self.multigpu:
            epoch_log = {
                'psnr': np.mean(psnrs),
                'ssim': np.mean(ssims)
            }
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            strings = f"""
            Epoch {self.epoch} learning rate: {current_lr}
            Epoch {self.epoch} train psnr: {epoch_log["psnr"]}
            Epoch {self.epoch} train ssim: {epoch_log["ssim"]}
            """
            self.write_txt_file(strings, 'a')

            if self.opt.use_tensorboard:
                self.tensorboard_scalar(self.writer, 'train/epoch', self.epoch, **epoch_log)
                self.tensorboard_scalar(self.writer, 'settings', self.epoch, **{'current_lr':current_lr})
            if self.opt.use_wandb:
                self.wandb_logger('train/epoch', step_name='epoch', step=self.epoch, **epoch_log)
                self.wandb_logger('settings', step_name='epoch', step=self.epoch, **{'epoch':self.epoch, 'current_lr':current_lr})

    @torch.no_grad()
    def val(self,):
        self.net = self.net.eval()
        task_kwargs = {
            'sparse_view': [18, 36, 72, 144],
            'limited_angle': [(0,90), (0,120), (0,150)]
        }
        
        for task, task_param_list in task_kwargs.items():
            for task_index, task_param in enumerate(task_param_list):
                psnrs, ssims = [], []
                pbar = tqdm.tqdm(self.val_loader, leave=False, ncols=100) if self.opt.use_tqdm else self.val_loader
                for _, (sources) in enumerate(pbar):  # one batch contain different images
                    contexts = next(iter(self.context_loader))
                    srcs_w_cons = self.prepare_ct_batch(sources, contexts)
                    images, labels, prompt, context = self.get_input_target_pair(srcs_w_cons, task_param, task=task)
                
                    _, task_ids = self.get_loss_weight(task=task, task_param=task_param, return_task_ids=True)
                    preds = self.forward_net(images, context, prompt, task_ids)
                    loss = F.mse_loss(preds, labels)  # save validation time
                    
                    rmse, psnr, ssim = self.get_metrics_by_window(preds, labels)
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    
                    if self.opt.use_tqdm:
                        pbar.set_postfix({'loss': '%.3f' % (loss.item()), 'psnr': '%.3f' % psnr})
                        pbar.update(1)
                    
                if self.opt.local_rank == 0 or not self.multigpu:
                    epoch_log = {
                        f'psnr_{task_param}': np.mean(psnrs),
                        f'ssim_{task_param}': np.mean(ssims)
                    }
                    
                    strings = f"""
                    Epoch {self.epoch} val psnr: {epoch_log[f'psnr_{task_param}']}
                    Epoch {self.epoch} val ssim: {epoch_log[f'ssim_{task_param}']}
                    """
                    self.write_txt_file(strings, 'a')

                    print(strings)
                    if self.opt.use_tensorboard:
                        self.tensorboard_scalar(self.writer, 'val/epoch', self.epoch, **epoch_log)
                    if self.opt.use_wandb:
                        self.wandb_logger('val/epoch', step_name='epoch', step=self.epoch, **epoch_log)
    
    @torch.no_grad()
    def get_metrics_by_window(self, preds, labels, to_hu=True):
        if to_hu:
            preds = self.train_dataset.window_transform(self.train_dataset.denormalize_hu(preds))
            labels = self.train_dataset.window_transform(self.train_dataset.denormalize_hu(labels))
        rmse, psnr, ssim = get_metrics(preds, labels, val_range=1)
        return rmse, psnr, ssim

