import os
import sys
sys.path.append('..')
import csv
import tqdm
import torch
import warnings
import einops as E
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.utils.data import DataLoader
from trainers.basic_trainer import BasicTrainer
from datasets.lowlevel_ct_dataset import SimpleCTDataset, PHANTOM, DEEPL_DIR, AAPM_DIR
from wrappers.basic_wrapper import BasicWrapper
from collections import defaultdict
from utilities.rec_loss import get_metrics
from utilities.misc import ensure_tuple_rep
from utilities.shaping import rescale_array
from torchvision.utils import save_image as torchvision_save_image        


class SimpleTester(BasicTrainer):
    def __init__(self, opt=None, net=None, test_window=None, save_rmse=False, **kwargs):
        super().__init__()
        assert opt is not None and net is not None
        self.net = net
        self.opt = opt
        self.save_rmse = save_rmse
        self.batch_size = 1  # fixed batch_size for test loader
        self.net_name = opt.network
        self.img_size = ensure_tuple_rep(opt.img_size, opt.spatial_dims)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.prepare_dataset()
        self.prepare_ct_tasks()
        self.prepare_windows(test_window)
        self.prepare_log()
        self.fix_seed(seed=3407)

    def prepare_dataset(self,):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        self.dataset_name = dataset_name
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
        
        self.train_dataset = None
        self.test_dataset = SimpleCTDataset(
            root_dir=dataset_root_dir, img_list_info=opt.val_json, img_shape=self.img_size, mode='val', 
            num_train=opt.num_train, num_val=opt.num_val, min_hu=opt.min_hu, max_hu=opt.max_hu, clip_hu=opt.clip_hu)
        
        if 'proct' not in self.net_name:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=opt.num_workers)
        else:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=opt.num_workers)
            # train dataset to provide in-context pairs
            self.train_dataset = SimpleCTDataset(
                root_dir=DEEPL_DIR, img_list_info=opt.train_json, img_shape=self.img_size, mode='train', 
                num_train=opt.num_train, num_val=opt.num_val, min_hu=opt.min_hu, max_hu=opt.max_hu, clip_hu=opt.clip_hu)

        self.phantom = self.prepare_phantom(PHANTOM)
    
    def prepare_phantom(self, phantom, remove_bone=True):
        if remove_bone:
            phantom[phantom == 1.] = 0.
        
        hu_range = (self.test_dataset.min_hu, self.test_dataset.max_hu)
        hu = rescale_array(phantom, hu_range, old_range=(0, 1))
        if hu.shape != self.test_dataset.img_shape:
            hu = self.test_dataset._resize_image(hu)
        
        hu = self.test_dataset.clip_range(hu)
        mu = self.test_dataset.normalize_hu(torch.from_numpy(hu).float())
        return mu.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    def prepare_ct_tasks(self,):
        task_list = [_.lower() for _ in self.opt.ct_task_list]
        self.task_kwargs = dict()
        
        for task in task_list:
            has_sv = ('sparse_view_' in task) and ('limited' not in task)
            has_la = ('limited_angle_' in task) and ('sparse' not in task)
            has_lasv = 'limitedsparse_view_' in task
            has_lasv2 = 'limitedsparse_view2_' in task
            
            if has_sv:
                self.task_kwargs.update({'sparse_view': sorted([int(n) for n in task.split('_')[2:]])})
            if has_la:
                self.task_kwargs.update({'limited_angle': [(0, int(n)) for n in task.split('_')[2:]]})
            if has_lasv:
                task_params = task.split('_')[2:]
                num_views = int(task_params[0])
                angle_range = (0, int(task_params[1]))
                self.task_kwargs.update({'limitedsparse_view':[(num_views, angle_range)]})
            if has_lasv2:
                task_params = task.split('_')[2:]
                num_views = int(task_params[0])
                angle_range = (0, int(task_params[1]))
                self.task_kwargs.update({'limitedsparse_view2':[(num_views, angle_range)]})
                
        if len(self.task_kwargs.keys()) == 0:
            raise NotImplementedError(f'Unsupported task: {task_list}')
            
        print('Task params:', self.task_kwargs)
    
    def prepare_windows(self, test_window):
        if test_window is not None:
            assert isinstance(test_window, list)
            print('Test window: ', test_window)
        else:
            test_window = [(3000,500), (300, 35), (1300,-600), (1000, 350)] 
            # (3000, 500): general window
            # (1300, -600): lung
            # (300, 35): soft tissues
            # (1000, 350): bones
            
        self.test_window = test_window
    
    def prepare_log(self,):
        self.tables = defaultdict(dict)  # simply record metrics
        self.tables_stat = defaultdict(dict)  # record statistics
        self.metric_names = ['psnr', 'ssim', 'rmse'] if self.save_rmse else ['psnr', 'ssim']
        
        for task, task_param_list in self.task_kwargs.items():
            for task_param in task_param_list:
                task_str = f"{self.dataset_name}_{task.replace('_','-')}-{task_param}".replace(' ','')
                for (width, center) in self.test_window:
                    window_str = f'({width}|{center})'
                    for metric_name in self.metric_names:
                        k = f'{window_str}_{metric_name}'
                        self.tables[task_str][k] = []
        
        save_name = self.opt.tester_save_name
        save_name = self.net_name if not save_name else save_name
        self.save_dir = os.path.join(self.opt.tester_save_dir, save_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Save figures to {self.save_dir}? : ', self.opt.tester_save_fig)
        self.num_saved_slices = 0
    
    def get_sino_mask(self, task_param, task='sparse_view'):
        self.net: BasicWrapper
        task_param_ = task_param
        if task == 'sparse_view':
            sino_mask = self.net.get_sparse_view_indices_from_full_angles(task_param_, return_mask=True)
        elif task == 'limited_angle':
            sino_mask = self.net.get_limited_angle_indices_from_full_angles(task_param_, return_mask=True)
        elif task == 'limitedsparse_view':
            num_views, angle_range = task_param_
            # This is hybrid CT scenario 1: SVCT within LACT.
            # Since the default number of full views is 720, here we 
            # have 1 view representing 2 degrees, thus the factor 2.
            start_view = angle_range[0] * 2  
            end_view = angle_range[1] * 2
            sino_mask = self.net.get_sparse_view_indices_from_full_angles(num_views, return_mask=True, start_view=start_view, end_view=end_view)
        elif task == 'limitedsparse_view2':
            # This is hybrid CT scenario 2: SVCT intersects LACT.
            num_views, angle_range = task_param_
            sino_mask1 = self.net.get_sparse_view_indices_from_full_angles(num_views, return_mask=True)
            sino_mask2 = self.net.get_limited_angle_indices_from_full_angles(angle_range, return_mask=True)
            sino_mask = sino_mask1 * sino_mask2
            
        return sino_mask
    
    def mask2prompt(self, sino_mask):
        return 1. - sino_mask[:, :, :, 0]
    
    def get_task_ids(self, task, task_param):
        task_ids = [None, None]
        if task == 'sparse_view':
            task_ids[0] = 0
            if task_param < 18:
                task_ids[1] = 0
            elif 16 <= task_param < 36:
                task_ids[1] = 1
            elif 36 <= task_param < 72:
                task_ids[1] = 2
            elif 72 <= task_param < 144:
                task_ids[1] = 3
            else:
                task_ids[1] = 4
        elif task == 'limited_angle':
            task_ids[0] = 1
            r = max(task_param) - min(task_param)
            if r < 90:
                task_ids[1] = 0
            elif 90 <= r < 120:
                task_ids[1] = 1
            elif 120 <= r < 150:
                task_ids[1] = 2
            else:
                task_ids[1] = 3
        return task_ids
            
    
    @torch.no_grad()
    def forward_net(self, src_w_con, task_param, task='sparse_view'):
        # For ProCT, `src_w_con` should be a concatenation of source (oracle) CT images and 
        # contextual (oracle) CT images, which will be used to obtain simulated source CT pairs 
        # and in-context CT pairs. They are concatenated along the batch dimension in order to 
        # enjoy the parallel processing capability of GPU.
        # For other networks that do not require contextual images, `src_w_con` should simply
        # be a mini-batch of source oracle CT images.
        
        if src_w_con.device.type == 'cpu':
            src_w_con = src_w_con.to(self.device, non_blocking=True)
            
        if self.net_name in ['fbp']:
            lq_imgs, gt_imgs = self.net.generate_input_target_mu(src_w_con, task_param, task=task)
            test_dict = {'in': lq_imgs,}
        elif 'proct' in self.net_name:
            batch_size = 1
            batch_size_eff = src_w_con.shape[0]
            batch_size_supp = batch_size_eff - 1
            
            input_imgs, target_imgs = self.net.generate_input_target_mu(src_w_con, task_param, task=task)
            lq_imgs, gt_imgs = input_imgs[:batch_size], target_imgs[:batch_size]  # [B, 1, H, W]
            lq_imgs_supp, gt_imgs_supp = input_imgs[-batch_size_supp:], target_imgs[-batch_size_supp:]
            
            lq_imgs_supp = E.rearrange(lq_imgs_supp, "(b s) c h w -> b s c h w", b=batch_size)
            gt_imgs_supp = E.rearrange(gt_imgs_supp, "(b s) c h w -> b s c h w", b=batch_size)
            supp_imgs = torch.cat([lq_imgs_supp, gt_imgs_supp], dim=-3) # [B, S, 2*C, H, W]
            
            sino_mask = self.get_sino_mask(task_param, task=task)
            sino_mask = sino_mask.float().repeat_interleave(batch_size, dim=0).to(lq_imgs.device)
            prompt = self.mask2prompt(sino_mask)
        
            task_ids = self.get_task_ids(task, task_param)  # Only useful when using learnable prompt
            pred_imgs = self.net(lq_imgs, context=supp_imgs, cond=prompt, task_ids=task_ids).clamp(0, 1)
            test_dict = {'out': pred_imgs,}
        else:
            raise NotImplementedError(f'network {self.net_name} not implemented')

        return lq_imgs, gt_imgs, test_dict
            
    
    def run(self):
        if self.net_name != 'fbp':
            self.net = self.load_net(net=self.net, net_checkpath=self.opt.net_checkpath, output=True)
            self.net = self.net.cuda()
            self.net = self.net.eval()
        
        for task, task_param_list in self.task_kwargs.items():
            for task_param in task_param_list:
                print(f'...Testing task {task}: {task_param}...')
                self.num_saved_slices = 0
                pbar = tqdm.tqdm(self.test_loader, ncols=100) if self.opt.use_tqdm else self.test_loader
                for i, mu_img in enumerate(pbar):
                    if 'proct' in self.net_name:
                        if self.train_dataset is not None and not self.opt.use_phantom:
                            # Use random images in the training set as in-context pairs
                            mu_con = [None] * self.opt.support_size
                            rand_indices = np.random.choice(len(self.train_dataset), size=self.opt.support_size, replace=False)
                            for i, r in enumerate(rand_indices):
                                mu_con[i] = self.train_dataset[r]
                            mu_con = torch.stack(mu_con, dim=0)
                            mu_img = torch.cat((mu_img, mu_con), dim=0)
                        elif self.opt.use_phantom:
                            # Use phantom as in-context pairs
                            batch_size_supp = self.batch_size
                            mu_con = self.phantom.repeat(batch_size_supp, 1, 1, 1)
                            mu_img = torch.cat((mu_img, mu_con), dim=0)
                    
                    in_mu, gt_mu, test_dict = self.forward_net(mu_img, task_param=task_param, task=task)
                    task_str = f"{self.dataset_name}_{task.replace('_','-')}-{task_param}".replace(' ','')
                    self.test_batch(in_mu, gt_mu, task_str, **test_dict)
        
        self.save_csv()
                
    def test_batch(self, in_mu, gt_mu, task_str, **test_dict):
        assert len(test_dict) > 0
        keys = list(test_dict.keys())
        batch = test_dict[keys[0]].shape[0]

        for b in range(batch):
            single_kwargs = {}
            for key in keys:
                single_kwargs[key] = test_dict[key][b:b+1]
            self.test_slice(in_mu[b:b+1], gt_mu[b:b+1], task_str, **single_kwargs)
    
    def test_slice(self, in_mu, gt_mu, task_str, **kwargs):
        gt_hu = self.test_dataset.denormalize_hu(gt_mu)
        in_hu = self.test_dataset.denormalize_hu(in_mu)
        
        for (width, center) in self.test_window:
            k1 = f'({width}|{center})_psnr'
            k2 = f'({width}|{center})_ssim'
            k3 = f'({width}|{center})_rmse'
            gt_win = self.test_dataset.window_transform(gt_hu, width=width, center=center)
            in_win = self.test_dataset.window_transform(in_hu, width=width, center=center)
            if self.opt.tester_save_fig:
                self.save_png(in_win, 'in', task_name=task_str, window_name=f'({width}|{center})')
                self.save_png(gt_win, 'gt', task_name=task_str, window_name=f'({width}|{center})')
                pass
            
            for tensor_name, tensor in kwargs.items():
                tensor_hu = self.test_dataset.denormalize_hu(tensor)
                tensor_win = self.test_dataset.window_transform(tensor_hu, width=width, center=center)                

                rmse, psnr, ssim = get_metrics(tensor_win, gt_win, val_range=1)
                ssim *= 100  # percentage
                
                if tensor_name == 'out':
                    self.tables[task_str][k1].append(psnr)
                    self.tables[task_str][k2].append(ssim)
                    if self.save_rmse:
                        self.tables[task_str][k3].append(rmse)

                if self.opt.tester_save_fig:
                    metric_values = str(psnr)[:5] + '-' + str(ssim)[:5]
                    metric_values = metric_values.replace('.','')
                    self.save_png(tensor_win, tensor_name, task_str, window_name=f'({width}|{center})', metric_values=metric_values)
        self.num_saved_slices += 1
    
    def save_png(self, tensor, tensor_name, task_name, window_name, metric_values=None):
        save_dir = os.path.join(self.save_dir, task_name, window_name)
        os.makedirs(save_dir, exist_ok=True)
        saved_slice = str(self.num_saved_slices).rjust(3, '0')
        if tensor_name in ['gt', 'in']:
            fullname = f'{saved_slice}_{tensor_name}.png'
        else:
            fullname = f'{saved_slice}_{tensor_name}_{self.net_name}_{metric_values}.png'
        save_path = os.path.join(save_dir, fullname)
        torchvision_save_image(tensor, save_path, normalize=False)
    
    def fill_stat_table(self, task_str):
        table_stat_tmp = defaultdict(dict)
        for (width, center) in self.test_window:
            for metric_name in self.metric_names:
                data_list = self.tables[task_str][f'({width}|{center})_{metric_name}']
                table_stat_tmp[f'({width}|{center})'][f'avg_{metric_name}'] = np.mean(data_list)
                table_stat_tmp[f'({width}|{center})'][f'std_{metric_name}'] = np.std(data_list)
                #table_stat_tmp[f'({width}|{center})']['min_' + metric_name] = np.min(data_list)
                #table_stat_tmp[f'({width}|{center})']['max_' + metric_name] = np.max(data_list)
                
            print(f"[{task_str}] PSNR under {(width, center)}: {table_stat_tmp[f'({width}|{center})']['avg_psnr']}")
            print(f"[{task_str}] SSIM under {(width, center)}: {table_stat_tmp[f'({width}|{center})']['avg_ssim']}")
            if self.save_rmse:
                print(f"[{task_str}] RMSE under {(width, center)}: {table_stat_tmp[f'({width}|{center})'][f'avg_rmse']}")
        
        self.tables_stat[task_str] = table_stat_tmp
        print('---check keys:', self.tables_stat[task_str].keys())

    def save_csv(self):
        task_strs = list(self.tables.keys())
        
        for task_str in task_strs:
            table_all = self.tables[task_str]
            full_table_keys = list(table_all.keys())
            print('full_table_keys: ', full_table_keys)
            self.fill_stat_table(task_str)
            
            if self.opt.tester_save_tab:
                csv_path = os.path.join(self.save_dir, f'{task_str}_{self.net_name}_all.csv')
                print('Wrting full table to: ', csv_path)
                with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(full_table_keys)
                    values = np.array([table_all[k] for k in full_table_keys]).T
                    writer.writerows(values)
        
        if self.opt.tester_save_tab:
            csv_stat_path = os.path.join(self.save_dir, f'{self.dataset_name}_{self.net_name}_stat.csv')
            print('Wrting statistic table to: ', csv_stat_path)
            
            with open(csv_stat_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for task_str in task_strs:
                    table_stat = self.tables_stat[task_str]
                    window_keys = list(table_stat.keys())
                    stat_keys = list(table_stat[window_keys[0]].keys())
                    
                    writer.writerow([task_str] + window_keys)
                    for s in stat_keys:
                        writer.writerow([s] + [table_stat[w][s] for w in window_keys])
                    writer.writerow([''] * (len(window_keys) + 1))


