import torch
import numpy as np
import torch.nn as nn
from torch_radon import RadonFanbeam


class BasicWrapper(nn.Module):
    """
    Provide functions for on-the-fly radon/iradon operations with TorchRadon V1.
    Source: https://github.com/matteo-ronchetti/torch-radon
    """
    def __init__(
        self, img_size=256, num_full_views=720, simul_poisson_rate=1e6, simul_gaussian_rate=0.01):
        super().__init__()
        self.num_full_views = num_full_views
        self.source_distance = 1075
        self.det_count = 672
        self.img_size = img_size
        self.simul_poisson_rate = simul_poisson_rate
        self.simul_gaussian_rate = simul_gaussian_rate
        print('full views: {}'.format(self.num_full_views))
        self.full_angles = self.get_angles(self.num_full_views)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward in wrapper should be implemented.')

    @staticmethod
    def circular_move(a, bias, lb, ub):
        if isinstance(a, list):
            a_new = list(map(lambda x: x+bias, a))
        else:
            a_new = a + bias
        
        a_new[a_new > ub] -= ub
        a_new[a_new < lb] += ub
        return a_new
    
    def get_sparse_view_indices_from_full_angles(
        self, num_views, return_mask=False, output_nchw=True, 
        start_view=None, end_view=None, num_avail_views=None):
        start_view = start_view if start_view is not None else 0
        end_view = end_view if end_view is not None else (self.num_full_views-1)
        num_avail_views = (end_view - start_view + 1) if num_avail_views is None else num_avail_views
        
        indices = np.arange(start_view+1, end_view+1+1)
        
        if num_views == self.num_full_views:
            bool_array = np.ones(self.num_full_views,)
        else:
            bool_array = np.zeros(self.num_full_views,)
            bool_array[indices-1] = np.ma.masked_equal((indices - start_view) % (num_avail_views//num_views), 1).mask  # shape:(num_full_views,)
           
            
        if not return_mask:
            selected_indices = sorted(list(indices[bool_array] - 1))
            return selected_indices
            #selected_indices = np.linspace(0, self.num_full_views-1, int(num_views), False).astype(int)
            #return sorted(list(set(selected_indices)))
        else:
            bool_array = torch.from_numpy(bool_array).float()
            bool_mask = bool_array.repeat(self.det_count, 1).permute(1, 0)
            if output_nchw:
                bool_mask = bool_mask.reshape(1, 1, *bool_mask.shape).contiguous()
            return bool_mask
            
    def get_limited_angle_indices_from_full_angles(self, angle_range, return_mask=False):
        """Get the corresponding indices of `angles` as a subset of `full_angles`."""
        # `ranges` is a list/tuple containing ranges of the selected views
        # For example, ranges = [(0, np.pi/6), (np.pi/3, np.pi/2)]
        # selects items from `self.full_angles` satisfying 
        #   0 <= items <= pi/6  or  pi/3 <= items <= pi/2
        selected_indices = []
        full_angles = self.full_angles
        start_degree, end_degree = min(angle_range), max(angle_range)
        
        ratio = np.pi / 180
        min_val = start_degree * ratio
        max_val = end_degree * ratio
        bool_array = np.logical_and(full_angles >= min_val, full_angles <= max_val)
        
        if not return_mask:
            selected_indices = sorted(np.where(bool_array)[0])
            return selected_indices
        else:
            bool_array = torch.from_numpy(bool_array).float()
            bool_mask = bool_array.repeat(self.det_count, 1).permute(1, 0)
            bool_mask = bool_mask.reshape(1, 1, *bool_mask.shape).contiguous()
            return bool_mask
       
    def generate_input_target_mu(self, mu_ct, task_param, task='sparse_view', return_sinogram=False):
        input_sinogram, target_sinogram, input_angles = self.generate_input_target_sinogram(mu_ct, task_param, task=task)
        target_mu = self.fbp(target_sinogram,)
        input_mu = self.fbp(input_sinogram, input_angles)
        
        if return_sinogram:
            return input_mu, target_mu, input_sinogram, target_sinogram
        else:
            return input_mu, target_mu
    
    def generate_input_target_sinogram(self, mu_ct, task_param, task='sparse_view'):
        # task_param could be num_views, view_ranges, or dose_factor
        target_sinogram = self.fp(mu_ct)
        target_sinogram = self.add_poisson_noise_to_sinogram(target_sinogram, self.simul_poisson_rate)
        target_sinogram = self.add_gaussian_noise_to_sinogram(target_sinogram, self.simul_gaussian_rate)
        
        if task == 'sparse_view':
            poisson_rate = self.simul_poisson_rate
            input_angles = self.get_angles(task_param)
        elif task == 'limited_angle':
            poisson_rate = self.simul_poisson_rate
            input_angles = self.get_limited_angles(task_param)  # input is a tuple (a1, a2)
        elif task == 'sparselimited_angle':
            poisson_rate = self.simul_poisson_rate
            assert isinstance(task_param, (list, tuple)), f'The parameter for a hybrid task ({task}) should be a tuple/list, got {task_param}.'
            num_views, angle_range = task_param
            input_angles = self.get_sparse_limited_angles(num_views, angle_range)
        elif task == 'limitedsparse_view':
            poisson_rate = self.simul_poisson_rate
            assert isinstance(task_param, (list, tuple)), f'The parameter for a hybrid task ({task}) should be a tuple/list, got {task_param}.'
            num_views, angle_range = task_param
            input_angles = self.get_limited_sparse_angles(num_views, angle_range, svct_within_lact=True)
        elif task == 'limitedsparse_view2':
            poisson_rate = self.simul_poisson_rate
            assert isinstance(task_param, (list, tuple)), f'The parameter for a hybrid task ({task}) should be a tuple/list, got {task_param}.'
            num_views, angle_range = task_param
            input_angles = self.get_limited_sparse_angles(num_views, angle_range, svct_within_lact=False)
        elif task == 'low_dose':
            poisson_rate = self.simul_poisson_rate * task_param
            input_angles = self.full_angles
        else:
            raise NotImplementedError(f"Unsupported task type: {task}, try one of ['sparse_view', 'limited_angle', 'low_dose', 'sparselimited_angle', 'limitedsparse_view'].")
        
        input_sinogram = self.fp(mu_ct, input_angles)
        input_sinogram = self.add_poisson_noise_to_sinogram(input_sinogram, poisson_rate)
        input_sinogram = self.add_gaussian_noise_to_sinogram(input_sinogram, self.simul_gaussian_rate)
        return input_sinogram, target_sinogram, input_angles
    
    @staticmethod
    def get_angles(num_views, start_degree=None, end_degree=None):
        start_radian = start_degree / 180 * np.pi if start_degree is not None else 0
        end_radian = end_degree / 180 * np.pi if end_degree is not None else np.pi*2
        angles = np.linspace(start_radian, end_radian, num_views, endpoint=False)  # select views according to the specified number of views
        return angles
    
    def get_limited_angles(self, angle_range):
        start_degree, end_degree = min(angle_range), max(angle_range)
        perc = (end_degree - start_degree) / 360
        num_views = int(perc * self.num_full_views)
        return self.get_angles(num_views, start_degree=start_degree, end_degree=end_degree)
    
    def get_sparse_limited_angles(self, num_views, angle_range):
        # Hybrid CT 1: union of SVCT and LACT
        la_angles = self.get_limited_angles(angle_range)
        sv_angles = self.get_angles(num_views)
        svla_angles = np.array(sorted(list(la_angles) + list(sv_angles)))
        return svla_angles
    
    def get_limited_sparse_angles(self, num_views, angle_range, svct_within_lact=True):
        # Hybrid CT 2: SVCT within LACT
        start_degree, end_degree = angle_range
        if svct_within_lact:
            lasv_angles = self.get_angles(num_views, start_degree=start_degree, end_degree=end_degree)
        else:
            la_angles = self.get_limited_angles(angle_range)
            sv_angles = self.get_angles(num_views)
            idx = np.argwhere(np.logical_or(sv_angles < min(la_angles), sv_angles > max(la_angles)))
            lasv_angles = np.delete(sv_angles, list(idx))
        return lasv_angles
    
    def generate_multiview_input_target_mu(self, mu_ct, task_param_list):
        assert isinstance(task_param_list, (tuple, list)), f'task_param_list should be a list or tuple, got {task_param_list}'
        sinogram_list, angle_list = self.generate_multiview_input_target_sinogram(mu_ct, task_param_list)
        num = len(sinogram_list)
        mu_list = [None] * num
        for i in range(num):
            mu_list[i] = self.fbp(sinogram_list[i], angle_list[i])
        return mu_list
    
    def generate_multiview_input_target_sinogram(self, mu_ct, task_param_list):
        # [TODO] task_param could be num_views, view_ranges, or dose_factor
        target_sinogram = self.fp(mu_ct)
        target_sinogram = self.add_poisson_noise_to_sinogram(target_sinogram, self.simul_poisson_rate)
        target_sinogram = self.add_gaussian_noise_to_sinogram(target_sinogram, self.simul_gaussian_rate)
        sinogram_list = [None] * (len(task_param_list) + 1)
        angle_list = [None] * (len(task_param_list) + 1)
        for i, task_param in enumerate(task_param_list):
            input_angles = self.get_angles(task_param)
            input_sinogram = self.fp(mu_ct, input_angles)
            input_sinogram = self.add_poisson_noise_to_sinogram(input_sinogram, self.simul_poisson_rate)
            input_sinogram = self.add_gaussian_noise_to_sinogram(input_sinogram, self.simul_gaussian_rate)
            sinogram_list[i] = input_sinogram
            angle_list[i] = input_angles
        
        sinogram_list[-1] = target_sinogram
        angle_list[-1] = self.full_angles
        return sinogram_list, angle_list
    
    # ------------ basic radon function ----------------
    # avoid possible cuda error, put radon func in the module
    def fbp(self, sinogram, angles=None):
        '''sinogram to ct image'''
        angles = self.full_angles if angles is None else angles
        img_size = self.img_size[0] if isinstance(self.img_size, (tuple,list)) else self.img_size
        radon_tool = RadonFanbeam(img_size, angles, self.source_distance, det_count=self.det_count,)
        filtered_sinogram = radon_tool.filter_sinogram(sinogram, "ram-lak")
        back_proj = radon_tool.backprojection(filtered_sinogram)
        return back_proj
    
    def fp(self, ct_image, angles=None):
        '''ct image to sinogram'''
        angles = self.full_angles if angles is None else angles
        img_size = self.img_size[0] if isinstance(self.img_size, (tuple,list)) else self.img_size
        radon_tool = RadonFanbeam(img_size, angles, self.source_distance, det_count=self.det_count,)
        sinogram = radon_tool.forward(ct_image)
        return sinogram
    
    @staticmethod
    def add_poisson_noise_to_sinogram(sinogram: torch.Tensor, noise_rate=1e6):
        # noise rate: background intensity or source influx
        # [TODO]: This simulation is not very accurate for noise_rate < 1e5.
        if noise_rate > 0:
            smin, smax = sinogram.min(), sinogram.max()
            max_val = smax.item() #* 0.14925  # this factor makes the noise more realistic?
            sinogram_ct = (noise_rate * torch.exp(-sinogram / max_val))
            sinogram_noise = torch.poisson(sinogram_ct).clamp(min=sinogram_ct.min())
            sinogram_out = - max_val * torch.log(sinogram_noise / noise_rate)
            sinogram_out = sinogram_out.clamp(smin, smax)
        else:
            sinogram_out = sinogram
        return sinogram_out

    @staticmethod
    def add_gaussian_noise_to_sinogram(sinogram: torch.Tensor, sigma=0.1):
        if sigma > 0:
            smin, smax = sinogram.min(), sinogram.max()
            dtype = sinogram.dtype
            if not sinogram.is_floating_point():
                sinogram = sinogram.to(torch.float32)
            sinogram_out = sinogram + sigma * torch.randn_like(sinogram)
            sinogram_out = sinogram_out.clamp(sinogram.min(), sinogram.max())
            if sinogram_out.dtype != dtype:
                sinogram_out = sinogram_out.to(dtype)
            sinogram_out = sinogram_out.clamp(smin, smax)
        else:
            sinogram_out = sinogram
        return sinogram_out

