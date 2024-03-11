import os
import sys
sys.path.append('..')
from datasets.base_ct_dataset import BaseCTDataset
import cv2
from PIL import Image
import torch
import SimpleITK as sitk
import numpy as np
from utilities.misc import ensure_tuple_rep


#torch.multiprocessing.set_start_method('spawn')

PHANTOM = np.load('datasets/phantom.npy')

class SimpleCTDataset(BaseCTDataset):
    def __init__(
        self, 
        img_list_info,
        root_dir='',
        img_shape=(512, 512),
        clip_hu=False,
        min_hu=-1024,
        max_hu=3072,
        mu_water=0.192,  #  0.192 [1/cm] = 192 [1/m]
        mu_air=0.0,
        mode='train',
        num_train=10000,
        num_val=1000):
        
        ############## Basic setting #################
        img_shape = ensure_tuple_rep(img_shape, 2)
        self.root_dir = root_dir
        self.img_shape = img_shape
        self.num_train = num_train
        self.num_val = num_val
        self.img_path_list = self.get_img_path_list_from_text(img_list_info, mode=mode)
        
        ############## CT settings #################
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.mu_water = mu_water
        self.mu_air = mu_air
        self.clip_hu = clip_hu
        if clip_hu:
            print('Clipping HU range to: ', (min_hu, max_hu))
    
    def get_img_path_list_from_text(self, txt_path, mode='train'):
        with open(txt_path, 'r') as f:
            img_path_list = [os.path.join(self.root_dir, _.strip()) for _ in f.readlines()]
        
        if mode == 'train':
            paths = img_path_list[:self.num_train]
        else:
            paths = img_path_list[-self.num_val:]
        return paths
        
    def __len__(self):
        return len(self.img_path_list)
    
    def _load_image(self, img_path):
        img_name = img_path.split('/')[-1]
        img_ext = img_name.split('.')[-1].lower()
        if img_ext == 'npy':
            img = np.load(img_path).squeeze()
        elif img_ext in ['jpg', 'jpeg', 'png']:
            img = np.asarray(Image.open(img_path))
            # img = cv2.imread(img_path).squeeze()
            # if img.ndim == 3:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img_ext in ['gz', 'dcm', 'ima']:
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).squeeze()
        else:
            raise NotImplementedError(f'Got unsupported image extension: {img_ext}')
        
        if 'deeple' in img_path.lower():  # preprocessing for deeplesion
            img = (img - 32768).astype(np.int16)
            
        return img
    
    def get_input_hu(self, img_path):
        while True:
            try:
                hu_img = self._load_image(img_path)
                break
            except:
                print('Error file:', img_path)
                img_path_new = np.random.choice(self.img_path_list)
                while img_path_new == img_path:
                    img_path_new = np.random.choice(self.img_path_list)
                img_path = img_path_new
                print('Trying alternative:', img_path)
        return hu_img
    
    def get_input_mu(self, img_path):
        hu_img = self.get_input_hu(img_path)
        
        if hu_img.shape != self.img_shape:
            hu_img = self._resize_image(hu_img)
        
        if self.clip_hu:
            hu_img = self.clip_range(hu_img)
        
        mu_img = self.normalize_hu(torch.from_numpy(hu_img))
        return mu_img.float().unsqueeze(0)
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        return self.get_input_mu(img_path)




if __name__ == '__main__':
    pass