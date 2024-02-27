import cv2
from torch.utils.data import Dataset


class BaseCTDataset(Dataset):
    def __init__(
        self, 
        img_shape=(512, 512),
        min_hu=-1024,
        max_hu=3072,
        mu_water=0.192,  # 0.192 [1/cm] = 192 [1/m]
        mu_air=0.0):
        
        ############## Basic setting #################
        self.img_shape = img_shape
        
        ############## CT settings #################
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.mu_water = mu_water
        self.mu_air = mu_air
    
    def _resize_image(self, img):
        dsize = list(self.img_shape)[::-1]
        return cv2.resize(img, dsize, cv2.INTER_CUBIC).astype(img.dtype)
    
    @staticmethod
    def clip_range(img, min_val=-1024, max_val=3072):
        assert min_val < max_val
        return img.clip(min_val, max_val)
    
    def normalize_hu(self, hu, norm_type='clip'):
        if norm_type == 'clip':
            hu_ = (hu - self.min_hu) / (self.max_hu - self.min_hu)
        else:  # mu (attenuation coefficients) as normalized values
            hu_ = hu / 1000 * (self.mu_water - self.mu_air) + self.mu_water
        return hu_

    def denormalize_hu(self, mu, norm_type='clip'):
        if norm_type == 'clip':
            return (self.max_hu - self.min_hu) * mu + self.min_hu
        else:
            return (mu - self.mu_water) / (self.mu_water - self.mu_air) * 1000

    @staticmethod
    def window_transform(hu, width=3000, center=500, norm_to_255=False):
        ''' hu_image -> 0-1 normalization'''
        window_min = float(center) - 0.5 * float(width)
        win_image = (hu - window_min) / float(width)
        win_image = win_image.clip(0, 1)
        if norm_to_255:
            win_image = (win_image * 255).astype('float')
        return win_image

    @staticmethod
    def back_window_transform(win_image, width=3000, center=500, norm=False):
        ''' 0-1 normalization -> hu_image'''
        window_min = float(center) - 0.5 * float(width)
        if norm:
            win_image = win_image / 255.
        hu = win_image * float(width) + window_min
        return hu