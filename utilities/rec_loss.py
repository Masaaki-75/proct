import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchvision.transforms as transforms
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class GaussianConv:
    def __init__(self, window_size, sigma=1.5, channel=1):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.window_size = window_size
        self.padding = (self.window_size - 1) // 2
        self.window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)])
        return gauss / gauss.sum()
    
    def __call__(self, x):
        window = self.window.to(x.device).type_as(x)
        return F.conv2d(x, window, padding=self.padding)
    
    

def get_metrics(preds, labels, val_range, eps=1e-10, window_size=11):
    mse = ((preds - labels) ** 2).mean().item()
    mse_ = mse + eps if mse == 0 else mse
    rmse = math.sqrt(mse)
    psnr = 10 * math.log10((val_range ** 2) / mse_)
    ssim = get_ssim(preds, labels, window_size=window_size, val_range=val_range).item()
    return rmse, psnr, ssim

def get_mse(img1, img2):
    return ((img1 - img2) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, data_range, window, K=(0.01, 0.03), eps=0):
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    window = window.to(img1.device).type_as(img1)
    num_channels = img1.shape[1]
    
    mu1 = F.conv2d(img1, window, groups=num_channels)
    mu2 = F.conv2d(img2, window, groups=num_channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,  groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=num_channels) - mu1_mu2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs_map = v1 / (v2 + eps)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2 + eps)
    return ssim_map, cs_map
    

def get_ssim(
    img1, img2, window_size=11, window=None, size_average=True, val_range=None, K=(0.01, 0.03),
    output_ssim_only=True):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    
    if len(img1.size()) == 2:  # [H, W] -> [1, 1, H, W]
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_ )
        img2 = img2.view(1, 1, *shape_ )
        
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        val_range = max_val - min_val

    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)

    ssim_map, cs_map = _ssim(img1, img2, data_range=val_range, window=window, K=K)

    if size_average:
        cs = cs_map.mean()  # scalar
        ret = ssim_map.mean()
    else:
        cs = cs_map.mean(1).mean(1).mean(1)  # [B,]
        ret = ssim_map.mean(1).mean(1).mean(1)

    return (ret, cs) if not output_ssim_only else ret

    

def get_msssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None, normalize='relu', K=(0.01, 0.03)):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mcs = []
    
    (_, channel, height, width) = img1.size()
    
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        val_range = max_val - min_val
        
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    
    for i in range(levels):
        ssim_map, cs_map = _ssim(img1, img2, data_range=val_range, window=window, K=K, eps=1e-8)
        cs = cs_map.flatten(2).mean(-1)  # [B, C]

        if i < levels - 1:  # do not compute for last one conv
            cs = torch.relu(cs) if normalize == 'relu' else cs
            mcs.append(cs)  # no need to save last cs, no need to save other ssim
            # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = F.avg_pool2d(img1, kernel_size=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, padding=padding)

    ssim = ssim_map.flatten(2).mean(-1)  # [B, C]
    ssim = torch.relu(ssim) if normalize == 'relu' else ssim
    mcs_and_ssim = torch.stack(mcs + [ssim], dim=0)  # [level, B, C]
    output = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)  # [B, C]
    return output.mean() if size_average else output.mean(1)


# Classes to re-use window
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, channel=1):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1. - get_ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range)

class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, img_shape=(256, 256), size_average=True, val_range=1, channel=1):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        
        height, width = img_shape
        real_size = min(window_size, height, width)
        self.window = create_window(real_size, channel=channel)
    
    def get_msssim(self, img1, img2, normalize='relu', K=(0.01, 0.03)):
        device = img1.device  # [N, C, H, W] shape
        weights = self.weights.to(device)
        levels = weights.size()[0]
        mcs = []
        
        val_range = self.val_range
        if val_range is None:
            max_val = 255 if torch.max(img1) > 128 else 1
            min_val = -1 if torch.min(img1) < -0.5 else 0
            val_range = max_val - min_val
        
        for i in range(levels):
            ssim_map, cs_map = _ssim(img1, img2, data_range=val_range, window=self.window, K=K, eps=1e-8)
            cs = cs_map.flatten(2).mean(-1)  # [B, C]

            if i < levels - 1:  # do not compute for last one conv
                cs = torch.relu(cs) if normalize == 'relu' else cs
                mcs.append(cs)  # no need to save last cs, no need to save other ssim
                # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
                padding = [s % 2 for s in img1.shape[2:]]
                img1 = F.avg_pool2d(img1, kernel_size=2, padding=padding)
                img2 = F.avg_pool2d(img2, kernel_size=2, padding=padding)

        ssim = ssim_map.flatten(2).mean(-1)  # [B, C]
        ssim = torch.relu(ssim) if normalize == 'relu' else ssim
        mcs_and_ssim = torch.stack(mcs + [ssim], dim=0)  # [level, B, C]
        output = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)  # [B, C]
        return output.mean() if self.size_average else output.mean(1)

    def forward(self, img1, img2):
        return 1. - self.get_msssim(img1, img2)



class VGGPerceptualLoss(nn.Module):
    # directly stolen from https://github.com/jeya-maria-jose/TransWeather/blob/main/perceptual.py
    def __init__(self, vgg_model):
        super().__init__() 
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred, gt):
        loss = []
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if gt.shape[1] == 1:
            gt = gt.repeat(1, 3, 1, 1)
        
        pred = self.normalize(pred)
        gt = self.normalize(gt)
        pred_features = self.output_features(pred)
        gt_features = self.output_features(gt)
        for pred_feature, gt_feature in zip(pred_features, gt_features):
            loss.append(F.mse_loss(pred_feature, gt_feature))

        return sum(loss)/len(loss)


if __name__ == '__main__':
    criterion = MSSSIMLoss(channel=1)
    img1 = torch.randint(10, 20, size=(1, 1, 256, 256)).float()
    img2 = img1 + torch.randn_like(img1)
    loss = criterion(img1, img2)
    print('loss: ', loss)