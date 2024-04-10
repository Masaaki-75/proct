import os
import wandb
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
from torch.optim.lr_scheduler import _LRScheduler

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class LinoPolyScheduler(_LRScheduler):
    r"""
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.PolyScheduler(optimizer, min_lr=0.01, steps_per_epoch=None, epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>     scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(self,
                 optimizer,
                 power=1.0,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 min_lr=0,
                 last_epoch=-1,
                 verbose=False):

        # Validate optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        # self.by_epoch = by_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.power = power

        # check param
        param_dic = {'total_steps': total_steps, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch}
        for k, v in param_dic.items():
            if v is not None:
                if v <= 0 or not isinstance(v, int):
                    raise ValueError("Expected positive integer {}, but got {}".format(k, v))

        # Validate total_steps
        if total_steps is not None:
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is None:
            self.total_steps = epochs
        elif epochs is not None and steps_per_epoch is not None:
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError("You must define either total_steps OR epochs OR (epochs AND steps_per_epoch)")

        super(LinoPolyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        coeff = (1 - step_num / self.total_steps) ** self.power

        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]


class BasicTrainer:
    def __init__(self):
        self.iter = 0
        self.epoch = 0
        #self.cttool = CTTool()
        self.multigpu = False
    
    @staticmethod
    def fix_seed(seed=3407):
        # with open('seed.txt', 'a') as f:
        #     f.write('Using seed '+str(seed)+' at '+str(datetime.now()))
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple gpu
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    @staticmethod
    def weights_init(m):             
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    @staticmethod
    def apply_monai_dict_transform(image1, image2=None, transform=None):
        if transform is not None:
            if image2 is not None:
                input_dict = {'image': image1, 'label': image2}
                output_dict = transform(input_dict)
                if isinstance(output_dict, list):
                    output_dict = output_dict[0]
                image1_out = output_dict['image'].as_tensor()
                image2_out = output_dict['label'].as_tensor()
                return image1_out, image2_out
            else:
                input_dict = {'image': image1}
                output_dict = transform(input_dict)
                if isinstance(output_dict, list):
                    output_dict = output_dict[0]
                    
                if hasattr(output_dict['image'], 'as_tensor'):
                    image1_out = output_dict['image'].as_tensor()
                else:
                    image1_out = output_dict['image']
                return image1_out
        else:
            return (image1, image2) if image2 is not None else image1

    def get_optimizer(self, net=None, lr=None):
        opt = self.opt
        net = self.net if net is None else net
        lr = opt.lr if lr is None else lr
        optimizer_name = opt.optimizer.lower()
        if optimizer_name == 'adam':
            return optim.Adam(net.parameters(), lr=lr, betas=(self.opt.beta1, self.opt.beta2))
        elif optimizer_name == 'sgd':
            return optim.SGD(net.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
        elif optimizer_name == 'adamw':
            return optim.AdamW(net.parameters(), lr=lr, betas=(self.opt.beta1, self.opt.beta2))
        else:
            raise NotImplementedError('Currently only support optimizers among: Adam, AdamW and SGD.')
    
    def get_scheduler(self, optimizer=None, last_epoch=-1):
        opt = self.opt
        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler_name = opt.scheduler.lower()
        if scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma, last_epoch=last_epoch)
        elif scheduler_name == 'mstep':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.step_gamma, last_epoch=last_epoch)
        elif scheduler_name == 'exp':
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.step_gamma)
        elif scheduler_name == 'poly':
            #return optim.lr_scheduler.PolynomialLR(optimizer, total_iters=opt.poly_iters, power=opt.poly_power)
            return LinoPolyScheduler(optimizer, power=opt.poly_power, epochs=opt.epochs, min_lr=1e-6)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
        else:
            warnings.warn(f'Currently only support schedulers among Step, MultiStep, Exp, Poly, Cosine, got {scheduler_name}. So using none (constant).')
            return None
            #raise NotImplementedError('Currently only support schedulers among Step, MultiStep, Exp, Poly, Cosine.')
            
    @staticmethod
    def save_checkpoint(param, path, name:str, epoch=None):
        # simply save the checkpoint by epoch
        if not os.path.exists(path):
            os.makedirs(path)
        if epoch is not None:
            checkpoint_path = os.path.join(path, name + '_e{}.pkl'.format(epoch))
        else:
            checkpoint_path = os.path.join(path, name + '_latest.pkl')
        torch.save(param, checkpoint_path)

    def save_net(self, net=None, save_latest=False, name=''):
        if net is None:
            net_param = self.net.module.state_dict() if self.multigpu else self.net.state_dict()
        else:
            net_param = net.module.state_dict() if self.multigpu else net.state_dict()
        checkpoint = {'network': net_param, 'epoch': self.epoch, 'iter': self.iter}
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        epoch = None if save_latest else self.epoch
        name = '-' + str(name) if name else str(name)
        self.save_checkpoint(checkpoint, checkpoint_path, self.opt.checkpoint_dir + name + '-net', epoch=epoch)
        print(f'Saved net at epoch {self.epoch} step {self.iter}.')

    def save_opt(self, optimizer=None, scheduler=None, save_latest=False):
        optimizer = self.optimizer if optimizer is None else optimizer
        optimizer_param = optimizer.state_dict()
        
        if scheduler is None and hasattr(self, 'scheduler') and self.scheduler is not None:
            scheduler_param = self.scheduler.state_dict()
        else:
            if hasattr(scheduler, 'state_dict'):
                scheduler_param = scheduler.state_dict()
            else:
                scheduler_param = None
                
        checkpoint = {'optimizer': optimizer_param, 'scheduler': scheduler_param, 'epoch' : self.epoch, 'iter': self.iter}
        checkpoint_path = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
        epoch = None if save_latest else self.epoch
        self.save_checkpoint(checkpoint, checkpoint_path, self.opt.checkpoint_dir + '-opt', epoch=epoch)
        print(f'Saved opt at epoch {self.epoch} step {self.iter}.')

    def load_net(self, net=None, net_checkpath=None, output=False):
        net_checkpath = self.opt.net_checkpath if net_checkpath is None else net_checkpath
        
        if os.path.exists(net_checkpath):
            net_checkpoint = torch.load(net_checkpath, map_location=self.device)
            
            if net is None:
                self.net.load_state_dict(net_checkpoint['network'], strict=True)
                print(f'Finish loading network from (e{self.epoch}, i{self.iter}):', net_checkpath)
                if output:
                    return self.net
            else:
                net.load_state_dict(net_checkpoint['network'], strict=True)
                print(f'Finish loading network from (e{self.epoch}, i{self.iter}):', net_checkpath)
                if output:
                    return net
        else:
            raise ValueError(f'Checkpoint path does not exist: {net_checkpath}.')


    def load_opt(self,):
        if self.opt.opt_checkpath:
            opt_checkpath = self.opt.opt_checkpath
        else:
            warnings.warn('opt_checkpath not provided, trying to load latest opt...')
            checkpoint_dir = os.path.join(self.opt.checkpoint_root, self.opt.checkpoint_dir)
            checkpoint_name = self.opt.checkpoint_dir + '-opt'
            opt_checkpath = os.path.join(checkpoint_dir, checkpoint_name + '_latest.pkl')
        
        if os.path.exists(opt_checkpath):
            opt_checkpoint = torch.load(opt_checkpath, map_location=self.device)
            if 'optimizer' in opt_checkpoint.keys():
                self.optimizer.load_state_dict(opt_checkpoint['optimizer'])
                print('finish loading optimizer from: ', opt_checkpath)
            if 'scheduler' in opt_checkpoint.keys() and opt_checkpoint['scheduler'] is not None and hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.load_state_dict(opt_checkpoint['scheduler'])
                print('finish loading scheduler from: ', opt_checkpath)
            if 'epoch' in opt_checkpoint.keys():
                self.epoch = opt_checkpoint['epoch']
                print('finish loading epoch: {}'.format(self.epoch))
            if 'iter' in opt_checkpoint.keys():
                self.iter = opt_checkpoint['iter']
                print('finish loading iteration: {}'.format(self.iter))
        else:
            warnings.warn(f'**opt_checkpath provided but does not exist ({opt_checkpath}).**')


    @staticmethod
    def get_pixel_criterion(mode='l1'):
        assert isinstance(mode, str)
        mode = mode.lower()
        if mode == 'l1':
            criterion = torch.nn.L1Loss(reduction='mean') 
        elif mode == 'sml1':
            criterion = torch.nn.SmoothL1Loss(reduction='mean')
        elif mode == 'l2':
            criterion = torch.nn.MSELoss(reduction='mean')
        elif mode == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError('pixel_loss error: mode not in [l1, sml1, l2].')
        return criterion

    # ---- basic logging function ----
    @staticmethod
    def tensorboard_scalar(writer, r_path, step, **kwargs):
        ''' log kwargs by path automatically
        args:
            writer: self.writer
            r_path: 相对路径 : /train/epoch/ + key
        kwargs:
            dict: {key: data}
        '''
        for key in kwargs.keys():
            path = os.path.join(r_path, key)
            writer.add_scalar(path, kwargs[key], global_step=step)

    @staticmethod
    def tensorboard_image(writer, r_path, **kwargs):
        ''' log kwargs image by path automatically
        args: ...
        kwargs: 
            dict: {key: data}
        writer.add_image():
            (tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
        '''
        for key in kwargs.keys():
            path = os.path.join(r_path, key)
            writer.add_image(tag=path, img_tensor=kwargs[key], global_step=0, dataformats='CHW',)

    @staticmethod
    def wandb_init(proj_name, net_name):
        wandb.init(project=str(proj_name), resume='allow', name=net_name,)
    
    @staticmethod
    def wandb_init2(opt):
        key = opt.wandb_key
        wandb.login(key=key)
        wandb_root = opt.tensorboard_root if opt.wandb_root == '' else opt.wandb_root
        wandb_dir = opt.tensorboard_dir if opt.wandb_dir == '' else opt.wandb_dir
        wandb_path = os.path.join(wandb_root, wandb_dir)
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        wandb.init(project=opt.wandb_project, name=str(wandb_dir), dir=wandb_path, resume='allow', reinit=True,)
    
    @staticmethod
    def to_wandb_img(**kwargs):
        # turn torch makegrid to wandb image
        for key, value in kwargs.items():
            kwargs[key] = wandb.Image(kwargs[key].squeeze().float().cpu())
        return kwargs

    @staticmethod
    def wandb_logger(r_path=None, step_name=None, step=None, **kwargs):
        log_info = {}
        if step is not None:
            log_info.update({str(step_name): step})
        for key, value in kwargs.items():
            if r_path is not None:
                key_name = str(os.path.join(r_path, key))
            else:
                key_name = key
            log_info[key_name] = kwargs[key]
        wandb.log(log_info)
        
    @staticmethod
    def wandb_scalar(r_path, step=None, **kwargs):
        for key in kwargs.keys():
            if step is not None:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]}, step=step)
            else:
                wandb.log({'{}'.format(os.path.join(r_path, key)): kwargs[key]})
    
    @staticmethod
    def wandb_image(step=None, **kwargs):
        for key in kwargs.keys():
            kwargs[key] = wandb.Image(kwargs[key].float().cpu())
        if step is not None:
            wandb.log(kwargs, step=step)
        else:
            wandb.log(kwargs)

    def fit(self):
        raise NotImplementedError('function fit() not implemented.')

    def train(self):
        pass

    def val(self):
        pass
