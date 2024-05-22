import os
import sys
import logging
import argparse
import torch
import torch.backends.cudnn

from wrappers.basic_wrapper import BasicWrapper
# from trainers.proct_trainer import ProCTTrainer
# from trainers.proct_trainer_context import ProCTTrainerContext
from trainers.simple_tester import SimpleTester
from sources.proct import ProCT


os.environ['WANDB_MODE'] = 'online'

def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')
    # logging interval by iter
    parser.add_argument('--log_interval', type=int, default=500, help='logging interval by iteration')
    # tensorboard
    parser.add_argument('--use_tensorboard', action='store_true', default=False,)
    parser.add_argument('--tensorboard_root', type=str, default='', help='root path of tensorboard, project path')
    parser.add_argument('--tensorboard_dir', type=str, default='', help='detail folder of tensorboard')
    # wandb config
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_key', type=str, default='', help='your wandb api key')
    parser.add_argument('--wandb_project', type=str, default='iclseg-test')
    parser.add_argument('--wandb_root', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    
    # tqdm config
    parser.add_argument('--use_tqdm', action='store_true', default=False)

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for torch distributed training')
    
    # data_path
    parser.add_argument('--img_size', default=256, nargs='+', type=int, help='combined image size')
    parser.add_argument('--dataset_name', default='deeplesion', type=str, help='aapm, deeplesion, etc.')
    parser.add_argument('--train_json', default='', type=str, help='')
    parser.add_argument('--val_json', default='', type=str, help='')
    parser.add_argument('--num_train', default=10000, type=int, help='Number of training examples')
    parser.add_argument('--num_val', default=5000, type=int, help='Number of validation examples')

    # dataloader
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, action='store_true', help='dataloader droplast')
    
    # optimizer
    parser.add_argument('--accum_steps', default=1, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')    
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam beta1')    
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=60, type=int, help='number of training epochs')
    parser.add_argument('--warmup_steps', default=10, type=int, help='number of epochs for warmup')
    parser.add_argument('--clip_grad', default=-1, type=float, help='clipped value of gradient.')
    
    # scheduler
    parser.add_argument('--scheduler', default='', type=str, help='name of the scheduler')
    parser.add_argument('--step_size', default=10, type=int, help='step size for StepLR')
    parser.add_argument('--milestones', nargs='+', type=int, help='milestones for MultiStepLR')
    parser.add_argument('--step_gamma', default=0.5, type=float, help='learning rate reduction factor')
    parser.add_argument('--poly_iters', default=10, type=int, help='the number of steps that the scheduler decays the learning rate')
    parser.add_argument('--poly_power', default=0.9, type=float, help='the power of the polynomial')
    parser.add_argument('--min_lr', default=0., type=float)
    
    # checkpath && resume training
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    parser.add_argument('--save_epochs', default=1, type=int, help='interval of epochs for saving checkpoints')
    parser.add_argument('--save_net_only', default=False, action='store_true', help='only save the network param, discard the optimizer and scheduler')
    parser.add_argument('--load_net', default=False, action='store_true', help='load network param or not')
    parser.add_argument('--load_opt', default=False, action='store_true', help='load optimizer and scheduler or not')
    parser.add_argument('--net_checkpath', default='', type=str, help='checkpoint path for network param')
    parser.add_argument('--net_checkpath2', default='', type=str, help='checkpoint path for network param')
    parser.add_argument('--opt_checkpath', default='', type=str, help='checkpoint path for optimizer and scheduler param')
    parser.add_argument('--log_images', default=False, action='store_true')
    
    # network hyper args
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data (2D/3D)')
    parser.add_argument('--trainer_mode', default='train', type=str, help='main function - trainer mode, train or test')
    parser.add_argument('--loss_type_list', nargs='+', default='sml1', type=str, help='loss functions list')
    parser.add_argument('--network', default='', type=str, help='name of the network, e.g. unet.')
    parser.add_argument('--net_dict', default="{}", type=str, help='dictionary specifying network architecture.')
    
    # tester args
    parser.add_argument('--tester_save_dir', default='', type=str)
    parser.add_argument('--tester_save_name', default='', type=str)
    parser.add_argument('--tester_save_fig', default=False, action='store_true')
    parser.add_argument('--tester_save_tab', default=False, action='store_true')
    
    # CT setting
    parser.add_argument('--ct_task_list', nargs='+', default='sparse_view', type=str, help='CT tasks')
    parser.add_argument('--poisson_rate', default=1e6, type=float)
    parser.add_argument('--gaussian_rate', default=0.01, type=float)
    parser.add_argument('--clip_hu', default=False, action='store_true')
    parser.add_argument('--min_hu', default=-1024, type=int)
    parser.add_argument('--max_hu', default=3072, type=int)
    
    parser.add_argument('--support_size', default=0, type=int, help='Number of support images corresponding to each input image')
    parser.add_argument('--loss_weight', default=False, action='store_true')
    parser.add_argument('--loss2_factor', default=1, type=float)
    parser.add_argument('--prob_flip', default=0, type=float)
    parser.add_argument('--use_phantom', default=False, action='store_true')
    parser.add_argument('--net_dict2', default="{}", type=str, help='dictionary specifying network architecture.')
    return parser


class FBP(BasicWrapper):
    def __init__(self, **wrapper_kwargs):
        super().__init__(**wrapper_kwargs)
        
    def forward(self, *args, **kwargs):
        return args, kwargs
        

def run_main(opt):
    print('Image shape:', opt.img_size)
    print('Batch size:', opt.batch_size)
    print('Loss type list: ', opt.loss_type_list)
    print('Net dict: ', opt.net_dict)
    
    print('torch.__version__: ', torch.__version__)
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    print('torch.backends.cudnn.version(): ', torch.backends.cudnn.version())
    print('torch.cuda.device_count(): ', torch.cuda.device_count())
    
    net_name = opt.network
    trainer_mode = opt.trainer_mode
    print('Network name: ', net_name)
    print('Network param dict: ', opt.net_dict)
    
    poisson_rate = opt.poisson_rate
    gaussian_rate = opt.gaussian_rate
    wrapper_kwargs = dict(img_size=opt.img_size[0], simul_poisson_rate=poisson_rate, simul_gaussian_rate=gaussian_rate)
    
    # [network selection]
    net = None
    if net_name == 'fbp':
        net = FBP(**wrapper_kwargs)
    elif net_name in ['proct']:
        net_dict = dict()
        net_dict.update(eval(opt.net_dict))
        net = ProCT(opt.in_channels, 1, **net_dict, **wrapper_kwargs)
    else:
        raise NotImplementedError(f'Network not supported: {net_name}')
    
    
    # [training mode]
    if trainer_mode == 'train':
        trainer = ProCTTrainer(opt=opt, net=net)
        trainer.fit()
    elif trainer_mode == 'train_slow':
        trainer = ProCTTrainerContext(opt=opt, net=net)
        trainer.fit()
    elif trainer_mode == 'test':
        tester = SimpleTester(opt=opt, net=net)
        tester.run()
    else:
        x = torch.randn(opt.batch_size, opt.in_channels, opt.dataset_shape, opt.dataset_shape)
        y = net(x)
        print(y.shape)
        raise ValueError('opt trainer mode error: must be train or test, not {}'.format(opt.trainer_mode))

    print('finish')


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    run_main(opt)


