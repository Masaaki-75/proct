# Warmup stage
net_name="proct"
lr=2e-4
net_dict="dict(use_rope=False,block_kwargs={'norm_type':'INSTANCE'},drop_path_rates=0.1,use_spectrals=[True,True,True,False,False],use_learnable_prompt=False,num_heads=[2,4,6,1,1],attn_ratio=[0,1/2,1,0,0])"
batch_size=2
support_size=1
epochs=60
param_name=${net_name}_warm
train_json="./datasets/dl_train.txt"
val_json="./datasets/dl_test.txt"
res_dir="./res"
tensorboard_root=${res_dir}"/tb"
checkpoint_root=${res_dir}"/ckpt"

CUDA_VISIBLE_DEVICES='3' python -m torch.distributed.launch \
--master_port 19999 --nproc_per_node 1 \
main.py --epochs ${epochs} \
--lr ${lr} --optimizer 'adam' --prob_flip 0.5 \
--use_phantom --warmup_steps 10000 --loss_weight \
--scheduler 'mstep' --milestones 10 20 30 40 50 --step_gamma 0.5 \
--dataset_name 'deeplesion' --num_train 10000 --num_val 1000 \
--train_json ${train_json} --val_json ${val_json} \
--ct_task_list 'sparse_view_18_36_72_144' 'limited_angle_90_120_150' \
--poisson_rate 1e6 --gaussian_rate 0.01 \
--clip_hu --min_hu -1024 --max_hu 3072 \
--loss_type_list 'l1' \
--img_size 256 256 --spatial_dims 2 \
--trainer_mode 'train' --network ${net_name} --net_dict ${net_dict} \
--wandb_root ${tensorboard_root} --wandb_dir ${param_name} \
--checkpoint_root ${checkpoint_root} --checkpoint_dir ${param_name} \
--batch_size ${batch_size} --support_size ${support_size} \
--num_workers 4 --log_interval 500 \
--save_epochs 1 --use_tqdm


# Nomral stage
net_name="proct"
lr=2e-4
net_dict="dict(use_rope=False,block_kwargs={'norm_type':'INSTANCE'},drop_path_rates=0.1,use_spectrals=[True,True,True,False,False],use_learnable_prompt=False,num_heads=[2,4,6,1,1],attn_ratio=[0,1/2,1,0,0])"
batch_size=2
support_size=1
epochs=60
param_name=${net_name}
train_json="./datasets/dl_train.txt"
val_json="./datasets/dl_test.txt"
res_dir="./res"
tensorboard_root=${res_dir}"/tb"
checkpoint_root=${res_dir}"/ckpt"
net_checkpath="/path/to/warmup/checkpoint"

CUDA_VISIBLE_DEVICES='3' python -m torch.distributed.launch \
--master_port 18889 --nproc_per_node 1 \
main.py --epochs ${epochs} \
--lr ${lr} --optimizer 'adam' --prob_flip 0.5 \
--use_phantom --warmup_steps -1 \
--scheduler 'mstep' --milestones 10 20 30 40 50 --step_gamma 0.5 \
--dataset_name 'deeplesion' --num_train 10000 --num_val 1000 \
--train_json ${train_json} --val_json ${val_json} \
--ct_task_list 'sparse_view_18_36_72_144' 'limited_angle_90_120_150' \
--poisson_rate 1e6 --gaussian_rate 0.01 \
--clip_hu --min_hu -1024 --max_hu 3072 \
--loss_type_list 'l1' 'msssim' --loss2_factor 0.1 \
--img_size 256 256 --spatial_dims 2 \
--trainer_mode 'train' --network ${net_name} --net_dict ${net_dict} \
--wandb_root ${tensorboard_root} --wandb_dir ${param_name} \
--checkpoint_root ${checkpoint_root} --checkpoint_dir ${param_name} \
--batch_size ${batch_size} --support_size ${support_size} \
--num_workers 4 --log_interval 500 \
--save_epochs 1 --use_tqdm \
--load_net --net_checkpath $net_checkpath