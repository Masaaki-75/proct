# Evaluation on DeepLesion dataset
net_name="proct"
net_dict="dict(use_rope=False,block_kwargs={'norm_type':'INSTANCE'},drop_path_rates=0.1,use_spectrals=[True,True,True,False,False],use_learnable_prompt=False,num_heads=[2,4,6,1,1],attn_ratio=[0,1/2,1,0,0])"
support_size=1
train_json="./datasets/dl_train.txt"
val_json="./datasets/dl_test.txt"
tester_save_dir="./res"
tester_save_name="proct_v1"
net_checkpath="./ckpt/proct_v1.pkl"

CUDA_VISIBLE_DEVICES='7' python main.py \
--dataset_name 'deeplesion' --num_train 10000 --num_val 1000 \
--train_json ${train_json} --val_json ${val_json} \
--support_size ${support_size} --use_phantom \
--ct_task_list 'sparse_view_18_36_72_144' 'limited_angle_90_120_150' \
--poisson_rate 1e6 --gaussian_rate 0.01 \
--clip_hu --min_hu -1024 --max_hu 3072 \
--img_size 256 256 --spatial_dims 2 \
--trainer_mode 'test' --network ${net_name} --net_dict ${net_dict} \
--tester_save_dir ${tester_save_dir} --tester_save_name ${tester_save_name} \
--load_net --net_checkpath $net_checkpath \
--num_workers 4 --use_tqdm --tester_save_tab #--tester_save_fig


# Evaluation on AAPM dataset
net_name="proct"
net_dict="dict(use_rope=False,block_kwargs={'norm_type':'INSTANCE'},drop_path_rates=0.1,use_spectrals=[True,True,True,False,False],use_learnable_prompt=False,num_heads=[2,4,6,1,1],attn_ratio=[0,1/2,1,0,0])"
support_size=1
train_json="./datasets/dl_train.txt"
val_json="./datasets/aapm_test.txt"
tester_save_dir="./res"
tester_save_name="proct_v1"
net_checkpath="./ckpt/proct_v1.pkl"

CUDA_VISIBLE_DEVICES='7' python main.py \
--dataset_name 'aapm' --num_train 10000 --num_val 1145 \
--train_json ${train_json} --val_json ${val_json} \
--support_size ${support_size} --use_phantom \
--ct_task_list 'sparse_view_18_36_72_144' 'limited_angle_90_120_150' \
--poisson_rate 1e6 --gaussian_rate 0.01 \
--clip_hu --min_hu -1024 --max_hu 3072 \
--img_size 256 256 --spatial_dims 2 \
--trainer_mode 'test' --network ${net_name} --net_dict ${net_dict} \
--tester_save_dir ${tester_save_dir} --tester_save_name ${tester_save_name} \
--load_net --net_checkpath $net_checkpath \
--num_workers 4 --use_tqdm --tester_save_tab #--tester_save_fig
