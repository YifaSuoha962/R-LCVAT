### 缺少的库
random
textdistance

### 指定设备
`export CUDA_VISIBLE_DEVICES=1`

### 训练命令
`python train.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE --beam_module OpenNMT --train_task prod2subs --loss_type Mixed-CE --lat_disc_size 90`