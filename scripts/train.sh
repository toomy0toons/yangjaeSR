NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=en,eth,em,bond \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Coarse.yml \
--auto_resume --launcher pytorch