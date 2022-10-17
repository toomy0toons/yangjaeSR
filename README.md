# HAT-L 모델 


## Reference
- HAT [[Paper Link]](https://arxiv.org/abs/2205.04437)
- HAT [[Code Link]](https://github.com/XPixelGroup/HAT)
- RealEsrgan [[Code Link]](https://github.com/xinntao/Real-ESRGAN)
- HAT-Pretrained Model - check Training section
- AI-HUB Dataset [[link]](https://www.aihub.or.kr/)

## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md)

### Installation

1. Clone Repo
```
git clone https://github.com/toomy0toons/yangjaeSR.git
cd yangjaeSR
```
2. Install depedencies
```
pip install -r requirements.txt
python setup.py develop
```

3. Troubleshooting
`when using docker container, install CV2 dependencies`
```
apt-get install ffmpeg libsm6 libxext6 -y
```

## Dataset Preperation

### Data Preprocessing


1. Dacon Data [download](https://dacon.io/competitions/official/235977/data) and unzip to `data` folder

```
data/
|-- README.md
|-- yangjaeSR 
|   |-- sample_submission.zip
|   |-- test
|   |   |-- lr
|   |-- test.csv
|   |-- train
|   |   |-- hr
|   |   `-- lr
|   `-- train.csv
```

2. Unzip `sample_submission.zip` under `test/hr` folder
Inference 시 HR PAIR 가 있어야 작동하기 때문에, 더미로 사용합니다. 학습과 무관함.
```
data/yangjaeSR/test
|-- hr
|   |-- 20000.png
|   |-- 20001.png
|   |-- 20002.png
|   |-- 20003.png
|   |-- 20004.png
|   |-- 20005.png
|   |-- 20006.png
|   |-- 20007.png
|   |-- 20008.png
|   |-- 20009.png
|   |-- 20010.png
|   |-- 20011.png
|   |-- 20012.png
|   |-- 20013.png
|   |-- 20014.png
|   |-- 20015.png
|   |-- 20016.png
|   `-- 20017.png
`-- lr
    |-- 20000.png
    |-- 20001.png
    |-- 20002.png
    |-- 20003.png
    |-- 20004.png
    |-- 20005.png
    |-- 20006.png
    |-- 20007.png
    |-- 20008.png
    |-- 20009.png
    |-- 20010.png
    |-- 20011.png
    |-- 20012.png
    |-- 20013.png
    |-- 20014.png
    |-- 20015.png
    |-- 20016.png
    `-- 20017.png
```
3. run `scripts/resize_image.py`
```
yangjaeSR# python scripts/resize_image.py 
mkdir data/processed/train/hr ...
Extract:  11%|██████████████████▌                
```

### AIHUB DATA[Optional]
AI 허브에서 데이터를 다운로드합니다.
- [야외 실제 촬영 한글 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=105)
- [초해상화(Super Resolution 이미지)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=77)
- [노후 시설물 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=166)

`scripts/convert_png.py` 를 이용하여 aihub data 를 2048 로 리사이징 한뒤, 잘라냅니다. 

`data/aihub-list` 에서 사용된 데이터 목록을 볼 수 있습니다.

```
    # HR images
    opt['input_folder'] = 'aihub-sr/노후 시설물 이미지'
    opt['save_folder'] = 'aihub-processed/노후 시설물 이미지'
    opt['crop_size'] = 480
    opt['step'] = 360
    opt['thresh_size'] = 0
    extract_subimages(opt)

```


## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1kTieuWJGSmmuuOmXiDvKptsuhCywHs-C?usp=sharing)
- Download model to `experiments/pretrained_models/dacon_submission.pth`

- Then run the follwing codes to reproduce submission resulsts
```
python hat/test.py -opt options/test/yangjaeSR/HAT-L_Dacon_Submission.yml
```
The testing results will be saved in the `./results` folder.

### Submission

## Training
- refer to `options/train` for config for each model runs
- download imagenet pretrained HAT-L model at

1. Download Pretrained HAT-L Model to `experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth`
- The pretrained models are available at
[Google Drive](https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1u2r4Lc2_EEeQqra2-w85Xg) (access code: qyrl).  

2. Configure Opt
```
# in options/train/train--- yml

batch_size_per_gpu: 4 <--- change this number

```
- Batch Size 4 needs ~40GB Gpu RAM

3. Run Train code
- Set GPU IDs for distributed training
For example, distributed training in 4 GPU

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Coarse.yml \
--auto_resume --launcher pytorch
```
- In case of containerized server, set NCCL sockets
- run containeer with net=host and ipc=host 
```
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=en,eth,em,bond \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Coarse.yml \
--auto_resume --launcher pytorch
```

### 대회 세팅

Coarse -> Pretrain -> Submission 순서대로

1. Coarse 를 225000 iter 까지

```
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=en,eth,em,bond \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Coarse.yml \
--auto_resume --launcher pytorch
```

2. Pretrain 을 335000 iter 까지

```
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=en,eth,em,bond \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Pretrain.yml \
--auto_resume --launcher pytorch
```

3. Submission 185000 까지 (AIHUB 데이터 필요)

```
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=en,eth,em,bond \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 hat/train.py \
-opt options/train/yangjaeSR/train_HAT-L_Dacon_Submission.yml \
--auto_resume --launcher pytorch
```

4. The training logs and weights will be saved in the `./experiments` folder.

## Results
The inference results will be saved under `results/modelname/visualization` folder

## Contact
