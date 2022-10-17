import cv2
import numpy as np
import os
import sys
import imutils

from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    Args:
        opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and
            longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.

            * DIV2K_train_HR
            * DIV2K_train_LR_bicubic/X2
            * DIV2K_train_LR_bicubic/X3
            * DIV2K_train_LR_bicubic/X4

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 128
    opt['compression_level'] = 9

    # HR images
    opt['input_folder'] = 'aihub-sr/노후 시설물 이미지'
    opt['save_folder'] = 'aihub-processed/노후 시설물 이미지'
    opt['crop_size'] = 480
    opt['step'] = 360
    opt['thresh_size'] = 0
    extract_subimages(opt)

def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, recursive=True, full_path=True, suffix=('.png', '.jpg', '.bmp','jpeg','JPG','JPEG')))

    for i in range((len(img_list)//1000)+1):
        os.makedirs(save_folder+'/'+str(i)+'000')
    os.makedirs(save_folder+'/'+str(i+1)+'000')

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for i,path in enumerate(img_list):
        opt['index']=str(i//1000)+'000'
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))
    index = opt['index']

    img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    scale = 4
    size_h, size_w, _ = img_gt.shape
    # sacle down to 2048x2048 max
    if size_h > 2048 or size_w > 2048:
        img_gt = imutils.resize(img_gt, width=2048, height=2048)

    size_h = size_h - size_h % scale
    size_w = size_w - size_w % scale
    img_gt = img_gt[0:size_h, 0:size_w, :]

    img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)

    result_path = [opt['save_folder'],str(index), f'{img_name}.png']
    #print(osp.join('',*result_path))
    cv2.imwrite(
                osp.join('',*result_path), img_gt,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])


    #del original jpeg file
    if osp.exists(path):
        #pass
        os.remove(path)
    else:
        print("The file does not exist")

    process_info = f'Processing {img_name} ...'
    return process_info

if __name__ == '__main__':
    main()
