#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random

import cv2
import nibabel as nib
import glob
import numpy as np

def to255(img):

    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    img = img * 255  # *255
    return img

def NN_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    retimg = np.zeros((dstH, dstW), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg

src_path = r'F:\data\ProstateSeg\Task05_Prostate\imagesTr'  # mat文件的上级文件夹路径，英文路径
label_path = r'F:\data\ProstateSeg\Task05_Prostate\labelsTr'
save_path = r'F:\data\ProstateSeg\processed_data' # 保存png路径

train_image_path = save_path + '\\' + 'train\\' + 'image'
test_image_path = save_path + '\\' + 'val\\' + 'image'
train_mask_path = save_path + '\\' + 'train\\' + 'mask'
test_mask_path = save_path + '\\' + 'val\\' + 'mask'

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(test_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

names = glob.glob(src_path + '\\' + '*.nii')
random.shuffle(names)
order = 0

for name in names:
    this_name = name.split('\\')[-1][:-4]
    this_vol = nib.load(name).get_fdata()[:, :, :, 0]
    x, y, z = this_vol.shape
    this_vol = to255(this_vol)
    # 下面对mask——vol进行同步处理
    this_seg = nib.load(label_path + '\\' + this_name + '.nii').get_fdata()
    this_seg = this_seg.astype(np.uint8)
    print(np.unique(this_seg))
    this_seg[np.where(this_seg == 1)] = 125
    this_seg[np.where(this_seg == 2)] = 250
    for i in range(z):
        this_slice = this_vol[:, :, i]
        this_mask = this_seg[:, :, i]
        this_mask = NN_interpolation(this_mask, 256, 256)
        this_slice = cv2.resize(this_slice, (256, 256))
        if this_mask.max() > 0:
            if order % 5 == 0:
                cv2.imwrite(test_image_path + '\\' + this_name + '_' + str(i) + '.png', this_slice)
                cv2.imwrite(test_mask_path + '\\' + this_name + '_' + str(i) + '.png', this_mask)
            else:
                cv2.imwrite(train_image_path + '\\' + this_name + '_' + str(i) + '.png', this_slice)
                cv2.imwrite(train_mask_path + '\\' + this_name + '_' + str(i) + '.png', this_mask)
    order += 1
    print(order)







