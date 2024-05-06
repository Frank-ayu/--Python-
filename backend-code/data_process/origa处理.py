import cv2
import glob
import os
import numpy as np
import scipy.io as scio

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

save_path = r'F:\data\RetinaSeg\processed_data\origa' # 保存png路径

train_image_path = save_path + '\\' + 'train\\' + 'image'
test_image_path = save_path + '\\' + 'val\\' + 'image'
train_mask_path = save_path + '\\' + 'train\\' + 'mask'
test_mask_path = save_path + '\\' + 'val\\' + 'mask'

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(test_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

images = glob.glob(r'F:\data\RetinaSeg\ORIGA\650image\*.jpg')
masks = glob.glob(r'F:\data\RetinaSeg\ORIGA\650mask\*.mat')

order = 0
for name in masks:
    this_name = name.split('\\')[-1].split('.')[0]
    data = scio.loadmat(name)
    print(type(data))
    # 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
    mask = data['maskFull']
    mask = NN_interpolation(mask, 512, 512)
    mask = mask * 125
    # 读取对应的原图
    image = cv2.imread(name.replace('650mask', '650image').replace('mat', 'jpg'))
    image = cv2.resize(image, (512, 512))
    if order % 5 == 0: # 在这段代码中，它被用来确定每个文件是应该被分配到训练集还是测试集中。具体地说，如果 order 能被5整除，那么就将文件分配到测试集中，否则将文件分配到训练集中。
        cv2.imwrite(test_image_path + '\\' + this_name + '.png', image)
        cv2.imwrite(test_mask_path + '\\' + this_name + '.png', mask)
    else:
        cv2.imwrite(train_image_path + '\\' + this_name + '.png', image)
        cv2.imwrite(train_mask_path + '\\' + this_name + '.png', mask)
    order += 1





