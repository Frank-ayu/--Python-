# 该代码从original_test_image_demo文件夹中选取原始图像 用于第一个页面的preprocess模块
# 里进行预处理 存放路径为 preprocessed_image_demo
# 注意 以上都是单张处理 仅用于第一个和第二个单张预测的演示
# 第三个功能的批量处理 仍然使用processed_data 因为batch预测必须是处理好的 所以就使用已经处理好的图片

# 选取一张图片 然后处理 保存在preprocessed_image_demo下对应的文件夹

import os
import cv2
import glob
import scipy.io as scio
import numpy as np

def NN_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    retimg = np.zeros((dstH, dstW), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg


def preprocess_single_image(choosed_image_path, save_path):
    print(choosed_image_path)
    print(save_path)
    # todo extract dataset type from the path
    # 同时处理image and mask
    type = choosed_image_path.split('/')[3]
    if type == "refuge":
        img = cv2.imread(choosed_image_path)
        img = cv2.resize(img, (512, 512))
        # 保存到test下的image
        image_path = choosed_image_path.replace('jpg', 'png')
        image_name = image_path.split('/')[-1]
        save_path_image = save_path + "/refuge/test/image/" + image_name
        # 保存image
        cv2.imwrite(save_path_image, img)
        # 保存到test下的mask
        # 开始处理对应的mask文件
        choosed_mask_path = choosed_image_path.replace('/image', '/mask')
        choosed_mask_path = choosed_mask_path.replace('jpg', 'bmp')
        mask = cv2.imread(choosed_mask_path, 0)
        # 近邻插值缩小图片
        mask = NN_interpolation(mask, 512, 512)
        # 新建一个新的矩阵 用于填充 转一下颜色
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask[np.where(mask == 255)] = 0
        new_mask[np.where(mask == 0)] = 255
        new_mask[np.where(mask == 128)] = 125

        mask_path = choosed_mask_path.replace('bmp', 'png')
        mask_name = mask_path.split('/')[-1]
        save_path_mask = save_path + "/refuge/test/mask/" + mask_name
        print("预处理的图片mask的保存位置如下：")
        print(save_path_mask)
        cv2.imwrite(save_path_mask, new_mask)
    else:
        print("origa还没处理")
        # 根据路径用glob找出image和mask的集合
        # choosed_image_path + "//*."
        # images = glob.glob(r'F:\data\RetinaSeg\ORIGA\650image\*.jpg')
        # masks = glob.glob(r'F:\data\RetinaSeg\ORIGA\650mask\*.mat')
        img = cv2.imread(choosed_image_path)
        img_preprocessed = cv2.resize(img, (512, 512))
        # 保存到test下的image
        # 'D:/RetainSeg/original_test_image_demo/origa/test/image/AGLAIA_GT_001.jpg'
        mask_path = choosed_image_path.replace('/image','/mask').replace('jpg', 'mat')
        this_name = mask_path.split('/')[-1].split('.')[0]
        data = scio.loadmat(mask_path)
        # print(type(data))
        # 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
        mask = data['maskFull']
        mask = NN_interpolation(mask, 512, 512)
        mask_preprocessed = mask * 125

        save_path_image = save_path + "/origa/test/image/"
        save_path_mask = save_path + "/origa/test/mask/"

        cv2.imwrite(save_path_image + '\\' + this_name + '.png', img_preprocessed)
        cv2.imwrite(save_path_mask + '\\' + this_name + '.png', mask_preprocessed)

if __name__ == '__main__':
    # preprocess_single_image("D:\\RetainSeg\\original_test_image_demo\\refuge\\test\\image\\T0002.jpg", "D:\\RetainSeg\\preprocessed_image_demo")

    img1 = cv2.imread(r"D:\RetainSeg\preprocessed_image_demo\refuge\test\mask\T0002.png")
    img2 = cv2.imread(r"D:\RetainSeg\processed_data\reguge\test\mask\T0002.png")

    # 比较两张图片是否完全相同
    if img1.shape == img2.shape and not (img1 - img2).any():
        print("两张图片完全相同")
    else:
        print("两张图片不完全相同")
