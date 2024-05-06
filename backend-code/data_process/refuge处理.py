import os

import cv2
import glob
import numpy as np

def NN_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    retimg = np.zeros((dstH, dstW), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg

names = glob.glob(r'F:\data\RetinaSeg\processed_data\reguge\*\*\*.*')
print(len(names))
order = 0
for name in names:
    if 'mask' in name:
        mask = cv2.imread(name, 0)
        mask = NN_interpolation(mask, 512, 512)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask[np.where(mask == 255)] = 0
        new_mask[np.where(mask == 0)] = 255
        new_mask[np.where(mask == 125)] = 125
        print(np.unique(new_mask))
        cv2.imwrite(name.replace('jpg', 'png').replace('bmp', 'png'), new_mask)
        os.remove(name)
    else:
        img = cv2.imread(name)
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(name.replace('jpg', 'png').replace('bmp', 'png'), img)
        os.remove(name)

    order += 1
    print(order)