#!/usr/bin/python3
# -*- coding: utf-8 -*
import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image
from utils.tools import process_mask_tensor, process_binary_mask_tensor
# from datasets.make_dataset import make_dataset
import os
import numpy as np


class Mydataset(Dataset):

    CHANNELS_NUM = 3
    NUM_CLASSES = 3

    def __init__(self, mode, transform=None, target_transform=None, BASE_PATH=""):
        print(mode)
        self.items_image, self.items_mask = make_dataset(mode, BASE_PATH)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items_image)

    def __str__(self):
        return 'Mydataset'

    def __getitem__(self, index):
        # image_path = self.items[index]['image_path']
        # mask_path = self.items[index]['mask_path']
        image_path = self.items_image[index]
        mask_path = self.items_mask[index]

        image = Image.open(image_path)
        mask = cv2.imread(mask_path, 0)
        mask = mask / 125
        mask = mask.astype(np.uint8)
        # print(np.unique(mask))
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.uint8)
        if self.transform:
            image = self.transform(image)

        return image, mask


def make_dataset(mode, base_path):
    print(mode)

    # assert mode in ['train', 'val']
    #
    # path = os.path.join(base_path, mode)
    image_path = os.path.join(base_path, "image")
    mask_path = os.path.join(base_path, "mask")
    # print(image_path)
    image_list = []
    for file in os.listdir(image_path):
        image_list.append(os.path.join(image_path, file))

    mask_list = []
    for file in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, file))

    # print(image_list)
    return image_list, mask_list

