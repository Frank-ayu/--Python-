# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*
from random import random

import torch
from torch.autograd import Variable
from models import unet
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
from utils.metrics import compute_metrics

# 权重地址
train_weights = r'F:\data\RetinaSeg\trainingrecords\checkpoint\refuge_unet_esa_grid_Lovasz\refuge_unet_esa_grid_Lovasz_best_epoch_23_0.932.pkl'

# 选择网络模型
# 模型声明
# net = unet.UNet(num_classes=3, in_channels=3, is_esa=False, is_grid=False)    # unet 加载模型
# net = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=False)    # unet_esa 加载模型
net = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=True)    # unet_esa_grid 加载模型
ckpt = torch.load(train_weights)
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    # name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[k] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(new_state_dict)
net.eval()

# pre_data_path = r'F:\data\RetinaSeg\processed_data\origa\val\image' # origia数据
pre_data_path = r'F:\data\RetinaSeg\processed_data\reguge\test\image' # reguge数据

dst_path = '\\'.join(train_weights.split('\\')[:-1]).replace('checkpoint', 'pred_val')
os.makedirs(dst_path, exist_ok=True)


image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))
# 验证
loss_all = []
predictions_all = []
labels_all = []

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("\\")[-1]
        org_image = cv2.imread(image)
        show_org_mask = cv2.imread(image.replace('image', 'mask'))

        labels = cv2.imread(image.replace('image', 'mask'), 0)
        labels = labels / 125
        labels = labels.astype(np.uint8)
        # labels = org_mask.astype(np.uint8)
        # labels = np.zeros(org_mask.shape, np.uint8)
        # labels[org_mask == 125] = 1
        # labels[org_mask == 250] = 2
        # labels[org_mask == 255] = 2
        image = Image.open(image)
        ori_size = image.size
        image = transforms.ToTensor()(image)
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        outputs = net(image)    # # [1, 2, 224, 224], 此2应该为类别数
        if isinstance(outputs, list):
            # 若使用deep supervision，用最后一个输出来进行预测
            predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(np.int)
        else:
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = labels.astype(np.int)
        predictions_all.append(predictions)
        labels_all.append(labels)
        outputs = outputs.squeeze(0)    # [2, 224, 224]
        mask = torch.max(outputs, 0)[1].cpu().numpy()

        # 下面进行拼接展示,依次为原图，GTmask，预测的mask
        mask = mask * 125
        mask = np.expand_dims(mask, axis=2)
        pred_mask = np.concatenate((mask, mask, mask), axis=-1)

        cat_img = np.hstack([org_image, show_org_mask, pred_mask])
        cv2.imwrite(dst_path + '\\' + str(name), cat_img)
    # 使用混淆矩阵计算语义分割中的指标
    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(predictions_all, labels_all,
                                                                                   num_classes=3)
    print('Testing: MIoU: {:.4f}, MDSC: {:.4f}, MPC: {:.4f}, AC: {:.4f}, MSE: {:.4f}, MSP: {:.4f}, MF1: {:.4f}'.format(
        miou, mdsc, mpc, ac, mse, msp, mf1
    ))


