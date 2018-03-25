#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/26 0:15
# @Author  : Wenbo Tang
# @File    : paint_log.py
import tensorboard
import re
from logger import Logger

dict_num = {
    0: 'epoch',
    2: 'lr',
    3: 'train_loss',
    4: 'train_acc',
    7: 'val_loss',
    8: 'val_acc'
}


def get_info(line):
    nums = re.findall(r"\d+\.?\d*", line)
    if len(nums) > 10:
        nums = [float(x) for x in nums]
        return nums
    else:
        return None


txt_name = 'coat_length_labels InceptionResnetV2'
txt_name2 = 'coat_length_labels googlenet'
infos = []
with  open(txt_name2) as txt:
    for line in txt:
        line = line.strip()
        nums = get_info(line)
        if nums == None:
            continue
        # print(nums)
        infos.append(nums)

        # print(infos)

logger = Logger('./logs/Inception V3')
step = 0
for line in infos:
    tmp_name = [dict_num[i] for i in dict_num.keys()]
    tmp_num = [line[i] for i in dict_num.keys()]
    info = dict(zip(tmp_name, tmp_num))
    print(info)
    for tag, value in info.items():
        # print(tag, value)
        logger.scalar_summary(tag, value, step + 1)
    step += 1

    # # (2) Log values and gradients of the parameters (histogram)
    # for tag, value in net.named_parameters():
    #     tag = tag.replace('.', '/')
    #     logger.histo_summary(tag, to_np(value), step + 1)
    #     logger.histo_summary(tag + '/grad', to_np(value.grad), step + 1)
    #
    # # (3) Log the images
    # info = {
    #     'images': to_np(images.view(-1, 28, 28)[:10])
    # }
    #
    # for tag, images in info.items():
    #     logger.image_summary(tag, images, step + 1)
