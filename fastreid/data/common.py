# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset

from .data_utils import read_image
import numpy
from PIL import Image, ImageFilter
import os
import random
import cv2
import numpy as np
import torch
from fastreid.data.transforms.functional import to_tensor
from .utils import background_construct, check_points
import torchvision.transforms.functional as TF
import PIL


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True, use_mask=False, syn_back=False, is_train=False):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.use_mask = use_mask
        self.syn_back = syn_back
        self.is_train = is_train
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])
        self.random_background = '/home/diskD/jbl/after-0415/dataset/Market-1501/background/'
    def __len__(self):
        if self.is_train==False:
            return len(self.img_items)
        if len(self.img_items[0])<4:
            return len(self.img_items)
        else:    
            counter = 0
            for i in self.img_items:
                is_real = i[3]
                if is_real:
                    counter+=1   
            return counter

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        
        output_dict = {}

        if self.transform is not None: 
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]        
        output_dict['images'] = img
        output_dict['targets'] = pid
        output_dict['camids'] = camid
        output_dict['img_paths'] = img_path
            
            
        return output_dict

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
