import os
import numpy as np
import torch
import numpy
import PIL
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF



def background_construct(img, background, mask):
    mask = numpy.array(mask)
    mask = mask.sum(2)
    H, W = mask.shape
    mask = numpy.clip(mask,0,1).reshape(H, W, 1).astype(float)
    mask_t = numpy.abs(mask-1)
    
    
    #print(mask.max(), mask.min())
    img = numpy.array(img)
    img = img * mask
    img = img + background * mask_t
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img
    
    
def check_points(img, mask):
    img = TF.pad(img, 10)
    rand_x = np.random.randint(0,20)
    rand_y = np.random.randint(0,20)
    img = TF.crop(img, rand_x, rand_y, 256, 128)
    
    mask = mask.permute(2,0,1)
    mask = TF.resize(mask, (256,128), PIL.Image.NEAREST)
    #print(mask.shape)
    mask = mask.int()
    mask = TF.pad(mask, 10)
    mask = TF.crop(mask, rand_x, rand_y, 256, 128)
    
    if np.random.randint(0,100) < 50:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
        
        
    
    
    img_ = img.permute(1,2,0).cpu().data.numpy()
    mask_ = mask.permute(1,2,0).cpu().data.numpy().astype(float)
    
    if mask_.max()==6:
        img_ = img_[:,:,::-1]
        for i in range(1,7):
            idx = np.where(mask_==i)
            zeros_mask = np.zeros(mask_.shape)
            zeros_mask[idx] = 1
            
            
            background = np.zeros((256,128*2,3))
            background[:,:128,:] = img_
            img_1 = img_*zeros_mask
            background[:,128:,:] = img_1
            #import cv2
            #cv2.imwrite('./vis/'+str(i)+'.png', background)
    
        
    return img, mask
    