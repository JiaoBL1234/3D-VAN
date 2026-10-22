import torch
from torchvision.utils import save_image
import random
import time
import numpy as np
import cv2
import torch.nn as nn
import math
import random
import pdb
counter = 0
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    
def Normalize(content_feat, mean, std):  
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    '''
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    std = torch.tensor(std).view(1,-1,1,1).cuda()
    mean = torch.tensor(mean).view(1,-1,1,1).cuda()
    '''
    std = torch.tensor(std).view(1,-1,1,1).cuda()
    mean = torch.tensor(mean).view(1,-1,1,1).cuda()
    normalized_feat = (content_feat - mean.expand(size))/std.expand(size)  
    
    
    return normalized_feat   

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
    
def visualization(x_ori, x_trans):
    global counter
    N = x_trans.shape[0]

    im_mean = [0.486, 0.459, 0.408]
    im_std = [0.229, 0.224, 0.225]
    
    background = np.zeros((256,128*2,3))
    
    im_trans = x_trans.detach().cpu().numpy()  # C,H,W
    im_trans = im_trans.transpose(1,2,0)
    im_trans = im_trans * np.array(im_std).astype(float)
    im_trans = im_trans + np.array(im_mean)
    im_trans = im_trans * 255.
    im_trans = np.clip(im_trans,0,255)
    h, w, _ = im_trans.shape
    im_trans = im_trans[...,::-1]
    
    
    
    im_ori = x_ori.detach().cpu().numpy()  # C,H,W
    im_ori = im_ori.transpose(1,2,0)
    im_ori = im_ori * np.array(im_std).astype(float)
    im_ori = im_ori + np.array(im_mean)
    im_ori = im_ori * 255.
    im_ori = np.clip(im_ori,0,255)
    h, w, _ = im_ori.shape
    im_ori = im_ori[...,::-1]
    
    
    background[:,:128,:] = im_ori
    background[:,128:,:] = im_trans
    
    cv2.imwrite('./visualization/'+str(counter%50)+'.jpg', background)
    counter += 1
    
    return   
    
def transfer(img, network, is_train=False):
    network.train()

    N = img.shape[0]
    
    
    if is_train:
        N, C, H, W = img.shape
        img_sp = img.view(int(N/4), 4, C, H, W)
        img_sp = img.view(int(N/8), 2, 4, C, H, W)
        
        
        img_real = img_sp[:,0,:,:,:,:].contiguous().view(int(N/2), C, H, W)
        img_syn = img_sp[:,1,:,:,:,:].contiguous().view(int(N/2), C, H, W)
    else:
        N, C, H, W = img.shape
        img_sp = img.view(int(N/2), 2, C, H, W)
        img_real = img_sp[:,0,:,:,:]
        img_syn = img_sp[:,1,:,:,:]
    
    
    

    
    
    #pdb.set_trace()
    
    #loss_c, loss_s, output_syn = network(img_syn, img_real)
    loss, output_syn = network(img_syn, img_real)
    #print(output_syn)
    loss = loss.mean()
    #print(output_syn)
    
    '''
    background_real = np.zeros((256,128*4,3))
    img_real_ = output_syn.cpu().permute(0,2,3,1).contiguous()
    img_real_ = img_real_.permute(1,0,2,3).contiguous().view(H, int(N/2)*W, C).contiguous().data.numpy()
    print(img_real_.shape, type(img_real_))
    background_real += img_real_[:,128*4:128*8,:][:,:,::-1]
    cv2.imwrite('im_real.png', background_real)
    '''
    if np.random.randint(0,100)>50:
        visualization(img_real[0], output_syn[0])
    if is_train:
        output_syn = output_syn.view(int(N/8), 1, 4, C,H,W)
        img_real = img_real.view(int(N/8), 1, 4, C,H,W)
        output = torch.cat([img_real, output_syn], dim=1).view(int(N/4),4,C,H,W).view(N,C,H,W)
    else:
        output = torch.cat([img_real.unsqueeze(1), output_syn.unsqueeze(1)], dim=1).view(N,C,H,W)
    '''
    background_real = np.zeros((256,128*4,3))
    img_real_ = output.cpu().permute(0,2,3,1).contiguous()
    img_real_ = img_real_.permute(1,0,2,3).contiguous().view(H, N*W, C).data.numpy()
    background_real = img_real_[:,128*4:128*8,:][:,:,::-1]
    cv2.imwrite('im_real.png', background_real)
    
    background_syn = np.zeros((256,128*4,3))
    img_syn_ = output.cpu().permute(0,2,3,1).contiguous()
    img_syn_ = img_syn_.permute(1,0,2,3).contiguous().view(H, N*W, C).data.numpy()
    background_syn = img_syn_[:,128*4:128*8,:][:,:,::-1]
    cv2.imwrite('im_syn.png', background_syn)
    '''
    
    
    
    
    
    
    
    
    return loss, output
    
    
    
    
    
def random_erase(imgs, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
    N = imgs.shape[0]
    imgs_ = torch.zeros(imgs.shape).cuda()
    for i in range(N):
        img = imgs[i]
        if random.uniform(0, 1) >= probability:
            imgs_[i] += img
            continue
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1 / r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                imgs_[i] += img
                continue
        imgs_[i] += img    
    return imgs_


    
''' 
def transfer(img, vgg, decoder):
    decoder.eval()
    vgg.eval()

    N = img.shape[0]
    style = torch.zeros(img.shape).cuda()
    style[:int(N/2)] = img[int(N/2):]
    style[int(N/2):] = img[:int(N/2)]
    content = img

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style,
                                    1.0)
    index = random.randint(0,N-1) 
    output_c = output[index]
    visualization(output_c)
    return output
'''
'''
class random_erase(nn.Module):
    def __init__(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def foward(imgs):
        N = imgs.shape[0]
        imgs_ = torch.zeros(imgs.shape).cuda()
        for i in range(N):
            img = imgs[i]
            if random.uniform(0, 1) >= self.probability:
                imgs_[i] += img
                continue
            for attempt in range(100):
                area = img.size()[1] * img.size()[2]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    if img.size()[0] == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    imgs_[i] += img
                    continue
            imgs_[i] += img    
        return imgs_
''' 
    