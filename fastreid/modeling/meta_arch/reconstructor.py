from easydict import EasyDict

from Texformer.transformers.texformer import Texformer
from Texformer.RSC_net.ra_test import RaRunner

from Texformer.NMR.neural_render_test import NrTextureRenderer
from Texformer.utils.scipy_deprecated import imresize
import matplotlib.pyplot as plt
import cv2
import os
from collections import defaultdict
from collections import OrderedDict
import Texformer.config as config
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import imageio
import pickle
class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.opts = EasyDict(src_ch=4, tgt_ch=3, feat_dim=128,
                             nhead=8, mask_fusion=1, out_ch=3,
                             checkpoint_path='Texformer/pretrained/ep050_camera_1_step000000_duke_ref.pt')
        self.checkpoint_path = self.opts.checkpoint_path
        
        self.model = Texformer(self.opts)
        
        
        state_dict = torch.load(self.checkpoint_path)
        state_dict_ = OrderedDict()
        for k, v in state_dict.items():
            new_k = k.split('module.')[1] if len(k.split('module.')) > 1 else k
            state_dict_[new_k] = v    
        self.model.load_state_dict(state_dict_) 
        #self.model.load_state_dict(torch.load(self.checkpoint_path))
        
        self.model.eval()
        self.model.to(self.device)

        self.tgt = torch.from_numpy(np.load(config.uv_encoding_path)).permute(2, 0, 1)[None]
        self.tgt = (self.tgt * 2 -1).float().to(self.device)

        # others
        self.rsc_runner = RaRunner()
        self.renderer_seg = NrTextureRenderer(224)
        self.renderer = NrTextureRenderer(224, factor=0.8)
        self.smpl_part_seg_mapping = torch.tensor([0, 3, 3, 1, 5, 5, 2, 4, 4, 6, 6, 7, 7]).long()
    
    def preprocess_img(self, img):
        img = img / 255.
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return img_tensor
    
    def get_smpl(self, img_tensor, given_pose=None, given_ori=None):
        scale = self.rsc_runner.get_scale(128)
        with torch.no_grad():
            # img_tensor is sent into GPU in runner
            pred_vertices, pred_cam_t, pred_rotmat, pred_betas = self.rsc_runner.get_3D_batch(img_tensor, scale, pre_process=True, given_pose=given_pose, given_oritation=given_ori)
        self.pred_vertices = pred_vertices
        self.pred_cam_t = pred_cam_t
        return pred_rotmat[:, 1:], pred_rotmat[:, 0]
    def get_segmentation(self):
        parts = self.renderer_seg.render_part(self.pred_vertices, self.pred_cam_t, crop_width=None)
        parts = parts.cpu().long()      
        parts = self.smpl_part_seg_mapping[parts]
        return parts
    
    def resize_224(self, img):
        h, w = img.shape[0:2]
        h_target = 224
        ratio = h_target / h 
        w_target = int(ratio * w)

        img = imresize(img, [h_target, w_target])

        diff = h_target - w_target
        before = abs(diff) // 2
        after = diff - before
        if diff > 0:
            # padding
            img = np.pad(img, [(0, 0), (before, after), (0, 0)])
        elif diff < 0:
            # crop
            img = img[:, before:- after]
        return img
    
    def preprocess_for_texture(self, img_tensor, parts):
        parts = parts.unsqueeze(1).float() / 7.0    # [1, 1, 224, 224], float, [0, 1]
        if len(parts.shape)>4:
            parts = parts[:,:,:,:,0]
        #print(parts.shape)
        h, w = parts.shape[2:]
        h_target = 128
        w_target = h_target * w / h 
        
        if h != 128:
            parts = torch.nn.functional.interpolate(parts, [int(h_target), int(w_target)], mode='nearest')
        if w_target > 64:
            start = int((w_target-64) // 2)
            parts = parts[:, :, :, start:start+64]
        elif w_target < 64:
            before = (64-w_target) // 2
            after = 64-w_target-before
            parts = torch.nn.functional.pad(parts, [before, after])
        parts = parts * 2 - 1
        parts = parts.to(self.device)
        
        img_tensor = img_tensor * 2 - 1
        img_tensor = torch.nn.functional.interpolate(img_tensor, [128, 128])
        img_tensor = img_tensor[:, :, :, 32:32+64].to(self.device)

        return img_tensor, parts
    
    def get_coord(self, shape):
        y = np.linspace(-1.0, 1.0, num=shape[0])
        x = np.linspace(-1.0, 1.0, num=shape[1])
        coord_y, coord_x = np.meshgrid(y, x, indexing='ij')
        coord = np.concatenate((coord_y[None], coord_x[None]), axis=0)
        return torch.from_numpy(coord).float()

    def get_texture(self, img, seg):
        coord = self.get_coord([128, 64]).unsqueeze(0).to(self.device)
        value = torch.cat([coord, img], dim=1)
        src = torch.cat([img, seg], dim=1)
        out = self.model(self.tgt, src, value)
        combine_mask = out[2]
        tex_flow = out[0]
        uvmap_flow = torch.nn.functional.grid_sample(img, tex_flow.permute(0, 2, 3, 1))
        uvmap_rgb = out[1]
        uvmap = uvmap_flow * combine_mask + uvmap_rgb * (1-combine_mask)
        return uvmap
    
    @torch.no_grad()
    def forward(self, img, can_view):
        rotation = can_view * 90
        N, C, H, W = img.shape
        with open('./std_posture.pkl', 'rb') as f:
            partition = pickle.load(f)
        std_pose = partition['std_pose']
        std_ori = partition['std_ori']
        std_cam = partition['std_cam']
        img = F.interpolate(img,(224,112))
        background = torch.zeros(N, C, 224, 224).cuda()
        background[:,:,:,56:168] = img
        img_ = background/255.
        
        
        sim_image = torch.zeros(N, C, 256, 128).cuda()
        for n in range(N):
            img = img_[n].unsqueeze(0)
            _, _ = self.get_smpl(img)
            parts = self.get_segmentation()
            
            img_text, parts = self.preprocess_for_texture(img, parts)
            uvmap = self.get_texture(img_text, parts)
            uvmap = (uvmap + 1) / 2
            uvmap = uvmap.clamp(0, 1)
            self.renderer.focal_length = 5000
            
            #img_tensor = self.preprocess_img(img_224)
            _, _ = self.get_smpl(img, given_pose=std_pose, given_ori=std_ori)
            rendered_img, _, parts = self.renderer.render(self.pred_vertices, std_cam, uvmap, euler=[0, rotation, 0])
            rendered_img = rendered_img[:,:,:,56:168]
            rendered_img = rendered_img*255
            sim_image[n] = F.interpolate(rendered_img,(256,128)).unsqueeze(0)
        #print(out)
        return sim_image