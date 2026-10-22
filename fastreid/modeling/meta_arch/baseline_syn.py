# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
import cv2
import numpy as np
from .net import vgg, decoder, Net
from .function import transfer
from .reconstructor import Reconstructor



counter = 0
@META_ARCH_REGISTRY.register()
class Baseline_SYN(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            cfg_,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self.cfg_ = cfg_
        ### AdaIN for style alignment, Not Used###
        
        
        self.Reconstructor = Reconstructor()
        self.syn_num = self.cfg_.TEST.SYN_NUM
        
        self.loss_kwargs = loss_kwargs
        
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                },
            'cfg_': cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, epoch=0):
        '''
        images_vis = batched_inputs['images'][:8]
        N, C, H, W = images_vis.shape
        #print(images_vis.shape)
        images_vis = images_vis.permute(2,0,3,1).contiguous()
        #print(images_vis.shape)
        images_vis = images_vis.view(H, N*W, C).cpu().data.numpy()[:,:,::-1]
        cv2.imwrite('vis.jpg', images_vis)
        '''
        
        targets = batched_inputs["targets"]
        images  = self.preprocess_image(batched_inputs)
        un_normed_image = images * self.pixel_std + self.pixel_mean
        # print(un_normed_image[0])
        if self.training:
            can_view = int(np.random.randint(0,4))
            can_images = self.Reconstructor(un_normed_image, can_view)
            can_images = can_images.sub_(self.pixel_mean/255).div_(self.pixel_std/255)
            N = images.shape[0]
            images = self.combine_real_syn(images, can_images)
            
            targets = self.combine_real_syn(targets.view(N,1,1,1), targets.view(N,1,1,1)).view(2*N)
            feats_real, feats_syn, feats_fused = self.backbone(images)
        else:
            feats_syn_cat = None
            for can_view in range(0,4):
                can_images = self.Reconstructor(un_normed_image, can_view)
                can_images = can_images.sub_(self.pixel_mean/255).div_(self.pixel_std/255)
                N = images.shape[0]
                images_ = self.combine_real_syn(images, can_images)
                feats_real, feats_syn, feats_fused = self.backbone(images_)
                if can_view == 0:
                    feats_syn_cat = feats_syn.unsqueeze(1)
                else:
                    feats_syn_cat = torch.cat([feats_syn_cat, feats_syn.unsqueeze(1)], dim=1)
            N, C, H, W = feats_real.shape
            feats_syn_cat = feats_syn_cat.view(N*4, C, H, W)
        
        
        
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"

            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(feats_real, feats_syn, feats_fused, targets)
            losses = self.losses(outputs, targets)
            ####not used####
            if self.cfg_.MODEL.BACKBONE.ADAIN:
                losses['loss_cons'] = loss_cons 
            return losses
        else:
            outputs = self.heads(feats_real, feats_syn_cat, feats_fused)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
        
        
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
    def combine_real_syn(self, real_feats, syn_feats):
    
        if self.training:
            
            N, C, H, W = real_feats.shape
            real_feats_ = real_feats.view(int(N/4), 4, C, H, W).unsqueeze(1)
            syn_feats_ = syn_feats.view(int(N/4), 4, C, H, W).unsqueeze(1)
            feats = torch.cat([real_feats_, syn_feats_],dim=1)
            feats = feats.view(int(N/2), 4, C, H, W).view(2*N, C, H, W)
            
        else:
            N, C, H, W = real_feats.shape
            real_feats_ = real_feats.unsqueeze(1)
            N, C, H, W = syn_feats.shape
            syn_feats_ = syn_feats.view(int(N/self.syn_num), self.syn_num, C, H, W)
            feats = torch.cat([real_feats_, syn_feats_], dim=1)
            feats = feats.contiguous().view(-1, C, H, W)
        return feats
    
    def split_gt_label(self,gt_labels):
        N = gt_labels.shape[0]
        gt_labels_ = gt_labels.view(int(N/4),4)
        gt_labels_ = gt_labels_.view(int(N/8),2,4)
        gt_labels_real = gt_labels_[:,0,:].contiguous().view(-1)
        gt_labels_syn = gt_labels_[:,1,:].contiguous().view(-1)
        return gt_labels_real, gt_labels_syn



        
    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        cls_fused       = outputs['cls_fused']
        pred_features_real     = outputs['features_real']
        pred_features_syn     = outputs['features_syn']
        
        # fmt: on
        gt_labels_real, gt_labels_syn = self.split_gt_label(gt_labels)
        
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') +\
            cross_entropy_loss(
                cls_fused,
                gt_labels_syn,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')
            
            
        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features_real,
                gt_labels_real,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') + \
            triplet_loss(pred_features_syn,
                gt_labels_syn,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')
            
            
        
        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict
