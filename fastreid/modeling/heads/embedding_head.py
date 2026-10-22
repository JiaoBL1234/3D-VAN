# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead_SYN(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat
        
        
        
        ########################################
        ########## required to be changed ######
        ########################################
        
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_real = nn.Sequential(*neck)
        
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_syn = nn.Sequential(*neck)
        
        '''parts'''
        
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_real_parts = nn.Sequential(*neck)
        
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_syn_parts = nn.Sequential(*neck)
        
        
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck_fused = nn.Sequential(*neck)
        
        self.embedding_dim = embedding_dim
        '''parts'''

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        
        self.weight_real = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer_real = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.weight_syn = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer_syn = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        
        self.weight_fused = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer_fused = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        
        
        
        '''parts'''
        
        self.weight_real_parts = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer_real_parts = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.weight_syn_parts = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer_syn_parts = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        '''parts'''

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck_real.apply(weights_init_kaiming)
        self.bottleneck_real_parts.apply(weights_init_kaiming)
        self.bottleneck_syn.apply(weights_init_kaiming)
        self.bottleneck_syn_parts.apply(weights_init_kaiming)
        self.bottleneck_fused.apply(weights_init_kaiming)
        
        nn.init.normal_(self.weight_real, std=0.01)
        nn.init.normal_(self.weight_syn, std=0.01)
        nn.init.normal_(self.weight_fused, std=0.01)
        nn.init.normal_(self.weight_real_parts, std=0.01)
        nn.init.normal_(self.weight_syn_parts, std=0.01)
    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, feats_real, feats_syn, feats_fused, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        
        
        #print(feats_real.shape, feats_syn.shape)
        
        pool_feat_real = self.pool_layer(feats_real)
        neck_feat_real = self.bottleneck_real(pool_feat_real)
        neck_feat_real = neck_feat_real[..., 0, 0]

        pool_feat_syn = self.pool_layer(feats_syn)
        neck_feat_syn = self.bottleneck_syn(pool_feat_syn)
        neck_feat_syn = neck_feat_syn[..., 0, 0]
        
        pool_feat_fused = self.pool_layer(feats_fused)
        neck_feat_fused = self.bottleneck_fused(pool_feat_fused)
        neck_feat_fused = neck_feat_fused[..., 0, 0]
        
  
        
        # Evaluation
        # fmt: off
        #if not self.training: return torch.cat([neck_feat_real, neck_feat_syn], dim=1)
        #if not self.training: return torch.cat([neck_feat_real, neck_feat_syn], dim=1)
        #if not self.training: return torch.cat([neck_feat_real,neck_feat_syn], dim=1)
        #if not self.training: return neck_feat_fused
        if not self.training: 
            N = neck_feat_real.shape[0]
            neck_feat_syn = neck_feat_syn.view(N, 4, -1).view(N, -1)
            return torch.cat([neck_feat_real, neck_feat_syn], dim=1)
        #if not self.training: return neck_feat_fused
        #if not self.training: return neck_feat_real
        # fmt: on

        # Training
        if self.cls_layer_real.__class__.__name__ == 'Linear':
            logits_real = F.linear(neck_feat_real, self.weight_real)
            logits_syn = F.linear(neck_feat_syn, self.weight_syn)
            logits_fused = F.linear(neck_feat_fused, self.weight_fused)
        else:
            logits_real = F.linear(F.normalize(neck_feat_real), F.normalize(self.weight_real))
            logits_syn = F.linear(F.normalize(neck_feat_syn), F.normalize(self.weight_syn))
            logits_fused = F.linear(F.normalize(neck_feat_fused), F.normalize(self.weight_fused))
        # Pass logits.clone() into cls_layer, because there is in-place operations
        if isinstance(targets, torch.Tensor):
            N = targets.shape[0]
            targets_ = targets.view(int(N/2),2)
            targets_real = targets_[:,0]
            targets_syn = targets_[:,1]
            targets_fused = targets_syn
        else:
            targets_real = None
            targets_syn = None
            targets_fused = None
        cls_outputs_real = self.cls_layer_real(logits_real.clone(), targets_real)
        cls_outputs_syn = self.cls_layer_syn(logits_syn.clone(), targets_syn)
        cls_outputs_fused = self.cls_layer_fused(logits_fused.clone(), targets_fused)
        
        N = cls_outputs_real.shape[0]
        cls_outputs_real = cls_outputs_real.view(int(N/4), 4, -1).unsqueeze(1)
        cls_outputs_syn = cls_outputs_syn.view(int(N/4), 4, -1).unsqueeze(1)
        
        cls_outputs = torch.cat([cls_outputs_real, cls_outputs_syn], dim=1)
        cls_outputs = cls_outputs.view(int(N/2), 4, -1).view(2*N, -1)
        pred_class_logits = cls_outputs.detach()
        
        pool_feat = torch.cat([pool_feat_real.unsqueeze(1), pool_feat_syn.unsqueeze(1)], dim=1).view(N, -1, 1, 1)
        neck_feat = torch.cat([neck_feat_real.unsqueeze(1), neck_feat_syn.unsqueeze(1)], dim=1).view(N, -1)
        logits = torch.cat([logits_real.unsqueeze(1), logits_syn.unsqueeze(1)], dim=1).view(N, -1)
        
        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = neck_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on
        
        return {
            "cls_outputs": cls_outputs,
            "cls_fused": cls_outputs_fused,
            "pred_class_logits": pred_class_logits,
            "features_real": pool_feat_real[..., 0, 0],
            "features_syn": pool_feat_syn[..., 0, 0],
            
        }





@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)
        
        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)
    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, feats, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        
        
        #print(feats_real.shape, feats_syn.shape)
        
        pool_feat = self.pool_layer(feats)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]
        

        # Evaluation
        # fmt: off
        #if not self.training: return torch.cat([neck_feat_real, neck_feat_syn], dim=1)
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer_real.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)
        pred_class_logits = cls_outputs.detach()
        
        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": pool_feat[..., 0, 0],
        }