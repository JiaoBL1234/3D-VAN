 # encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm
from collections import OrderedDict
import torch.nn.functional as F
from fastreid.modeling.losses import intra_instance_contrastive, inter_instance_contrastive
from .ops import My_bn_norm2d
from .sublayers import Attn_Block, Cross_Attn_Block
from einops import rearrange
from einops.layers.torch import Rearrange



logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, int(in_channels//reduction), kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((int(in_channels//reduction), 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(int(in_channels//reduction), num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x
        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = My_bn_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = My_bn_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = My_bn_norm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, flag=1):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, flag)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, flag)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, flag)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.downsample[1](residual, flag)

        out += residual
        out = self.relu(out)

        return out


class Resformer_Half(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers, with_sty_rand=False, selected_split_stage=[3,4], syn_num=1):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = My_bn_norm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.contrastive_stage = []
        self.selected_split_stage = selected_split_stage
        

        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        
        self.syn_num = syn_num
        
        self.with_sty_rand = with_sty_rand
        ### style random Not used ###
        if with_sty_rand:
            self.sty_random_1 = StyleRandomization(dim=512)
            self.sty_random_2 = StyleRandomization(dim=1024)
        
    
        self.trans_block_3 = Cross_Attn_Block(1024,
                                  1024,
                                  num_heads=4,
                                  mlp_ratio=1.,
                                  qkv_bias=False,
                                  drop=0.,
                                  attn_drop=0.,
                                  drop_path=0.,
                                  act_layer=nn.GELU,
                                  norm_layer=nn.LayerNorm,)
                                  
                                  
        self.trans_block_4 = Cross_Attn_Block(2048,
                                  2048,
                                  num_heads=4,
                                  mlp_ratio=1.,
                                  qkv_bias=False,
                                  drop=0.,
                                  attn_drop=0.,
                                  drop_path=0.,
                                  act_layer=nn.GELU,
                                  norm_layer=nn.LayerNorm,
                                  with_cls_token=True)
        self.IN_1 = nn.InstanceNorm2d(256)
        self.IN_2 = nn.InstanceNorm2d(512)
        self.IN_3 = nn.InstanceNorm2d(1024)
        self.IN_4 = nn.InstanceNorm2d(2048)                         
        self.distiller_1 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                     gate_activation='sigmoid', reduction=0.5, layer_norm=False)
        self.distiller_2 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                     gate_activation='sigmoid', reduction=0.5, layer_norm=False)
        self.distiller_3 = ChannelGate_sub(1024, num_gates=1024, return_gates=False,
                     gate_activation='sigmoid', reduction=0.5, layer_norm=False)
        self.distiller_4 = ChannelGate_sub(2048, num_gates=2048, return_gates=False,
                     gate_activation='sigmoid', reduction=0.5, layer_norm=False)

                     
        self.random_init()
        # fmt: off
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                My_bn_norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
    def content_feats_extraction(self, norm_layer, distiller, inputs):
        
        inputs_normed = norm_layer(inputs)
        inputs_style = inputs - inputs_normed
        inputs_style_useful, inputs_style_useless, weight = distiller(inputs_style)
        output = inputs_normed + inputs_style_useful
        return output
    def forward(self, x):
        
        
        x = self.conv1(x)
        x_real, x_syn = self.split_real_syn(x)
        x_real = self.bn1(x_real, 1)
        x_syn = self.bn1(x_syn, 2)
        x = self.combine_real_syn(x_real, x_syn)
        x = self.relu(x)
        x = self.maxpool(x)

        ##############split masks Not used#################
        # mask_real, mask_syn = self.split_real_syn(mask) #
        # mask_real = mask_real.float()                   #
        # mask_syn = mask_syn.float()                     #
        ###################split masks#####################

        # layer 1

        x_real, x_syn = self.split_real_syn(x)
        for i in range(len(self.layer1)):
            x_real = self.layer1[i](x_real, 1)     
            x_syn = self.layer1[i](x_syn, 2) 
        
        x_real = self.content_feats_extraction(self.IN_1, self.distiller_1, x_real)
        x_syn = self.content_feats_extraction(self.IN_1, self.distiller_1, x_syn) 
        for i in range(len(self.layer2)):
            x_real = self.layer2[i](x_real, 1)     
            x_syn = self.layer2[i](x_syn, 2)  
        
        x_real = self.content_feats_extraction(self.IN_2, self.distiller_2, x_real)
        x_syn = self.content_feats_extraction(self.IN_2, self.distiller_2, x_syn) 
        for i in range(len(self.layer3)):
            x_real = self.layer3[i](x_real, 1)     
            x_syn = self.layer3[i](x_syn, 2)
            
        x_real = self.content_feats_extraction(self.IN_3, self.distiller_3, x_real)
        x_syn = self.content_feats_extraction(self.IN_3, self.distiller_3, x_syn) 
        N, C, H, W = x_real.shape
        x_real = rearrange(x_real, 'b c h w -> b (h w) c')
        x_syn = rearrange(x_syn, 'b c h w -> b (h w) c')
        x_real, x_syn = self.trans_block_3(x_real, x_syn, H, W)
        x_real = rearrange(x_real, 'b (h w) c -> b c h w', h=H, w=W)
        x_syn = rearrange(x_syn, 'b (h w) c -> b c h w', h=H, w=W)
        
        
        for i in range(len(self.layer4)):
            x_real = self.layer4[i](x_real, 1)     
            x_syn = self.layer4[i](x_syn, 2) 
          
        x_real = self.content_feats_extraction(self.IN_4, self.distiller_4, x_real)
        x_syn = self.content_feats_extraction(self.IN_4, self.distiller_4, x_syn) 
        N, C, H, W = x_real.shape
        x_real = rearrange(x_real, 'b c h w -> b (h w) c')
        x_syn = rearrange(x_syn, 'b c h w -> b (h w) c')
        x_real, x_syn, fused_feats = self.trans_block_4(x_real, x_syn, H, W)
        x_real = rearrange(x_real, 'b (h w) c -> b c h w', h=H, w=W)
        x_syn = rearrange(x_syn, 'b (h w) c -> b c h w', h=H, w=W)
        fused_feats = fused_feats.view(N, C, 1, 1)
        
        
        if len(self.selected_split_stage)>0:
            if self.training==False and self.syn_num > 1:
                N,C,H,W = x_syn.shape
                x_syn = x_syn.view(int(N/self.syn_num), self.syn_num, C, H, W).mean(1)
            return x_real, x_syn, fused_feats
        else:
            return x_syn
    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
    def split_real_syn(self, inputs):
        if self.training:
            N, C, H, W = inputs.shape
            inputs_sp = inputs.view(int(N/4), 4, C, H, W)
            inputs_sp = inputs.view(int(N/8), 2, 4, C, H, W)
            
            
            inputs_real = inputs_sp[:,0,:,:,:,:].contiguous().view(int(N/2), C, H, W)
            inputs_syn = inputs_sp[:,1,:,:,:,:].contiguous().view(int(N/2), C, H, W)
            
            
        else:
            N, C, H, W = inputs.shape
            inputs_sp = inputs.view(int(N/(self.syn_num+1)), (self.syn_num+1), C, H, W)
            
            inputs_real = inputs_sp[:,0,:,:,:]
            inputs_syn = inputs_sp[:,1:,:,:,:].contiguous().view(-1, C, H, W)
        return inputs_real, inputs_syn
        
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
        
        
        
def init_pretrained_weights(key, selected_stage=[3,4]):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        logger.info(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
    
    
    state_dict_ = OrderedDict()
    for k, v in state_dict.items():
        if ('bn' in k or 'downsample.1' in k) and ('weight' in k):
            new_k = k.replace('weight', 'weight_1')
            state_dict_[k] = v
            state_dict_[new_k] = v
            new_k = k.replace('weight', 'weight_2')
            state_dict_[new_k] = v
            new_k = k.replace('weight', 'weight_3')
            state_dict_[new_k] = v
        elif ('bn' in k or 'downsample.1' in k) and ('bias' in k):   
            new_k = k.replace('bias', 'bias_1')
            state_dict_[k] = v
            state_dict_[new_k] = v
            new_k = k.replace('bias', 'bias_2')
            state_dict_[new_k] = v
            new_k = k.replace('bias', 'bias_3')
            state_dict_[new_k] = v
        elif ('bn' in k or 'downsample.1' in k) and ('running_mean' in k):   
            new_k = k.replace('running_mean', 'running_mean_1')
            state_dict_[k] = v
            state_dict_[new_k] = v
            new_k = k.replace('running_mean', 'running_mean_2')
            state_dict_[new_k] = v
            new_k = k.replace('running_mean', 'running_mean_3')
            state_dict_[new_k] = v
        elif ('bn' in k or 'downsample.1' in k) and ('running_var' in k):   
            new_k = k.replace('running_var', 'running_var_1')
            state_dict_[k] = v
            state_dict_[new_k] = v
            new_k = k.replace('running_var', 'running_var_2')
            state_dict_[new_k] = v
            new_k = k.replace('running_var', 'running_var_3')
            state_dict_[new_k] = v
        else:
            state_dict_[k] = v   
            
    '''
    for k, v in state_dict.items():
        if 2 in selected_stage and 'layer2' in k:
            new_k_real = k.split('layer2')[0] + 'layer2_real' + k.split('layer2')[1]
            new_k_syn = k.split('layer2')[0] + 'layer2_syn' + k.split('layer2')[1]
            state_dict_[new_k_real] = v
            state_dict_[new_k_syn] = v
        elif 3 in selected_stage and 'layer3' in k:
            new_k_real = k.split('layer3')[0] + 'layer3_real' + k.split('layer3')[1]
            new_k_syn = k.split('layer3')[0] + 'layer3_syn' + k.split('layer3')[1]
            state_dict_[new_k_real] = v
            state_dict_[new_k_syn] = v
        elif 4 in selected_stage and 'layer4' in k:
            new_k_real = k.split('layer4')[0] + 'layer4_real' + k.split('layer4')[1]
            new_k_syn = k.split('layer4')[0] + 'layer4_syn' + k.split('layer4')[1]
            state_dict_[new_k_real] = v
            state_dict_[new_k_syn] = v
        else:
            state_dict_[k] = v
    '''

    return state_dict_


@BACKBONE_REGISTRY.register()
def build_resformer_half_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    with_sty_rand = cfg.MODEL.BACKBONE.RANDOM
    selected_stage = cfg.MODEL.BACKBONE.SELECTED_STAGE
    syn_num       = cfg.TEST.SYN_NUM
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = Resformer_Half(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage, with_sty_rand=with_sty_rand, selected_split_stage=selected_stage, syn_num=syn_num)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key, selected_stage=selected_stage)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
