import torch
import torch.nn as nn
from .cross_entroy_loss import cross_entropy_loss
import torch.nn.functional as F
def intra_instance_contrastive(parts_feature_real, parts_feature_syn, head_num=4):
    N, C = parts_feature_syn.shape
    parts_feature_syn = parts_feature_syn.view(int(N/4), 4, C).mean(1)
    parts_feature_syn = parts_feature_syn.view(int(N/4), 1, C).expand(int(N/4), 4, C)
    
    parts_feature_syn = parts_feature_syn.contiguous().view(N, C)
    
    stride = int(C/head_num)
    loss = 0
    for i in range(head_num):
        real_feats = parts_feature_real[:,i*stride: (i+1)*stride].view(N, stride, 1)
        part_feats = parts_feature_syn.view(N, head_num, stride)
        part_feats = F.normalize(part_feats, dim=2)
        real_feats = F.normalize(real_feats, dim=1)
        #import pdb
        #pdb.set_trace()
        pred_score = torch.bmm(part_feats, real_feats).view(N, head_num)
        gt = (torch.zeros(N, dtype=torch.int64) + i).cuda()
        loss += cross_entropy_loss(pred_score, gt, 0.01)
    loss = loss/head_num  
    
    return loss    
        
        
    




def inter_instance_contrastive(parts_feature_real, parts_feature_syn, head_num=4):
    N, C = parts_feature_syn.shape
    
    parts_feature_syn = parts_feature_syn.view(int(N/4), 4, C).mean(1)
    parts_feature_syn = parts_feature_syn.view(int(N/4), C).permute(1,0)
    
    parts_feature_syn = F.normalize(parts_feature_syn, dim=0)
    parts_feature_real = F.normalize(parts_feature_real, dim=1)
    pred_score = torch.mm(parts_feature_real, parts_feature_syn)
    
    gt = torch.zeros(N, dtype=torch.int64).cuda()
    for i in range(N):
        gt[i]=int(i/head_num)
    loss = cross_entropy_loss(pred_score, gt, 0.01)
    #print(gt, loss)
    return loss