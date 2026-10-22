import torch
import torch.nn.functional as F


def div_loss(score_list):
    loss = 0
    for score in score_list:
        loss += -(torch.std(score[:,0])+torch.std(score[:,1]))
    loss = loss/3
    
    return loss