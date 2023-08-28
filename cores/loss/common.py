import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

def dice_loss_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.01

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat, iflat), weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))
    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.

    # print('common dice function', 
    #         'min', torch.min(pred).item(), 
    #         'max', torch.max(pred).item(),
    #         'label', torch.min(target).item(), 
    # )
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_power_weights(pred, target, weights, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)
    
    p_k_alpha = torch.pow(iflat+delta, alpha)

    intersection = 2. * torch.sum(torch.mul(torch.mul(p_k_alpha, tflat),weights_flat))

    A_sum = torch.sum(torch.mul(torch.mul(p_k_alpha, p_k_alpha),weights_flat))
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_power(pred, target, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    p_k_alpha = torch.pow(iflat+delta, alpha)

    intersection = 2. * torch.sum(torch.mul(p_k_alpha, tflat))

    A_sum = torch.sum(torch.mul(p_k_alpha, p_k_alpha))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
    
    return (intersection) / (A_sum + B_sum + 0.0001)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, include_background=True, 
                 smooth=1., sigmoid=True, softmax=False,
                 square=False):
        super(SoftDiceLoss, self).__init__()
        self.square = square
        self.do_bg = include_background
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.call = 0

    def forward(self, logits, labels):
        dices = []
        # sigmoid / softmax so that predicted map can be distributed in [0, 1]
        if self.sigmoid==True:
            logits = torch.sigmoid(logits)
        elif self.softmax==True:
            if len(logits.shape)==5:
                dim = 1
            elif len(logits.shape)==4:
                dim = 0            
            logits = torch.softmax(logits, dim=dim)
            
        if self.do_bg==False:
            logits = logits[:,1:]
        
        # print('common dice', self.sigmoid, 
        #         'min', torch.min(logits).item(), 
        #         'max', torch.max(logits).item(),
        # )
            
        if logits.shape[1]==labels.shape[1]:
            return dice_loss(logits, labels) 
        else:
            labels = torch.ceil(labels)
            for c in range(logits.shape[1]):
                logit = logits[:,c]
                label = torch.zeros_like(logit)
                label[labels[:,0]==c] = 1
                dices.append(dice_loss(logit, label))
            return torch.stack(dices).sum()/logits.shape[1] 
