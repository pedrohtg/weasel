import torch
import numpy as np

from torch import nn
import torch.nn.functional as F

# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

# Implementation from: https://github.com/hubutui/DiceLoss-PyTorch
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def loss_dice(prd, msk, w, device):
    
    prd_lin = prd.permute(0, 2, 3, 1)
    prd_lin = torch.reshape(prd_lin, (-1, prd_lin.size(-1)))
    msk_lin = msk.view(-1)
    
    criterion = DiceLoss(weight=w, ignore_index=-1).to(device)
    one_hot_msks = F.one_hot(msk_lin[msk_lin != -1], num_classes=2)
    
    return criterion(prd_lin[msk_lin != -1, :], one_hot_msks)

def loss_sce(prd, msk, w):
    
    loss = F.cross_entropy(prd, msk, ignore_index=-1, weight=w, reduction='mean')
    
    return loss

def loss_fn(prd, msk, w, loss_name, device):
    
    loss = 0.0
    if '+' in loss_name:
        for ln in loss_name.split('+'):
            if ln == 'SCE':
                loss += loss_sce(prd, msk, w)
            elif ln == 'Dice':
                loss += loss_dice(prd, msk, w, device)
        loss /= len(loss_name.split('+'))
    elif loss_name == 'SCE':
        loss = loss_sce(prd, msk, w)
    elif loss_name == 'Dice':
        loss = loss_dice(prd, msk, w)
    
    return loss

def loss_weights(msk, device):
    
    weights = [torch.sum((msk == 0).int()),
               torch.sum((msk == 1).int())]
    if weights[0] == 0 or weights[1] == 0:
        weights = torch.FloatTensor([1.0, 1.0]).to(device)
    else:
        if weights[0] > weights[1]:
            weights = torch.FloatTensor([1.0, weights[0] / weights[1]]).to(device)
        else:
            weights = torch.FloatTensor([weights[1] / weights[0], 1.0]).to(device)
            
    return weights

# Implementation from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25,0.75], gamma=2, balance_index=-1, size_average=True):
        
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def focal_loss(lab, prd, num_classes=4):
    
    criterion = FocalLoss(num_class=num_classes, alpha=0.25, gamma=2.0, balance_index=2)
    return criterion(F.softmax(prd, dim=1), lab)