from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import LOSSES
from .utils import weight_reduce_loss


# Varies Diverse Loss
# input x should be a list of Tensor of [batch, features], features could have any dimension


@LOSSES.register_module()
class DiverseCosineLoss(nn.Module):
    def __init__(self, loss_weight=1.0, margin=0.2, z_bias=1e-6, temperature=1., simple=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.margin = margin
        self.z_bias = z_bias
        self.simple = simple
        self.temperature = temperature
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def wang_forward(self, x):
        loss = 0.0
        num = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                x0 = torch.flatten(x[i], start_dim=1)
                x1 = torch.flatten(x[j], start_dim=1)

                dot = torch.sum(x0*x1, 1)
                D1 = torch.sqrt(torch.sum(torch.pow(x0,2), 1))
                D2 = torch.sqrt(torch.sum(torch.pow(x1,2), 1))
                
                t = dot * (D1*D2 + self.z_bias)**(-1)
                dist = torch.clamp(t, min=self.margin)

                loss = loss + torch.sum(dist) / x0.shape[0]
                num = num + 1
        return loss / num * self.loss_weight
    
    def xue_forward(self, x):
        loss_sum = 0.0
        num = 0
        if len(x) <= 1:
            return torch.tensor(0.).cuda()
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                p = torch.flatten(x[i], start_dim=1)
                q = torch.flatten(x[j], start_dim=1)
                p = F.normalize(p, dim=1)  # l2-normalize
                q = F.normalize(q, dim=1)  # l2-normalize

                loss = p * q
                loss = torch.exp(loss / self.temperature) - 1
                
                # loss = loss.sum(dim=1).mean()
                loss = torch.clamp(loss, min=self.margin)
                loss = loss.mean()

                # version2 
                # loss = torch.log(loss)
                # loss = F.cosine_embedding_loss(p, q, torch.ones(p.shape[0]).cuda(), margin=self.margin, reduction='mean')
                # loss = self.cos(p, q)

                loss_sum = loss_sum + loss
                num += 1
        return loss_sum / num * self.loss_weight

    def forward(self, x):
        if self.simple:
            return self.xue_forward(x)
        return self.wang_forward(x)


@LOSSES.register_module()
class DiverseEuclidLoss(torch.nn.Module):
    def __init__(self, margin=1.0, loss_weight=1.0):
        super().__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, x:List):
        """
        x: numbers of attention maps, shape(H, W)
        [Tensor((B, C, H, W)), ]
        """
        loss = 0.0
        num = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                x0 = torch.squeeze(x[i])
                x1 = torch.squeeze(x[j])
                # euclidian distance
                diff = x0 - x1
                dist_sq = torch.sum(torch.sum(torch.pow(diff, 2), 1),1)
                dist = torch.sqrt(dist_sq) / (x0.size()[2] * x0.size()[1])

                mdist = self.margin - dist
                dist = torch.clamp(mdist, min=0.0)
                loss += torch.sum(dist) / 2.0 / x0.size()[0]    #TODO: loss function
                num += 1
        loss = loss / num
        return loss * self.loss_weight


@LOSSES.register_module()
class KLDivLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x:List):
        loss_sum = 0.0
        num = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                p = torch.flatten(x[i], start_dim=1)
                q = torch.flatten(x[j], start_dim=1)
                p = torch.softmax(p, dim=1)
                q = torch.softmax(q, dim=1)

                loss = self.criterion(torch.log(p), q)
                loss_sum -= loss
                num += 1
        if num == 0:
            return torch.tensor(0.).cuda()
        loss = loss_sum / num
        return loss * self.loss_weight


@LOSSES.register_module()
class SparceLoss(nn.Module):
    
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, x:List):
        loss3 = 0.0
        for i in range(len(x)):
            mask = x[i].squeeze()   # [H, W]
            loss3 += mask.pow(2).sum(1).mean()
        loss3 = loss3/len(x)
        return loss3 * self.loss_weight
