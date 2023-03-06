import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class AdaRegLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
    
    def cal_weights(self, gt_labels):
        value, count = torch.unique(gt_labels, return_counts=True)
        results = dict()
        for v, c in zip(value, count):
            results[v.item()] = c.item()
        weights = []
        for i in range(self.num_classes):
            weights.append(1- results.get(i, 0) / len(gt_labels))
        return weights

    def forward(self, x, gt_label):
        """
        Args:
            x: averaged 256d feature (batch_size, 256).
            gt_label: ground truth labels(7 or 8 classes).
        """
        dist = torch.nn.functional.pdist(x, p=2)
        weights = self.cal_weights(gt_label)
        dist_weight = []
        for i in range(len(gt_label)):
            dist_weight +=  [weights[gt_label[i]]] * (len(gt_label) - i - 1)
        dist_weight = torch.tensor(dist_weight).cuda()
        dist = torch.mul(dist, dist_weight)
        loss = 1 / (10 * dist.sum())
        loss *= self.loss_weight

        return loss



