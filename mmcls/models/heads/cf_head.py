import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init

from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class CFHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                coarse_num_classes=4,
                ada_weight=0.01,
                coarse_weight=3,
                fine_weight=5,
                 topk=(1, )):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.topk = topk
        self.compute_accuracy = Accuracy(topk=self.topk)
        
        self.coarse_num_classes = coarse_num_classes
        # build layers
        self.fc1f = nn.Linear(self.in_channels, 256)
        self.bn1f = nn.BatchNorm1d(256)
        self.fc2f = nn.Linear(256, num_classes)

        self.fc1c = nn.Linear(self.in_channels, 256)
        self.bn1c = nn.BatchNorm1d(256)
        self.fc2c = nn.Linear(256, coarse_num_classes)
        self.bn2c = nn.BatchNorm1d(coarse_num_classes)
        self.fc3c = nn.Linear(coarse_num_classes, num_classes)
        self.relu = nn.ReLU()
        # build losses
        self.ada_loss = build_loss(dict(type='AdaRegLoss', loss_weight=ada_weight))
        self.coarse_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=coarse_weight))
        self.fine_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=fine_weight))

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.fc, val=0, bias=0)
    
    def extract_feat(self, x1, x2):
        i1 = self.fc1f(x1)
        x = self.bn1f(i1)
        x = self.relu(x)
        k1 = self.fc2f(x)

        i2 = self.fc1c(x2)
        x = self.bn1c(i2)
        x = self.relu(x)
        j2 = self.fc2c(x)
        x = self.bn2c(j2)
        x = self.relu(x)
        k2 = self.fc3c(x)
        return dict(embedding_256=[i1, i2], coarse_score=j2, fine_score=[k1, k2])
        
    def simple_test(self, x1, x2):
        """Test without augmentation."""
        r = self.extract_feat(x1, x2)
        fine_score = (r['fine_score'][0] + r['fine_score'][1]) / 2

        fine_score = list(fine_score.detach().cpu().numpy())
        coarse_score = list(r['coarse_score'].detach().cpu().numpy())
        return coarse_score, fine_score

    def forward_train(self, x1, x2, gt_label, coarse_gt_label):
        r = self.extract_feat(x1, x2)
        embedding_256 = (r['embedding_256'][0] + r['embedding_256'][1]) / 2
        ada_loss = self.ada_loss(embedding_256, gt_label)
        coarse_loss = self.coarse_loss(F.softmax(r['coarse_score'], dim=1), coarse_gt_label)

        fine_score = (r['fine_score'][0] + r['fine_score'][1]) / 2
        fine_loss = self.fine_loss(F.softmax(fine_score, dim=1), gt_label)
        losses = dict(ada_loss=ada_loss, coarse_loss=coarse_loss, fine_loss=fine_loss)
        # compute accuracy
        coarse_acc = self.compute_accuracy(r['coarse_score'], coarse_gt_label)
        assert len(coarse_acc) == len(self.topk)
        losses['coarse_accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, coarse_acc)}

        fine_acc = self.compute_accuracy(fine_score, gt_label)
        assert len(fine_acc) == len(self.topk)
        losses['fine_acccuracy'] = {f'top-{k}': a for k, a in zip(self.topk, fine_acc)}

        return losses


