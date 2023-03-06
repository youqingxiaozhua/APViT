import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init
import numpy as np

from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class CFSiamHead(BaseHead):

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
        # self.fc1f = nn.Linear(self.in_channels, 256)
        # self.bn1f = nn.BatchNorm1d(256)
        self.fc2f = nn.Linear(self.in_channels, num_classes)

        # self.fc1c = nn.Linear(self.in_channels, 256)
        # self.bn1c = nn.BatchNorm1d(256)
        self.fc2c = nn.Linear(self.in_channels, coarse_num_classes)

        # self.relu = nn.ReLU()
        # build losses
        # self.ada_loss = build_loss(dict(type='AdaRegLoss', loss_weight=ada_weight))
        self.coarse_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=coarse_weight))
        self.fine_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=fine_weight))

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.fc, val=0, bias=0)
    
    def extract_feat(self, input):
        # i1 = self.fc1f(input)
        # x = self.bn1f(i1)
        # x = self.relu(x)
        k1 = self.fc2f(input)

        # i2 = self.fc1c(input)
        # x = self.bn1c(i2)
        # x = self.relu(x)
        j2 = self.fc2c(input)
        return dict(
            # embedding_256=[i1, i2], 
            coarse_score=j2, fine_score=k1)
        
    def simple_test(self, x):
        """Test without augmentation."""
        r = self.extract_feat(x)
        # fine_score = (r['fine_score'][0] + r['fine_score'][1]) / 2
        fine_score = r['fine_score']
        fine_score = list(fine_score.detach().cpu().numpy())
        coarse_score = list(r['coarse_score'].detach().cpu().numpy())
        return coarse_score, fine_score

    def forward_train(self, x, gt_label, coarse_label):
        r = self.extract_feat(x)
        # embedding_256 = (r['embedding_256'][0] + r['embedding_256'][1]) / 2
        # ada_loss = self.ada_loss(embedding_256, gt_label)
        coarse_loss = self.coarse_loss(F.softmax(r['coarse_score'], dim=1), coarse_label)

        fine_score = r['fine_score']
        fine_loss = self.fine_loss(F.softmax(fine_score, dim=1), gt_label)
        losses = dict(
            # ada_loss=ada_loss, 
            coarse_loss=coarse_loss, fine_loss=fine_loss)
        # compute accuracy
        coarse_acc = self.compute_accuracy(r['coarse_score'], coarse_label)
        assert len(coarse_acc) == len(self.topk)
        losses['coarse_accuracy'] = {f'coarse_top-{k}': a for k, a in zip(self.topk, coarse_acc)}

        fine_acc = self.compute_accuracy(fine_score, gt_label)
        assert len(fine_acc) == len(self.topk)
        losses['fine_acccuracy'] = {f'fine_top-{k}': a for k, a in zip(self.topk, fine_acc)}

        return losses

@HEADS.register_module()
class CFTreeHead(BaseHead):
    """
    The first branch A cls to 4 coarse classes, 
    the second branch B cls to 4 negative classes.
    """

    def __init__(self,
                 in_channels,
                 neg_num_classes=4,
                coarse_num_classes=4,
                neg_weight=0.5,
                coarse_weight=0.5,
                 topk=(1, )):
        super().__init__()
        self.in_channels = in_channels
        self.neg_num_classes = neg_num_classes
        self.coarse_num_classes = coarse_num_classes


        if self.neg_num_classes <= 0 or self.coarse_num_classes <= 0:
            raise ValueError(
                f'num_classes={neg_num_classes} must be a positive integer')

        self.topk = topk
        self.compute_accuracy = Accuracy(topk=self.topk)
        
        # build layers
        self.fc1 = nn.Linear(self.in_channels, coarse_num_classes)
        self.fc2 = nn.Linear(self.in_channels, neg_num_classes)

        self.coarse_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=coarse_weight))
        self.neg_loss = build_loss(dict(type='CrossEntropyLoss', loss_weight=neg_weight))

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.fc, val=0, bias=0)
    
    def extract_feat(self, input):

        k1 = self.fc1(input)
        j2 = self.fc2(input)
        return dict(
            coarse_score=k1, neg_score=j2)
        
    def simple_test(self, x):
        """Test without augmentation."""
        r = self.extract_feat(x)
        coarse_score = r['coarse_score']
        neg_score = r['neg_score']
        coarse_score = coarse_score.detach().cpu().numpy()
        neg_score = neg_score.detach().cpu().numpy()
        results = np.array([coarse_score, neg_score]).transpose((1, 0, 2))
        return list(results)

    def forward_train(self, x, gt_label, coarse_label):
        r = self.extract_feat(x)
        coarse_score = r['coarse_score']
        coarse_loss = self.coarse_loss(F.softmax(coarse_score, dim=1), coarse_label)

        neg_index = coarse_label == 0
        neg_score = r['neg_score'][neg_index]
        neg_label = gt_label[neg_index]
        neg_loss = self.neg_loss(F.softmax(neg_score, dim=1), neg_label)

        losses = dict(
            coarse_loss=coarse_loss, 
            neg_loss=neg_loss
            )
        # compute accuracy
        coarse_acc = self.compute_accuracy(coarse_score, coarse_label)
        assert len(coarse_acc) == len(self.topk)
        losses['coarse_accuracy'] = {f'coarse_top-{k}': a for k, a in zip(self.topk, coarse_acc)}

        neg_acc = self.compute_accuracy(neg_score, neg_label)
        assert len(neg_acc) == len(self.topk)
        losses['neg_acccuracy'] = {f'neg_top-{k}': a for k, a in zip(self.topk, neg_acc)}

        return losses


