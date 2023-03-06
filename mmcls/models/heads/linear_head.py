import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init

from ..builder import HEADS, build_head
from .cls_head import ClsHead
from .base_head import BaseHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=[dict(type='CrossEntropyLoss', loss_weight=1.0),],
                 topk=(1, ),
                 cal_acc=True,
                 ):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk, cal_acc=cal_acc)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.fc, val=0, bias=0)
    
    def extract_feat(self, img):
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score

    def simple_test(self, img):
        if isinstance(img, tuple):
            assert len(img) == 1
            img = img[0]
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        if os.environ.get('MODEL_VIS', '0') == '0':
            pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            assert len(x) == 1
            x = x[0]
        gt_index = gt_label != 255 if gt_label.dim() == 1 else gt_label[:, 0] != 255
        x = x[gt_index]
        gt_label = gt_label[gt_index]
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses


@HEADS.register_module()
class MultiLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=True):
        super().__init__(loss=loss, topk=topk, cal_acc=cal_acc)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.last_score = None

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.bn = nn.BatchNorm1d(self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.num_classes)
        self.relu = nn.ReLU()

    def init_weights(self):
        constant_init(self.fc1, val=0, bias=0)
        constant_init(self.fc2, val=0, bias=0)        
    
    def extract_feat(self, img):
        x = self.fc1(img)
        x = self.bn(x)
        x = self.relu(x)
        cls_score = self.fc2(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.extract_feat(img)
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        gt_index = gt_label != 255 if gt_label.dim() == 1 else gt_label[:, 0] != 255
        x = x[gt_index]
        gt_label = gt_label[gt_index]

        if len(x) >= 1:
            cls_score = self.extract_feat(x)
            losses = self.loss(cls_score, gt_label)
        else:
            # self.zero_grad()
            # cls_score = self.extract_feat(torch.zeros([8, self.in_channels], device=x.device)).detach()
            # losses = dict(ce_loss=torch.sum(cls_score)*0.)
            losses = dict()
        return losses


@HEADS.register_module()
class MultiTaskClsHead(BaseHead):
    """Head for process CE and AU in parallel."""
    def __init__(self, fc1, fc2, fc3=None):
        super().__init__()
        self.fc1 = build_head(fc1)
        self.fc2 = build_head(fc2)
        if fc3 is not None:
            self.fc3 = build_head(fc3)
    
    def forward_train(self, x, gt_label, au_label, va_label=None):
        losses = dict()
        # gt_index = gt_label!=255
        # gt_label = gt_label[gt_index]
        # ce_x = x[gt_index]
        ce_losses = self.fc1.forward_train(x, gt_label)
        losses.update(ce_losses)

        # au_index = au_label[:, 0] != 255
        # au_label = au_label[au_index]
        # au_x = x[au_index]
        # if au_label[0][0] != 255:
        au_losses = self.fc2.forward_train(x, au_label)
        if 'ce_loss' in au_losses:
            losses['au_loss'] = au_losses['ce_loss']
        if va_label is not None:
            av_loss = self.fc3.forward_train(x, va_label)
            if 'ce_loss' in av_loss:
                losses['av_loss'] = av_loss['ce_loss']
        return losses
    
    def simple_test(self, img):
        return self.fc1.simple_test(img)
    
    def aug_test(self, img):
        pred = self.fc1.extract_feat(img)
        pred = pred.detach().cpu()
        return pred

