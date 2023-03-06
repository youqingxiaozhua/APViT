from typing import List
import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(ImageClassifier, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        if pretrained:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(ImageClassifier, self).init_weights(pretrained)
        
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if isinstance(x, dict):
            loss = x['loss']
            x = x['x']
        else:
            loss = dict()
        if self.with_neck:
            x = self.neck(x)
        return x, loss

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, losses = self.extract_feat(img)

        loss = self.head.forward_train(x, gt_label=gt_label, **kwargs)
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x, _ = self.extract_feat(img)
        return self.head.simple_test(x)
    
    def inference(self, img):
        x, _ = self.extract_feat(img)
        x = self.head.extract_feat(x)
        return x
    
    def aug_test(self, imgs, **kwargs): # TODO: pull request: add aug test to mmcls
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred


@CLASSIFIERS.register_module()
class SiamImageClassifier(BaseClassifier):

    def __init__(self, backbone: List[dict], neck=None, head=None, pretrained=None):
        super().__init__()
        self.backbone1 = build_backbone(backbone[0])
        if len(backbone) == 1:
            self.backbone2 = self.backbon1
        else:
            self.backbone2 = build_backbone(backbone[1])

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x1, x2 = self.backbone1(img), self.backbone2(img)
        if self.with_neck:
            x1, x2 = self.neck(x1), self.neck(x2)
        return x1, x2

    def forward_train(self, img, gt_label, coarse_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x1, x2 = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x1, x2, gt_label, coarse_label)
        losses.update(loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x1, x2 = self.extract_feat(img)
        return self.head.simple_test(x1, x2)
    
    def inference(self, img):
        raise NotImplementedError
        x = self.extract_feat(img)
        x = self.head.extract_feat(x)
        return x
    
    def aug_test(self, imgs, **kwargs): # TODO: pull request: add aug test to mmcls
        raise NotImplementedError
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred
