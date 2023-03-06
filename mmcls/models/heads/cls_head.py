import torch
import torch.nn.functional as F

from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict | list): Config of classification loss.  # TODO: ClsHead allow multi loss
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 cal_acc=True,
                 ):
        super(ClsHead, self).__init__()

        if isinstance(loss, dict):
            loss = [loss]
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        self.compute_loss = [build_loss(i) for i in loss]
        self.cal_acc = cal_acc
        if cal_acc:
            self.compute_accuracy = Accuracy(topk=self.topk)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        # TODO: export every loss to losses dict
        loss = sum([f(cls_score, gt_label, avg_factor=num_samples) for f in self.compute_loss])
        losses['ce_loss'] = loss
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, cls_score):
        """Test without augmentation."""
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
