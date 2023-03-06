import pytest
import unittest
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import _BatchNorm

# from mmcls.models.backbones import ResNet, ResNetV1d
# from mmcls.models.backbones.resnet import (BasicBlock, Bottleneck, ResLayer,
#                                            get_expansion)
from mmcls.models.vit.vit import HeadDropOut


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck)):
        return True
    return False


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class ModelTest(unittest.TestCase):
    def test_head_drop(self):
        model = HeadDropOut(p=1.)
        x = torch.rand((2, 1, 3, 8, 4))   # 8 heads
        train_out = model(x)
        count = 0
        for i in range(8):
            s = torch.sum(train_out[:,:,:,i,...])
            print(s)
            if s == 0:
                count += 1
        print('-----eval----')
        count = 0
        model.eval()
        x = torch.rand((2, 1, 3, 8, 4))   # 8 heads
        train_out = model(x)
        for i in range(8):
            s = torch.sum(train_out[:,:,:,i,...])
            print(s)
            if s == 0:
                count += 1
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()

