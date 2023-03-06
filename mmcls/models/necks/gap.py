import torch
import torch.nn as nn

from ..builder import NECKS
from ..utils import top_pool


@NECKS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


@NECKS.register_module()
class GlobalAveragePoolingWithAttention(nn.Module):
    """Global Average Pooling neck with Attentive Pooling.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self, keep_num):
        super().__init__()
        self.k = keep_num
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward_one(self, x):
        assert isinstance(x, torch.Tensor)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(2, 1)  # [B, N, C]
        keep_indexes = top_pool(x, dim=C, keep_num=self.k, exclude_first=False, attn_method='ABS')
        if keep_indexes is not None:
            x = x.gather(dim=1, index=keep_indexes)
        x = x.transpose(1, 2)   # [B, C, k]
        x = nn.AdaptiveAvgPool1d(1)(x)      # [B, C, 1]
        return x.view(x.size(0), -1)


    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.forward_one(x) for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.forward_one(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
