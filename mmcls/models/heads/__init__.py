from .cls_head import ClsHead
from .linear_head import LinearClsHead, MultiLinearClsHead
from .cf_head import CFHead
from .cf_siam_head import CFSiamHead, CFTreeHead

__all__ = ['ClsHead', 'LinearClsHead', 'MultiLinearClsHead', 'CFHead', 'CFSiamHead', 'CFTreeHead'
]
