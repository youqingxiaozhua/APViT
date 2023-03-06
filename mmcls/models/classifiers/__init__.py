from .base import BaseClassifier
from .image import ImageClassifier, SiamImageClassifier, SiamImageClassifier
from ..vit.vit import VisionTransformer
# from ..vit.vit_vis import VisionTransformerVis
from .atten_vit import AttentionVitClassifier
from .pool_vit import PoolingVitClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'VisionTransformer', 'SiamImageClassifier',
# 'VisionTransformerVis',
'PoolingVitClassifier',
 'AttentionVitClassifier']
