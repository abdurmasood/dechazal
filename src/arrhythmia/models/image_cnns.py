"""Layer specs of the two 2019 image CNNs, ported unchanged apart from 1-channel input.

Source: scripts/cnn_model.py createModel(). AlexNet-style follows the 2019 file as last
committed. NovelNet is taken from the commit before aff2afe, because aff2afe added a sixth
3x3 conv that would run on a 2x2 feature map and cannot be built.
"""

from arrhythmia.models.blocks import ConvSpec, DenseSpec, ImageCnnSpec

ALEXNET = ImageCnnSpec(
    name="alexnet",
    convs=(
        ConvSpec(96, 11, strides=4, pool=True),
        ConvSpec(256, 11, pool=True),
        ConvSpec(384, 3),
        ConvSpec(384, 3),
        ConvSpec(256, 3, pool=True),
    ),
    denses=(DenseSpec(4096, 0.4), DenseSpec(4096, 0.4), DenseSpec(1000, 0.4)),
)

NOVELNET = ImageCnnSpec(
    name="novelnet",
    convs=(
        ConvSpec(96, 13, strides=4, pool=True),
        ConvSpec(256, 11, pool=True),
        ConvSpec(384, 3),
        ConvSpec(434, 3),
        ConvSpec(500, 3, pool=True),
    ),
    denses=(DenseSpec(4096, 0.4), DenseSpec(4096, 0.6), DenseSpec(1000, 0.7)),
)
