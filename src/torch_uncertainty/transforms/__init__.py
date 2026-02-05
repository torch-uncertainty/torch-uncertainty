# ruff: noqa: F401
from .batch import MIMOBatchFormat, RepeatTarget
from .corruption import Brightness as BrightnessCorruption
from .corruption import Contrast as ContrastCorruption
from .corruption import (
    DefocusBlur,
    Elastic,
    Fog,
    Frost,
    GaussianNoise,
    GlassBlur,
    ImpulseNoise,
    JPEGCompression,
    MotionBlur,
    Pixelate,
    ShotNoise,
    Snow,
    ZoomBlur,
    corruption_transforms,
)
from .cutout import Cutout
from .image import (
    AutoContrast,
    Brightness,
    Color,
    Contrast,
    Equalize,
    Posterize,
    RandomRescale,
    Rotate,
    Sharpen,
    Shear,
    Solarize,
    Translate,
)
from .mixup import MIXUP_PARAMS, Mixup, MixupIO, MixupMP, RegMixup, WarpingMixup, build_mixup

augmentations = [
    AutoContrast,
    Equalize,
    Posterize,
    Rotate,
    Solarize,
    Shear,
    Translate,
    Contrast,
    Brightness,
    Color,
    Sharpen,
]
