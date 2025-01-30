from importlib.metadata import version
__version__ = version("mypackage")

from .lop.nets.torchvision_modified_resnet import build_resnet18