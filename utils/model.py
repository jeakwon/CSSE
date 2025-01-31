import torch

from csse.external_codes.lop.torchvision_modified_resnet import build_resnet18
from csse.utils.tools import load_state_dict

# Mapping of algorithm abbreviations to their full names
ALGORITHM = {
    'bp': 'base_deep_learning_system',
    'cb': 'continual_backpropagation',
    'hr': 'head_resetting',
    'rt': 'retrained_network',
    'sp': 'shrink_and_perturb',
}

def load_lop_resnet18_state_dict(algo, seed, session):
    algorithm = ALGORITHM[algo]
    file_name = f'index-{seed}_epoch-{session*200}.pt'
    url = f"https://huggingface.co/onlytojay/lop-resnet18/resolve/main/{algorithm}/model_parameters/{file_name}"
    state_dict = load_state_dict(url)
    return state_dict

def load_lop_resnet18(algo, seed, session):
    state_dict = load_lop_resnet18_state_dict(algo, seed, session)
    model = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    model.load_state_dict(state_dict)
    return model
