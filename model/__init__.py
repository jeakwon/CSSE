from csse.lop.nets.torchvision_modified_resnet import build_resnet18

import torch

def reconstruct_resnet18(weight_file_path: str):
    model = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    state_dict = torch.load(weight_file_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model