
import torch
from copy import deepcopy

from csse.external_codes.lop.torchvision_modified_resnet import build_resnet18
from csse.utils.data import load_cifar100, load_class_order, parse_class_order
from csse.utils.model import load_lop_resnet18_state_dict
from csse.utils.evaluate import selected_class_accuracy

class Load_ResNet18_CIFAR100_CIL_Experiment:
    def __init__(self, algo, seed, sessions=range(21), device='cuda', batch_size=100, num_workers=2):
        self.algo = algo
        self.seed = seed
        self.device = device
        
        self.class_order = load_class_order(algo, seed)

        self.train_loader = load_cifar100(train=True, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = load_cifar100(train=False, batch_size=batch_size, num_workers=num_workers)

        self.backbone = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d).to(device)
        self.sessions = { session: Session(session=session, experiment=self) for session in sessions }

    def __getitem__(self, session):
        return self.sessions[session]
    
    def session(self, session):
        return self.sessions[session]

class Session:
    def __init__(self, session, experiment):
        self.session = session
        self.experiment = experiment

        self.algo = experiment.algo
        self.seed = experiment.seed
        self.device = experiment.device
        self.backbone = experiment.backbone
        self.train_loader = experiment.train_loader
        self.test_loader = experiment.test_loader

        self.state_dict = load_lop_resnet18_state_dict(self.algo, self.seed, self.session)
        self.class_info = parse_class_order(self.experiment.class_order, self.session, num_classes_per_session=5)
        self.trained_classes = self.class_info['trained_classes']
        self.previous_classes = self.class_info['previous_classes']
        self.recent_classes = self.class_info['recent_classes']
        self.unseen_classes = self.class_info['unseen_classes']

    def model(self, inplace=False):
        model = self.backbone
        model.load_state_dict(self.state_dict)
        if inplace:
            return model
        return deepcopy(model)

    # def train_acc(self):
    #     model = self.model(inplace=True)
    #     accuracies = {}
    #     for class_name, selected_classes in self.class_info.items():
    #         accuracies[class_name] = selected_class_accuracy(model, self.train_loader, selected_classes, self.device)
    #     return accuracies
    
    # def test_acc(self):
    #     model = self.model(inplace=True)
    #     accuracies = {}
    #     for class_name, selected_classes in self.class_info.items():
    #         accuracies[class_name] = selected_class_accuracy(model, self.test_loader, selected_classes, self.device)
    #     return accuracies
