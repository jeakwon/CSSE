
from tqdm import tqdm
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

    def train_accs(self, verbose=True):
        accs = []
        for i in tqdm(self.sessions, desc="Evaluating Train Accs", disable=not verbose):
            acc_dict = self.session(i).train_accs()
            acc_dict.update({'algo':self.algo, 'seed':self.seed, 'session':i, 'acc_type':'train'})
            accs.append(acc_dict)
        return accs

    def test_accs(self, verbose=True):
        accs = []
        for i in tqdm(self.sessions, desc="Evaluating Test Accs", disable=not verbose):
            acc_dict = self.session(i).test_accs()
            acc_dict.update({'algo':self.algo, 'seed':self.seed, 'session':i, 'acc_type':'test'})
            accs.append(acc_dict)
        return accs

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
        self.class_order = experiment.class_order
        self.train_loader = experiment.train_loader
        self.test_loader = experiment.test_loader

        self.state_dict = load_lop_resnet18_state_dict(self.algo, self.seed, self.session)
        self.class_info = parse_class_order(self.experiment.class_order, self.session, num_classes_per_session=5)
        self.all_classes = self.class_info['all_classes']
        self.old_classes = self.class_info['old_classes']
        self.new_classes = self.class_info['new_classes']

    def model(self, inplace=False):
        model = self.backbone
        model.load_state_dict(self.state_dict)
        model.to(self.device)
        if inplace:
            return model
        return deepcopy(model)

    def get_train_acc(self, selected_classes):
        return selected_class_accuracy(self.model(inplace=True), self.train_loader, selected_classes, self.device)

    def get_test_acc(self, selected_classes):
        return selected_class_accuracy(self.model(inplace=True), self.test_loader, selected_classes, self.device)

    def train_accs(self):
        return { k: self.get_train_acc(v) for k, v in self.class_info.items() }
    
    def test_accs(self):
        return { k: self.get_test_acc(v) for k, v in self.class_info.items() }