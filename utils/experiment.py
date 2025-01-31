from csse.utils.data import load_cifar100, load_class_info
from csse.utils.model import load_lop_resnet18
from csse.utils.evaluate import selected_class_accuracy

class Load_ResNet18_CIFAR100_CIL_Experiment:
    def __init__(self, algo, seed, session, device='cuda', batch_size=100, num_workers=2):
        self.algo=algo
        self.seed=seed
        self.session=session
        self.device=device

        self.model = load_lop_resnet18(algo, seed, session).to(device)
        self.class_info = load_class_info(algo, seed, session)
        self.train_loader = load_cifar100(train=True, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = load_cifar100(train=False, batch_size=batch_size, num_workers=num_workers)

        self.class_order = self.class_info['class_order']
        self.learned_classes = self.class_info['learned_classes']
        self.earlier_classes = self.class_info['earlier_classes']
        self.current_classes = self.class_info['current_classes']
        self.unknown_classes = self.class_info['unknown_classes']
    
    def eval_train_acc(self, selected_classes):
        return selected_class_accuracy(self.model, self.train_loader, selected_classes, self.device)
    
    def eval_test_acc(self, selected_classes):
        return selected_class_accuracy(self.model, self.test_loader, selected_classes, self.device)
    
    @property
    def train_acc(self):
        return dict(learned_class_acc = self.eval_train_acc(self.learned_classes),
                    earlier_class_acc = self.eval_train_acc(self.earlier_classes),
                    current_class_acc = self.eval_train_acc(self.current_classes),
                    unknown_class_acc = self.eval_train_acc(self.unknown_classes))

    @property
    def test_acc(self):
        return dict(learned_class_acc = self.eval_test_acc(self.learned_classes),
                    earlier_class_acc = self.eval_test_acc(self.earlier_classes),
                    current_class_acc = self.eval_test_acc(self.current_classes),
                    unknown_class_acc = self.eval_test_acc(self.unknown_classes))
