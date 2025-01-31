from csse.utils.data import load_cifar100, load_class_info
from csse.utils.model import load_lop_resnet18
from csse.utils.evaluate import selected_class_accuracy

class LopExperiment:
    def __init__(self, algo, seed, session):
        self.algo=algo
        self.seed=seed
        self.session=session

        self.model = load_lop_resnet18(algo, seed, session)
        self.class_info = load_class_info(algo, seed, session)
        print(class_info)

        self.train_loader = load_cifar100(train=True)
        self.test_loader = load_cifar100(train=False)