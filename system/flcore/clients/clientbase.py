import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, iov_test_view, iov_eval_weights
from utils.fast_loader import TensorBatches


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        # Evaluation has no gradients and no optimiser state, so it can use a
        # far larger batch than training. Keeping them separate stops the
        # training batch size (10 by default) from turning a 42M-row test pass
        # into 4.2M tiny kernel launches. Metrics are batch-size independent:
        # accuracy, summed loss and the confusion matrix all come out the same.
        self.test_batch_size = getattr(args, 'test_batch_size', 8192)
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if self.dataset == "IoV":
            x, y = train_data.tensors
            # Shard few-shot (10-shot, 1%) co the it hon mot batch. drop_last=True
            # khi do tra ve 0 batch va moi thu phia sau vo nghia, nen chi bo batch
            # cuoi khi con it nhat mot batch day.
            return TensorBatches(x, y, batch_size, shuffle=True,
                                 drop_last=x.shape[0] >= batch_size)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.test_batch_size
        if self.dataset == "IoV":
            # iov_test_view() loc tap test ve dung cac lop cua task dang chay.
            # No shuffle: the metrics are order-independent, and shuffling
            # would cost a 42M-element randperm per client per round.
            x, y = iov_test_view()
            return TensorBatches(x=x, y=y, batch_size=batch_size,
                                 shuffle=False, drop_last=False)
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        C = self.num_classes

        # Everything is accumulated on the model's device and read back once,
        # after the loop. The previous version called .item() twice and
        # sklearn.metrics.confusion_matrix once per batch: three CPU/GPU
        # synchronisations and a host round-trip of every prediction, per batch.
        # Khi tap test da bi lay mau, moi dong dai dien cho w[y] dong that.
        ew = iov_eval_weights() if self.dataset == "IoV" else None
        ew = ew.to(self.device) if ew is not None else None

        acc_t = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_t = torch.zeros((), dtype=torch.float64, device=self.device)
        num_t = torch.zeros((), dtype=torch.float64, device=self.device)
        # Flat (C*C + 1) histogram; index = true * C + pred. The extra last
        # bucket collects labels outside [0, C), which sklearn silently drops.
        cm_flat = torch.zeros(C * C + 1, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                w = torch.ones_like(y, dtype=torch.float64) if ew is None else ew[y]

                per = torch.nn.functional.cross_entropy(output, y, reduction='none')
                loss_t += (per.double() * w).sum()

                preds = torch.argmax(output, dim=1)
                acc_t += ((preds == y).double() * w).sum()
                num_t += w.sum()

                # Same layout as
                #   sklearn.metrics.confusion_matrix(y, preds, labels=arange(C))
                # rows = true label, cols = predicted label.
                in_range = (y >= 0) & (y < C)
                idx = torch.where(in_range, y * C + preds,
                                  torch.full_like(y, C * C))
                cm_flat += torch.bincount(idx, weights=w, minlength=C * C + 1)

        test_acc = float(acc_t.item())
        test_num = float(num_t.item())
        test_loss = float(loss_t.item())
        confusion_matrix = cm_flat[:C * C].reshape(C, C).cpu().numpy()

        return test_acc, test_num, test_loss, confusion_matrix

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
