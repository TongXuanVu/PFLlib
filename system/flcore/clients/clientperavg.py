import numpy as np
import torch
import time
import copy
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client


class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Per-FedAvg (Fallah et al., 2020, Alg. 1) dung HAI hoc suat: alpha cho
        # buoc thich nghi (step 1) va beta cho buoc meta (step 2). PFLlib upstream
        # comment mat args.beta va ep beta = alpha, khien -bt hoan toan vo tac dung.
        # Mac dinh -bt 0.0 van roi ve alpha, nen khong doi hanh vi cua cac lan
        # chay truoc; truyen -bt <gia tri> moi kich hoat beta rieng.
        self.beta = args.beta if getattr(args, 'beta', 0.0) > 0 else self.learning_rate

        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        # Reused buffer holding the parameters as they were before step 1.
        # The original code rebuilt this with copy.deepcopy on every batch.
        self._param_backup = None

    def _snapshot_params(self):
        params = list(self.model.parameters())
        if self._param_backup is None:
            self._param_backup = [p.detach().clone() for p in params]
        else:
            for buf, p in zip(self._param_backup, params):
                buf.copy_(p.data)

    def _restore_params(self):
        for p, buf in zip(self.model.parameters(), self._param_backup):
            p.data.copy_(buf)

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):  # local update
            for X, Y in trainloader:
                # Batch duoc nap voi co 2*batch_size roi xe doi: nua dau cho buoc
                # thich nghi, nua sau cho buoc meta. Batch cuoi (hoac shard
                # few-shot) co the ngan hon, nen cat theo do dai thuc te.
                # BatchNorm can it nhat 2 mau moi nua.
                half = min(self.batch_size, (X[0].shape[0] if type(X) == type([])
                                             else X.shape[0]) // 2)
                if half < 2:
                    continue
                # Was: temp_model = copy.deepcopy(list(self.model.parameters()))
                # A full model copy allocated once per batch. Copying into a
                # preallocated buffer gives the same values with no allocation.
                self._snapshot_params()

                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:half].to(self.device)
                    x[1] = X[1][:half]
                else:
                    x = X[:half].to(self.device)
                y = Y[:half].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][half:2 * half].to(self.device)
                    x[1] = X[1][half:2 * half]
                else:
                    x = X[half:2 * half].to(self.device)
                y = Y[half:2 * half].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                self._restore_params()

                self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        trainloader = self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        try:
            (x, y) = next(iter_loader)
        except StopIteration:
            # Shard rong hoac ngan hon mot batch: khong co gi de cap nhat.
            return
        if (x[0].shape[0] if type(x) == type([]) else x.shape[0]) < 2:
            return          # BatchNorm can it nhat 2 mau
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        output = self.model(x)
        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.cpu()


    def train_metrics(self, model=None):
        trainloader = self.load_train_data(self.batch_size*2)
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        for X, Y in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss1 = self.loss(output, y)

            train_num += y.shape[0]
            losses += loss1.item() * y.shape[0]

        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
