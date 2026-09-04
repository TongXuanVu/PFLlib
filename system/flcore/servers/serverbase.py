import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import (read_client_data, set_iov_task, set_iov_eval_cap,
                              set_iov_fed_dir, set_iov_eval_cumulative,
                              set_dataset, CAU_HINH_BO)
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        # PHAI goi truoc set_iov_fed_dir: set_dataset() dat lai _IOV_FED_DIR ve
        # mac dinh cua bo, roi -fed moi de len tren neu nguoi dung chi dinh.
        _bo = getattr(args, 'data_variant', 'can_iov')
        set_dataset(_bo)
        _n = CAU_HINH_BO[_bo]['num_classes']
        if args.num_classes != _n:
            raise SystemExit(
                f"[main] -ncl {args.num_classes} khong khop bo '{_bo}' "
                f"(can {_n}). Dung lai de tranh chay sai am tham.")
        set_iov_fed_dir(getattr(args, 'fed_dir', None) or CAU_HINH_BO[_bo]['fed_dir'])
        set_iov_task(getattr(args, 'task_id', 0))
        set_iov_eval_cumulative(getattr(args, 'eval_cumulative', False))
        set_iov_eval_cap(getattr(args, 'eval_sample_cap', 0))
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        if getattr(self.args, 'uniform_agg', False):
            # Per-FedAvg Alg. 1 lay trung binh DEU tren cac client duoc chon:
            # w = (1/|S|) * sum(w_i). PFLlib mac dinh theo kieu FedAvg, tuc la
            # trong so theo so mau. Giu mac dinh cu de PerAvg/FedAvg/FedProx
            # trong cung bo thi nghiem van so sanh duoc voi nhau.
            n = len(self.uploaded_weights)
            self.uploaded_weights = [1.0 / n] * n
        else:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def _model_dir(self):
        """Thu muc checkpoint. Co -tid thi tach rieng theo task, neu khong bon
        task chay noi tiep se ghi de len nhau: ten file chi co round nen
        PerAvg_server_round_5.pt cua task 2 bi task 3 xoa mat."""
        tid = getattr(self.args, 'task_id', 0)
        fed = getattr(self.args, 'fed_dir', 'federated_data')
        # kich ban: '' (full) | '_10shot' | '_fewshot'
        scen = fed.replace('federated_data', '')
        name = f"{self.dataset}{scen}_task{tid}" if tid else f"{self.dataset}{scen}"
        return os.path.join("models", name)

    def save_global_model(self, round_num=None):
        model_path = self._model_dir()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if round_num is not None:
            model_path = os.path.join(model_path, f"{self.algorithm}_server_round_{round_num}.pt")
        else:
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_init_checkpoint(self, path):
        """Nap trong so khoi tao tu checkpoint cua task TRUOC do, nhung danh so
        round lai tu dau. Day la cach noi cac task thanh mot mach
        class-incremental: task 2 bat dau tu mo hinh cuoi cua task 1."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"--init_checkpoint khong ton tai: {path}")
        # map_location: checkpoint duoc luu tu GPU, nap lai tren may chi co CPU
        # (hoac nguoc lai) se hong neu khong chi dinh thiet bi.
        try:
            self.global_model = torch.load(path, map_location=self.device,
                                           weights_only=False)
        except TypeError:
            self.global_model = torch.load(path, map_location=self.device)
        self.global_model.to(self.device)
        print(f"[init] bat dau tu checkpoint: {path}", flush=True)

    def load_model(self, round_num=None):
        model_path = self._model_dir()
        if round_num is not None:
            model_path = os.path.join(model_path, f"{self.algorithm}_server_round_{round_num}.pt")
        else:
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        # The checkpoint is a pickled nn.Module, not a state_dict. From torch
        # 2.6 onwards torch.load defaults to weights_only=True and refuses it,
        # which broke -mode resume. The file is written by this same script.
        try:
            self.global_model = torch.load(model_path, map_location=self.device,
                                           weights_only=False)
        except TypeError:  # torch < 1.13 has no weights_only argument
            self.global_model = torch.load(model_path, map_location=self.device)
        self.global_model.to(self.device)

    def model_exists(self):
        model_path = self._model_dir()
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_losses = []
        confusion_matrices = []
        for c in self.clients:
            ct, ns, loss, cm = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_losses.append(loss)
            num_samples.append(ns)
            confusion_matrices.append(cm)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_losses, confusion_matrices

    def evaluate(self, round_num=0):
        stats = self.test_metrics()
        
        # global metrics
        global_cm = np.sum(stats[4], axis=0)
        TP = np.diag(global_cm)
        FP = global_cm.sum(axis=0) - TP
        FN = global_cm.sum(axis=1) - TP
        
        micro_p = TP.sum() / (TP.sum() + FP.sum() + 1e-12)
        micro_r = TP.sum() / (TP.sum() + FN.sum() + 1e-12)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-12)
        
        # Chi trung binh tren cac lop THUC SU co mat trong tap test. Neu khong,
        # moi lop khong xuat hien deu dong gop F1 = 0 va keo macro-F1 xuong mot
        # cach vo nghia (vi du: chay 1 task 3 lop nhung chia cho 13).
        present = global_cm.sum(axis=1) > 0
        _p = (TP / (TP + FP + 1e-12))[present]
        _r = (TP / (TP + FN + 1e-12))[present]
        macro_p = np.mean(_p)
        macro_r = np.mean(_r)
        macro_f1 = np.mean(2 * _p * _r / (_p + _r + 1e-12))
        
        weights = global_cm.sum(axis=1)
        total = weights.sum()
        weighted_p = np.sum(weights * (TP / (TP + FP + 1e-12))) / total
        weighted_r = np.sum(weights * (TP / (TP + FN + 1e-12))) / total
        weighted_f1 = np.sum(weights * 2 * (TP / (TP + FP + 1e-12)) * (TP / (TP + FN + 1e-12)) / ((TP / (TP + FP + 1e-12)) + (TP / (TP + FN + 1e-12)) + 1e-12)) / total

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_loss = sum(stats[3])*1.0 / sum(stats[1])
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(test_loss)
        self.rs_test_auc.append(macro_f1)  # slot kept for schema compatibility

        print(f"Round {round_num} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        
        import pandas as pd
        csv_file = f"../results/{self.dataset}_{self.algorithm}_{self.goal}_metrics.csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy', 'Micro_P', 'Micro_R', 'Micro_F1', 'Macro_P', 'Macro_R', 'Macro_F1', 'Weighted_P', 'Weighted_R', 'Weighted_F1'])
            df.to_csv(csv_file, index=False)
            
        row = {'Round': round_num, 'Loss': test_loss, 'Accuracy': test_acc, 
               'Micro_P': micro_p, 'Micro_R': micro_r, 'Micro_F1': micro_f1,
               'Macro_P': macro_p, 'Macro_R': macro_r, 'Macro_F1': macro_f1,
               'Weighted_P': weighted_p, 'Weighted_R': weighted_r, 'Weighted_F1': weighted_f1}
        df = pd.DataFrame([row])
        df.to_csv(csv_file, mode='a', header=False, index=False)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
