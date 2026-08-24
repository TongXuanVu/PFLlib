import copy
import os
import time
import numpy as np
import pandas as pd
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from threading import Thread


class PerAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)

        # How many times client.train() runs per round. PFLlib upstream calls
        # it twice; -plp 1 halves the training cost per round and matches the
        # single local update loop described in the Per-FedAvg paper. Left at
        # 2 by default so existing runs stay comparable.
        self.local_passes = max(1, int(getattr(args, 'peravg_local_passes', 2)))
        # Write a server checkpoint every save_gap rounds (plus the last one).
        self.save_gap = max(1, int(getattr(args, 'save_gap', 1)))

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Local passes per round: {self.local_passes} | checkpoint every {self.save_gap} round(s)")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        start_round = 0
        if hasattr(self.args, 'mode') and self.args.mode in ['resume', 'test']:
            start_round = self.args.resume_round
            print(f"\nLoading model from round {start_round}...")
            self.load_model(start_round)

        if hasattr(self.args, 'mode') and self.args.mode == 'test':
            print(f"\n-------------Testing Round: {start_round}-------------")
            self.evaluate_one_step(round_num=start_round)
            return

        init = getattr(self.args, 'init_checkpoint', None)
        if init and self.args.mode == 'train':
            self.load_init_checkpoint(init)
        elif getattr(self.args, 'task_id', 0) > 1 and not init:
            print("[canh bao] task > 1 nhung khong co --init_checkpoint: chay tu "
                  "mo hinh ngau nhien, KHONG phai class-incremental.", flush=True)

        for i in range(start_round, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # send all parameter for clients
            self.send_models()

            eval_cost = 0.0
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                e_t = time.time()
                self.evaluate_one_step(round_num=i)
                eval_cost = time.time() - e_t

            # choose several clients to send back upated model to server
            t_t = time.time()
            for client in self.selected_clients:
                for _ in range(self.local_passes):
                    client.train()
            train_cost = time.time() - t_t

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            if i % self.save_gap == 0 or i == self.global_rounds:
                self.save_global_model(round_num=i)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            print(f"    (eval {eval_cost:.1f}s | train {train_cost:.1f}s)")

            if self.auto_break and hasattr(self, 'rs_test_acc') and len(self.rs_test_acc) > 0 and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        if self.rs_test_acc:
            print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPerAvg)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def evaluate_one_step(self, round_num=0):
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()
        stats = self.test_metrics()
        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)
            
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
        
        # evaluate_one_step used to append nothing, so rs_test_acc stayed
        # empty: save_results() wrote no .h5 and max(self.rs_test_acc) at the
        # end of train() raised ValueError on an empty sequence.
        self.rs_test_acc.append(test_acc)
        self.rs_train_loss.append(test_loss)
        self.rs_test_auc.append(macro_f1)  # slot kept for schema compatibility

        print(f"Round {round_num} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        
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
