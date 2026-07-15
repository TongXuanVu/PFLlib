import copy
import time
import numpy as np
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from threading import Thread


class PerAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
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

        for i in range(start_round, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # send all parameter for clients
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step(round_num=i)

            # choose several clients to send back upated model to server
            for client in self.selected_clients:
                client.train()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.save_global_model(round_num=i)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and hasattr(self, 'rs_test_acc') and len(self.rs_test_acc) > 0 and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
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
        
        macro_p = np.mean(TP / (TP + FP + 1e-12))
        macro_r = np.mean(TP / (TP + FN + 1e-12))
        macro_f1 = np.mean(2 * (TP / (TP + FP + 1e-12)) * (TP / (TP + FN + 1e-12)) / ((TP / (TP + FP + 1e-12)) + (TP / (TP + FN + 1e-12)) + 1e-12))
        
        weights = global_cm.sum(axis=1)
        total = weights.sum()
        weighted_p = np.sum(weights * (TP / (TP + FP + 1e-12))) / total
        weighted_r = np.sum(weights * (TP / (TP + FN + 1e-12))) / total
        weighted_f1 = np.sum(weights * 2 * (TP / (TP + FP + 1e-12)) * (TP / (TP + FN + 1e-12)) / ((TP / (TP + FP + 1e-12)) + (TP / (TP + FN + 1e-12)) + 1e-12)) / total

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_loss = sum(stats[3])*1.0 / sum(stats[1])
        
        print(f"Round {round_num} - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        
        import pandas as pd
        import os
        csv_file = f"../results/{self.dataset}_{self.algorithm}_metrics.csv"
        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy', 'Micro_P', 'Micro_R', 'Micro_F1', 'Macro_P', 'Macro_R', 'Macro_F1', 'Weighted_P', 'Weighted_R', 'Weighted_F1'])
            df.to_csv(csv_file, index=False)
            
        row = {'Round': round_num, 'Loss': test_loss, 'Accuracy': test_acc, 
               'Micro_P': micro_p, 'Micro_R': micro_r, 'Micro_F1': micro_f1,
               'Macro_P': macro_p, 'Macro_R': macro_r, 'Macro_F1': macro_f1,
               'Weighted_P': weighted_p, 'Weighted_R': weighted_r, 'Weighted_F1': weighted_f1}
        df = pd.DataFrame([row])
        df.to_csv(csv_file, mode='a', header=False, index=False)