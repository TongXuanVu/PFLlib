# Task 1 (full) — checkpoint va metric

Lan chay PerAvg tren IoV task 1 (lop 0,1,2 = Benign/DoS/double), 50 client,
`-lbs 256 -lr 0.01 -eg 5 -plp 1`. Phien Kaggle bi cat o gioi han 12 gio nen
**dung o round 25**, chua toi round 30.

| round | Accuracy | Macro-F1 |
|---|---|---|
| 0  | 0.0015 | 0.0026 |
| 5  | 0.9962 | 0.3327 |
| 10 | 0.9962 | 0.3327 |
| 15 | 0.9962 | 0.3327 |
| 20 | 0.9966 | 0.4030 |
| 25 | 0.9965 | 0.4651 |

Macro-F1 = 0.33270 o round 5-15 dung bang nguong doan thuan lop da so tren
3 lop (Benign chiem 99.6236% tap test task 1): 2p/(1+p)/3 = 0.33270. Tu round
20 mo hinh moi thoat khoi trang thai do, va van dang tang khi bi cat.

Dung `PerAvg_server_round_25.pt` lam diem xuat phat cho task 2:

    python main.py ... -tid 2 -init <duong-dan>/PerAvg_server_round_25.pt -ecum
