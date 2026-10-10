# PerAvg IoV 100 client, FULL, task 1 — diem noi tiep sau Round 14

- `IoV_task1_resume_rr15.pt`: mo hinh sau khi train + tong hop round 14 (file goc `PerAvg_server_round_14.pt`).
  Day chinh la mo hinh se duoc chay thanh "Round 15", nen dat ten `PerAvg_server_round_15.pt` roi chay `-mode resume -rr 15`.
- `IoV_PerAvg_full_t1_metrics_r0-14.csv`: metric Round 0..14 da do (-eg 1 -ecum -esc 0, toan bo test).
- Lenh goc: `-data IoV -m CNN1D -ncl 13 -algo PerAvg -tid 1 -nc 50 -gr 30 -lbs 256 -lr 0.01 -ls 1 -tbs 8192 -eg 1 -ecum -esc 0 -go full_t1`

## Cap nhat: task 1 da train xong round 29 (metric Round 0..29)
- `IoV_task1_round_29.pt`: model cuoi task 1 (dung lam `-init` cho task 2). Metric "Round 30" = chinh model nay, CHUA do.
- `IoV_PerAvg_full_t1_metrics_r0-29.csv`: metric Round 0..29. Luu y: Round 15 co loss 1.28 (nhay) ngay tai diem noi tiep.

## Luu y BatchNorm khi resume
`set_parameters`/aggregate chi chuyen `.parameters()`, KHONG chuyen buffer BatchNorm (running_mean/var). Client moi tao (sau resume) co buffer khoi tao
=> lan cham dau tien sau resume (Round 15 da biet; "Round 30" khi resume rr=30) cho loss tang vot (~1.26-1.28) va F1 ve 0.3327. Day la hien tuong cua resume,
khong phai ket qua that. Cell Kaggle resume LUI MOT round (rr=j tu round_{j-1}.pt) va bo dong cham dau tien.
- `IoV_task1_round_28.pt`: model sau round 28, dung de chay lai round 29 va co Round 30 hop le.
