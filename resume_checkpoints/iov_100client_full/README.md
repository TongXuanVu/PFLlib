# PerAvg IoV 100 client, FULL, task 1 — diem noi tiep sau Round 14

- `IoV_task1_resume_rr15.pt`: mo hinh sau khi train + tong hop round 14 (file goc `PerAvg_server_round_14.pt`).
  Day chinh la mo hinh se duoc chay thanh "Round 15", nen dat ten `PerAvg_server_round_15.pt` roi chay `-mode resume -rr 15`.
- `IoV_PerAvg_full_t1_metrics_r0-14.csv`: metric Round 0..14 da do (-eg 1 -ecum -esc 0, toan bo test).
- Lenh goc: `-data IoV -m CNN1D -ncl 13 -algo PerAvg -tid 1 -nc 50 -gr 30 -lbs 256 -lr 0.01 -ls 1 -tbs 8192 -eg 1 -ecum -esc 0 -go full_t1`
