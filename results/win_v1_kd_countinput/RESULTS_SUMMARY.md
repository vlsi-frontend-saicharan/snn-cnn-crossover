# win_v1_kd_countinput

SNN T=1,2 widths S/M/L, count input + BN + KD from DeepSNN T5, CNN baselines with same recipe

code hash 25a82c4a  |  epochs 20  |  KD alpha 0.7 temp 4.0  |  gpu Tesla T4

| spec          | kind   | width   |   T | kd    |   params |   seeds | acc_%        |   latency_eager_ms |   latency_graph_ms |   E_main_uJ |   E_strict_uJ |
|:--------------|:-------|:--------|----:|:------|---------:|--------:|:-------------|-------------------:|-------------------:|------------:|--------------:|
| CNN-L_T1_nokd | CNN    | L       |   1 | False |   411210 |       1 | 98.78        |             0.3954 |             0.1347 |     191.15  |       113.776 |
| CNN-L_T1_kd   | CNN    | L       |   1 | True  |   411210 |       3 | 98.77 ± 0.08 |             0.3907 |             0.1347 |     191.15  |       115.884 |
| SNN-L_T1_kd   | SNN    | L       |   1 | True  |   411210 |       1 | 97.61        |             0.7475 |             0.1497 |       7.998 |         7.998 |
| SNN-L_T2_kd   | SNN    | L       |   2 | True  |   411210 |       3 | 98.65 ± 0.06 |             1.4076 |             0.2495 |      16.977 |        16.977 |
| SNN-L_T3_kd   | SNN    | L       |   3 | True  |   411210 |       3 | 98.87 ± 0.07 |             1.9729 |             0.366  |      27.017 |        27.017 |
| CNN-M_T1_nokd | CNN    | M       |   1 | False |   113450 |       1 | 98.70        |             0.4247 |             0.0922 |      49.366 |        30.633 |
| CNN-M_T1_kd   | CNN    | M       |   1 | True  |   113450 |       3 | 98.72 ± 0.07 |             0.4164 |             0.0936 |      49.366 |        29.353 |
| SNN-M_T1_nokd | SNN    | M       |   1 | False |   113450 |       1 | 96.83        |             0.7936 |             0.1117 |       3.858 |         3.858 |
| SNN-M_T1_kd   | SNN    | M       |   1 | True  |   113450 |       3 | 96.57 ± 0.65 |             0.7872 |             0.1193 |       3.657 |         3.657 |
| SNN-M_T2_nokd | SNN    | M       |   2 | False |   113450 |       1 | 98.25        |             1.4541 |             0.197  |       8.326 |         8.326 |
| SNN-M_T2_kd   | SNN    | M       |   2 | True  |   113450 |       1 | 98.23        |             1.4607 |             0.2324 |       7.381 |         7.381 |
| CNN-S_T1_nokd | CNN    | S       |   1 | False |    33690 |       1 | 98.34        |             0.3961 |             0.0673 |      13.131 |         8.086 |
| CNN-S_T1_kd   | CNN    | S       |   1 | True  |    33690 |       3 | 98.55 ± 0.14 |             0.3947 |             0.067  |      13.131 |         7.322 |
| SNN-S_T1_kd   | SNN    | S       |   1 | True  |    33690 |       3 | 94.57 ± 0.62 |             0.7605 |             0.0817 |       1.782 |         1.782 |
| SNN-S_T2_kd   | SNN    | S       |   2 | True  |    33690 |       1 | 97.43        |             1.4151 |             0.1361 |       3.626 |         3.626 |


| SNN           | CNN           |   seeds |   acc_diff_pts | acc_sig_wins/losses   | ACC   |   latency_ratio_snn/cnn | LAT   |   energy_ratio_cnn/snn | ENERGY   | overall   | same_width   |
|:--------------|:--------------|--------:|---------------:|:----------------------|:------|------------------------:|:------|-----------------------:|:---------|:----------|:-------------|
| SNN-L_T1_kd   | CNN-L_T1_kd   |       1 |          -1.13 | 0/1                   | LOSS  |                   1.111 | LOSS  |                  14.35 | WIN      | MIXED     | True         |
| SNN-L_T2_kd   | CNN-L_T1_kd   |       3 |          -0.12 | 0/1                   | LOSS  |                   1.852 | LOSS  |                   6.83 | WIN      | MIXED     | True         |
| SNN-L_T3_kd   | CNN-L_T1_kd   |       3 |           0.1  | 1/0                   | WIN   |                   2.717 | LOSS  |                   4.29 | WIN      | MIXED     | True         |
| SNN-M_T1_nokd | CNN-M_T1_nokd |       1 |          -1.87 | 0/1                   | LOSS  |                   1.211 | LOSS  |                   7.94 | WIN      | MIXED     | True         |
| SNN-M_T1_kd   | CNN-M_T1_kd   |       3 |          -2.15 | 0/3                   | LOSS  |                   1.275 | LOSS  |                   8.03 | WIN      | MIXED     | True         |
| SNN-M_T2_nokd | CNN-M_T1_nokd |       1 |          -0.45 | 0/1                   | LOSS  |                   2.137 | LOSS  |                   3.68 | WIN      | MIXED     | True         |
| SNN-M_T2_kd   | CNN-M_T1_kd   |       1 |          -0.52 | 0/1                   | LOSS  |                   2.483 | LOSS  |                   3.99 | WIN      | MIXED     | True         |
| SNN-S_T1_kd   | CNN-S_T1_kd   |       3 |          -3.98 | 0/3                   | LOSS  |                   1.219 | LOSS  |                   4.11 | WIN      | MIXED     | True         |
| SNN-S_T2_kd   | CNN-S_T1_kd   |       1 |          -0.97 | 0/1                   | LOSS  |                   2.031 | LOSS  |                   2.1  | WIN      | MIXED     | True         |
| SNN-L_T1_kd   | CNN-M_T1_kd   |       1 |          -1.14 | 0/1                   | LOSS  |                   1.599 | LOSS  |                   3.68 | WIN      | MIXED     | False        |
| SNN-L_T1_kd   | CNN-S_T1_kd   |       1 |          -0.79 | 0/1                   | LOSS  |                   2.234 | LOSS  |                   0.95 | TIE      | MIXED     | False        |
| SNN-L_T2_kd   | CNN-M_T1_kd   |       3 |          -0.07 | 0/0                   | TIE   |                   2.666 | LOSS  |                   1.73 | WIN      | MIXED     | False        |
| SNN-L_T2_kd   | CNN-S_T1_kd   |       3 |           0.1  | 1/0                   | WIN   |                   3.724 | LOSS  |                   0.43 | LOSS     | MIXED     | False        |
| SNN-L_T3_kd   | CNN-M_T1_kd   |       3 |           0.14 | 1/0                   | WIN   |                   3.91  | LOSS  |                   1.09 | WIN      | MIXED     | False        |
| SNN-L_T3_kd   | CNN-S_T1_kd   |       3 |           0.32 | 3/0                   | WIN   |                   5.463 | LOSS  |                   0.27 | LOSS     | MIXED     | False        |
| SNN-M_T1_nokd | CNN-L_T1_nokd |       1 |          -1.95 | 0/1                   | LOSS  |                   0.829 | WIN   |                  29.49 | WIN      | MIXED     | False        |
| SNN-M_T1_nokd | CNN-S_T1_nokd |       1 |          -1.51 | 0/1                   | LOSS  |                   1.66  | LOSS  |                   2.1  | WIN      | MIXED     | False        |
| SNN-M_T1_kd   | CNN-L_T1_kd   |       3 |          -2.2  | 0/3                   | LOSS  |                   0.886 | WIN   |                  31.69 | WIN      | MIXED     | False        |
| SNN-M_T1_kd   | CNN-S_T1_kd   |       3 |          -1.98 | 0/3                   | LOSS  |                   1.781 | LOSS  |                   2    | WIN      | MIXED     | False        |
| SNN-M_T2_nokd | CNN-L_T1_nokd |       1 |          -0.53 | 0/1                   | LOSS  |                   1.463 | LOSS  |                  13.67 | WIN      | MIXED     | False        |
| SNN-M_T2_nokd | CNN-S_T1_nokd |       1 |          -0.09 | 0/0                   | TIE   |                   2.927 | LOSS  |                   0.97 | TIE      | MIXED     | False        |
| SNN-M_T2_kd   | CNN-L_T1_kd   |       1 |          -0.51 | 0/1                   | LOSS  |                   1.725 | LOSS  |                  15.55 | WIN      | MIXED     | False        |
| SNN-M_T2_kd   | CNN-S_T1_kd   |       1 |          -0.17 | 0/0                   | TIE   |                   3.469 | LOSS  |                   1.03 | TIE      | MIXED     | False        |
| SNN-S_T1_kd   | CNN-L_T1_kd   |       3 |          -4.2  | 0/3                   | LOSS  |                   0.607 | WIN   |                  65.02 | WIN      | MIXED     | False        |
| SNN-S_T1_kd   | CNN-M_T1_kd   |       3 |          -4.15 | 0/3                   | LOSS  |                   0.873 | WIN   |                  16.47 | WIN      | MIXED     | False        |
| SNN-S_T2_kd   | CNN-L_T1_kd   |       1 |          -1.31 | 0/1                   | LOSS  |                   1.01  | TIE   |                  31.65 | WIN      | MIXED     | False        |
| SNN-S_T2_kd   | CNN-M_T1_kd   |       1 |          -1.32 | 0/1                   | LOSS  |                   1.454 | LOSS  |                   8.13 | WIN      | MIXED     | False        |