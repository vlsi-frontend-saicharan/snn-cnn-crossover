# win_v2_epochs60

60 epochs for SNN and CNN, one seed, same budget

code hash 25a82c4a  |  epochs 60  |  KD alpha 0.7 temp 4.0  |  gpu Tesla T4

| spec        | kind   | width   |   T | kd   |   params |   seeds |   acc_% |   latency_eager_ms |   latency_graph_ms |   E_main_uJ |   E_strict_uJ |
|:------------|:-------|:--------|----:|:-----|---------:|--------:|--------:|-------------------:|-------------------:|------------:|--------------:|
| CNN-L_T1_kd | CNN    | L       |   1 | True |   411210 |       1 |   98.68 |             0.4003 |             0.1346 |     191.15  |       106.974 |
| SNN-L_T1_kd | SNN    | L       |   1 | True |   411210 |       1 |   97.75 |             0.76   |             0.1489 |       7.511 |         7.511 |
| SNN-L_T2_kd | SNN    | L       |   2 | True |   411210 |       1 |   98.67 |             1.4652 |             0.2669 |      15.335 |        15.335 |
| SNN-L_T3_kd | SNN    | L       |   3 | True |   411210 |       1 |   98.91 |             1.855  |             0.3818 |      23.551 |        23.551 |


| SNN         | CNN         |   seeds |   acc_diff_pts | acc_sig_wins/losses   | ACC   |   latency_ratio_snn/cnn | LAT   |   energy_ratio_cnn/snn | ENERGY   | overall   | same_width   |
|:------------|:------------|--------:|---------------:|:----------------------|:------|------------------------:|:------|-----------------------:|:---------|:----------|:-------------|
| SNN-L_T1_kd | CNN-L_T1_kd |       1 |          -0.93 | 0/1                   | LOSS  |                   1.106 | LOSS  |                  14.24 | WIN      | MIXED     | True         |
| SNN-L_T2_kd | CNN-L_T1_kd |       1 |          -0.01 | 0/0                   | TIE   |                   1.983 | LOSS  |                   6.98 | WIN      | MIXED     | True         |
| SNN-L_T3_kd | CNN-L_T1_kd |       1 |           0.23 | 1/0                   | WIN   |                   2.837 | LOSS  |                   4.54 | WIN      | MIXED     | True         |