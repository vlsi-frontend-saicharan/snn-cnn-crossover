# DVS128 Gesture - second architecture family (residual network)   [resnet_v1]
What: spiking ResNet (rsnn) versus ReLU ResNet (rcnn), width 1.0 (~0.16 M parameters), same recipe as the main study.
Naming: results/resnet_<arch>_T<T>K<K>_w1_plain_s<seed>.json   arch rsnn = spiking, rcnn = ReLU;
        rsnn: T = timesteps (K=1); rcnn: K = number of stacked bins B (T=1).   One JSON per run (final-epoch test accuracy + predictions).
ckpt/resnet_*.pth  final weights.   ops/resnet_stats_*.json  op-count estimate and spike rates.
measure/proto_pm_<date>_resnet.json  batch-1 latency (eager, CUDA graph, BN folded) and NVML energy.
summary/resnet_family_models.csv  and  resnet_family_snn_vs_cnn.csv  the final tables (T* for estimate / GPU latency / GPU energy).
Protocol: {"tag": "resnet", "epochs": 100, "bs": 32, "lr": 0.001, "wd": 0.01, "warmup": 5, "ls": 0.1, "drop": 0.3, "kd_alpha": 0.7, "kd_T": 4.0, "aug": true}   Reported accuracy = FINAL epoch, 3 seeds, no distillation.
Code: code/dvs128_resnet.py (this experiment), dvs128_lowT_kd.py, dvs128_proto.py (shared training / measurement code).
