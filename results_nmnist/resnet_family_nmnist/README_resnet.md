# N-MNIST second architecture family (residual network)   [nmnist_resnet_v1]
Spiking ResNet (RSNN-R_T<T>_nokd) vs ReLU ResNet on the time-summed frame (RCNN-R_T1_nokd); ~0.16 M parameters; no distillation.
Recipe = snn_win.py: AdamW lr 0.002, wd 0.0001, cosine, batch 128, 20 epochs, last-epoch accuracy, 3 seeds, count input log1p, LIF beta 0.9.
results.json  one entry per (model, seed): acc, params, E_main_uJ (SNN: first layer dense MACs + ACs x spikes + membrane updates; CNN: dense MACs),
              E_strict_uJ (CNN with zero-skipping = generous baseline), E_evt_uJ.
preds/<model>_s<seed>.npy  test predictions;  ckpt/*_final.pt  weights;  resnet_latency_energy.json  batch-1 latency + NVML energy;
tables/resnet_family_models.csv, resnet_family_snn_vs_cnn.csv   final tables with T* for estimate / GPU latency / GPU energy.
Code: code_snapshot_* (snn_win.py) and nmnist_resnet.py in this folder.
