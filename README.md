# Code and results: SNN vs CNN crossover timestep (anonymous)

code/dvs128_lowT_kd.py   training, evaluation, McNemar test, op counts (DVS128 Gesture)
code/dvs128_proto.py     model grid, BN folding, CUDA-graph latency, NVML energy
code/paper_checks.py     per-seed McNemar tables, identical-accuracy checks
results/                 one JSON per run: final-epoch test accuracy and test predictions (3 seeds)
measure/                 latency and energy measurement JSONs (Tesla T4)
ops/                     operation counts and spike rates per model
results_summary.csv      run id, seed, final test accuracy

Protocol: last-epoch accuracy, 3 seeds (0,1,2); McNemar per seed (exact binomial if <25 discordant, else chi-square with continuity correction);
energy: NVML above idle, batch 64, 10 s, 3 repeats; latency: batch-1 CUDA-graph replay, BN folded.
