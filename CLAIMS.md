# Headline claims and their evidence

Each row gives a claim of the paper, the value reported, the file that supports it and the wording the evidence allows. Paths are relative to the repository root.

## GPU energy (Table III, Table V, Fig. 4)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| SNN uses more GPU energy than the reference CNN at every T from 2 | ratio 2.45 to 17.65, every 95% interval excludes 1 | `review/results/r8_ci.json` (raw sessions: `r8_sess1.json` to `r8_sess4.json`) | "higher in every case"; ratios are from 60 units (4 sessions x 3 seeds x 5 repeats) |
| Measured GPU-energy crossover is of order one | T\* 0.82-1.44 (S), 0.67-0.97 (M), 0.77-0.91 (L) | `review/results/r8_ci.json` | "close to one"; board power from NVML at batch 64, not wall-plug power |
| Reference CNN energy per sample | 1.54, 2.14, 4.00 mJ for S, M, L | `review/results/r8_means.json` | per-session means differ by up to 17%; the intervals do not model drift between GPU instances |
| The 16-bin width-M CNN is cheap partly because of a lower GPU clock | 1.33 mJ, about 955 MHz | `review/results/r8_means.json` (session 1) | "partly because of the clock"; the clock varied 795-1590 MHz and could not be locked |

## Accuracy and statistics (Table II, Table III, Section IV-A)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| Reference CNN bin counts were chosen on a validation split | B = 1, 2, 2 for S, M, L | `review/results/valsplit.json`, `review/results/r2_locked_selection.json` | split is stratified, not subject-disjoint |
| SNN exceeds the best CNN only at T=16 | +3.36, +2.78, +2.66 points; Holm p = 0.056, 0.113, 0.113 | `review/results/r10_vs_best.json` | "higher mean, not confirmed after correction" |
| SNN is ahead of the stacked-bin CNN (B=16) | +7.2, +6.7, +7.6 points, Holm p < 0.001 | `review/results/r10_family.json` | this CNN is a weak control |
| A per-bin ReLU CNN is more accurate than the SNN at every T (width M only) | gap 1.9 to 5.5 points | `results/perbin/` | tested for width M of the first family only |
| Residual family reproduces the crossover | see Table VII | `results/resnet_family_dvs128/` (`resnet_family_snn_vs_cnn.csv`, `cross_width_tests.csv`) | reference CNN chosen on test accuracy, so exploratory |

## GPU latency (Table II, Table III, Table V)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| Batch-1 GPU latency of every DVS128 model | Table II "Lat." column; GPU latency crossover 0.9-2.3 | `measure/proto_pm_1020_2331_m1.json` (all CNN bin counts and SNN at T=2-16), `measure/proto_pm_1023_0727_t1.json` (T=1 SNN) | CUDA-graph latency on a Tesla T4, one session, seed 0; the table values agree with these files within about 5% |

## CPU latency (Table V)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| CPU batch-1 crossover is close to one on DVS128 | T\* 0.70-1.00 (S), 0.79-1.06 (M), 0.91-1.03 (L) | `review/results/cpu_latency.json` | shared cloud CPU, seed-0 models, median of 15 round-robin rounds; the N-MNIST cell comes from an earlier run that was not repeated |

## Event density (Fig. 5)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| SNN accuracy falls faster than the CNN's when events are removed | 14 to 32 points below the reference CNN at q=0.25 | `review/results/r7_density.json` | synthetic random removal, not a validated camera-noise model; no retraining |

## BatchNorm folding

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| Folded and original logits agree | CNN 9.3e-6 (fp32); SNN 0.19 (fp32), 0 (fp64) | `review/results/r1_fold_audit.json` | first architecture family only |

## Operation-count estimate and sensitivity (Section IV-E)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| Estimated crossover is 5.2 to 9.3 on DVS128 | see Table V | `analysis/tstar_sensitivity.csv`, `analysis/dvs128_pareto_data.csv`, script `analysis/make_pareto_and_sensitivity.py` | historical 45 nm constants, no memory access; a lower bound |

## FPGA neuron slices (Table V, Section IV-F)

| Claim | Value in paper | Evidence file | Allowed wording |
|---|---|---|---|
| SNN slice needs fewer LUTs | 8.4x (18 inputs), 6.9x (288), 6.8x (576) | `hardware/results/slices_r4.xlsx` and the `R_*` report folders | single-neuron slices |
| Slice crossover depends on the spike rate | T\* 4.6-9.7 (288 inputs), 4.5-9.3 (576 inputs); 4.8 at 33% | `hardware/results/slices_r4.xlsx`, `hardware/fpga_slices/tstar_r4.py` | dynamic power only, 8-bit, out-of-context; no network-level FPGA claim |
| DSP variant is not used | timing not met | `hardware/results/R_*_cnn_dsp_*` | no reported timing slack |

## Sensor-constrained selection (Table VI)

| Claim | Value in paper | Evidence | Allowed wording |
|---|---|---|---|
| Measured energy selects a CNN in all three scenarios; the estimate selects an SNN | 1.54, 1.33, 2.14 mJ against SNNs at 3.76, 3.76, 17.1 mJ | enumeration over the 18 candidates, using `review/results/r8_means.json` and the accuracy and latency in `results/` | a finite enumeration, not a new optimisation method |

## Results reported in the paper without a released raw file

No raw output file for these results is released in this repository. They cannot be re-derived from the files here without rerunning the measurement.

| Claim | Value in paper | Note |
|---|---|---|
| Test-time truncation and confidence-threshold early exit | Section IV-G | raw output not released |
| Distillation control | M CNN 90.28 to 91.67%, S CNN 88.66 to 90.86%, T=2 SNN 88.43 to 90.39% (Section IV-G) | re-runs, not rows of Table II; repeated training is reproducible to about 0.6 points |
