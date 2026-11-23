# Crossover Timesteps of Spiking Networks for Event-Based Sensing

Code, per-run results and measurement files for the paper
**"Crossover Timesteps of Spiking Networks for Event-Based Sensing: Measured Energy, Latency and Accuracy Against Matched CNNs"**
(Sai Charan, Mayur Katwe, Bhavana Agrawal; NIT Raipur).

The crossover timestep T\* is the number of timesteps at which a spiking network (SNN) costs as much as its matched convolutional network (CNN). The paper measures it on the same trained models for an operation-count estimate, GPU energy (batch 64), GPU latency (batch 1), CPU latency (batch 1) and FPGA neuron slices, next to accuracy on DVS128 Gesture and N-MNIST.

## Where the paper's results are

| Paper part | Folder |
|---|---|
| DVS128 Gesture models, training and accuracy | `code/`, `results/` |
| N-MNIST models, training and accuracy | `code_nmnist/`, `results_nmnist/` |
| GPU energy and latency, CPU latency | `measure/` |
| Paired tests, controls, sensor-constrained selection, event density | `analysis/` |
| FPGA neuron slices (Table V, Section IV-F) | `hardware/fpga_slices/`, `hardware/results/` |

Everything else (`ops/`, `code_nmnist/fpga_nmnist/`) is supporting or exploratory code and is not used for any result in the paper. In particular, `code_nmnist/fpga_nmnist/` is an exploratory hardware model; the paper makes no network-level FPGA claim.

## Setup

- Python 3 and PyTorch; install the packages in `requirements.txt` if it is present.
- GPU measurements: one Tesla T4 on a cloud instance. The GPU clock could not be locked and varied between 795 and 1590 MHz (logged for every reading).
- CPU latency: a shared cloud CPU, batch 1.
- FPGA slices: Vivado, device xczu5eg (Zynq UltraScale+), out-of-context synthesis.

## FPGA slices (`hardware/`)

- `fpga_slices/`: matched single-neuron slices (`cnn_slice.sv`, `snn_slice.sv`, DSP variant `cnn_slice_dsp.sv`), testbench `tb_slices.sv`, Vivado script `run_slice.tcl`, driver `run_all.py`, and the crossover calculation `tstar_r4.py` / `tstar_from_csv.py`.
- `results/`: summary table `slices_r4.xlsx` and, for each run `R_<slice>_<activity>_p<period>`, the Vivado power and utilization reports.
- Protocol: 8-bit LUT-multiplier MAC (CNN) against a spike-gated 8-bit add (SNN); 288 inputs (DVS128) and 576 inputs (N-MNIST conv2) are used for the results, the 18-input slice is not interpreted. The clock period starts at 10 ns and is raised until timing is met. Dynamic power comes from Vivado power analysis with explicit switching activity (CNN toggle rate 50%, SNN spike probability 0.05 to 0.5) at 25 C ambient. The DSP-multiplier CNN variant did not meet timing and is not used.
- Run `python run_all.py` inside the Vivado Tcl shell, then `python tstar_r4.py` to print the table of T\*.
- These are neuron slices, not networks. Static power and memory are excluded.

## Claims and evidence

`CLAIMS.md` (if present) lists each headline claim of the paper, its value, the file that supports it and the wording the evidence allows.

## Reproducibility notes

- Accuracies are the last epoch of three seeds; the seeds change initialisation and data order, not the data.
- The reference CNN of each DVS128 width is chosen on a 20% validation split of the training recordings (not subject-disjoint).
- GPU energy values are means over 60 units (4 sessions x 3 seeds x 5 repeats) with 95% bootstrap intervals over units.

## License and citation

Add your license here. Please cite the paper if you use this code or data.
