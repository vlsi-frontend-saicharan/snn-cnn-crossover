# snn_cnn_fpga  -  N-MNIST SNN-L (T=1,2,3) vs CNN-L on FPGA (board-independent first stage)

Vivado project name (created by the script): `snn_cnn_fpga_<cfg>`, cfg = snn_T1 | snn_T2 | snn_T3 | cnn

## Files
| file | what it is |
|---|---|
| rtl/nmnist_core.v | the inference core (SNN mode and CNN mode share lanes, memories, schedule) |
| rtl/tb_nmnist_core.v | self-checking testbench: compares class + all 10 logits + cycle count with the Python model |
| scripts/run_vivado.tcl | creates the project, runs simulation / synthesis / implementation, prints Fmax and utilisation |
| scripts/power_saif.tcl | optional: activity-based power from post-implementation simulation |
| scripts/constraints.xdc | 100 MHz clock constraint (no pins, no board) |
| py/hwmodel.py | bit-exact integer model, quantiser, .mem writers, cycle model |
| py/export_fpga.py | Colab cell: reads your trained checkpoints, quantises, writes export/<cfg>/*.mem |
| py/selftest_hwmodel.py | runs without torch; checks the Python model |
| export/ | (created by export_fpga.py) weights, thresholds, test vectors per config |

## Steps
1. Put this folder on Drive (e.g. `nmnist_snn_cnn_work/fpga_nmnist`). In the Colab session that has `W` imported and the run
   folder active, paste `py/export_fpga.py` into a cell and run it. Look at the printed table:
   int_acc within ~0.3 points of torch_acc, agreement above ~99 %, acc_bits_needed <= 24.
2. Copy the whole folder (rtl, scripts, export) to the PC that has Vivado.
3. From that folder:
   `vivado -mode batch -source scripts/run_vivado.tcl -tclargs snn_T3 xczu7ev-ffvc1156-2-e sim`
   expect `TEST PASSED` (and cycle-count mismatches 0). Repeat for snn_T1, snn_T2, cnn.
   (In the Vivado Tcl console: `set argv {snn_T3 xczu7ev-ffvc1156-2-e sim}; source <full path>/scripts/run_vivado.tcl`)
4. `... -tclargs snn_T3 xczu7ev-ffvc1156-2-e all` for synthesis + implementation. The log prints
   `SUMMARY ... FMAX_MHZ=...` and `UTIL ...` lines; reports are in vivado/<cfg>/.
5. Latency = cycles per sample (testbench RESULT line) / Fmax.  Same for all four configs.
6. Optional: `power_saif.tcl` for activity-based power -> energy = power x latency.

## Notes
* Part: pick any (no board needed). SNN-L needs ~165 BRAM36 equivalents; Zynq-7020 (140) is too small, xczu7ev / xc7a200t are fine.
* P (lanes) supports 32 or 64; ACCW=24 and LOGW=40 are checked by the export script.
* The core has no AXI/UART wrapper yet: that depends on the board and comes after the part is fixed.
