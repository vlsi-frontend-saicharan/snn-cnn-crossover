"""Reads results/slices.csv and prints LUT ratios and the slice crossover T* = E_cnn / E_snn_per_cycle.
Energy per output = dynamic power x cycles / f, one cycle for the CNN slice, one cycle per timestep for the SNN slice."""
import csv, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "slices.csv"))))
def num(v):
    try: return float(str(v).lstrip("<"))
    except ValueError: return None
by = {r["tag"]: r for r in rows}
def show(cnn_tag, snn_tag, label):
    c, s = by.get(cnn_tag), by.get(snn_tag)
    if not c or not s: return
    f = 1e9 / float(c["period"])              # Hz
    ec, es = num(c["dyn_W"]) / f * 1e9, num(s["dyn_W"]) / f * 1e9   # nJ
    lr = num(c["luts"]) / num(s["luts"]) if num(s["luts"]) else float("nan")
    print(f"{label:28s} LUT cnn {c['luts']:>7s} snn {s['luts']:>6s} ratio {lr:5.2f} | "
          f"E_cnn {ec:8.2f} nJ  E_snn/cycle {es:7.2f} nJ  T* {ec/es:5.2f}  | WNS cnn {c['wns']} snn {s['wns']}")
print("--- A: Zynq-7020, as in the paper (paper: N-MNIST 284/1482 LUT T*=3.39; DVS128 3299/20425 LUT T*=5.21)")
for nm in ("nmnist18", "dvs288", "conv576"): show(f"A_{nm}_cnn", f"A_{nm}_snn", f"7z020 {nm}")
print("--- B: XCZU5EG, vectorless")
for nm in ("nmnist18", "dvs288", "conv576"): show(f"B_{nm}_cnn_vl", f"B_{nm}_snn_vl", f"zu5eg {nm}")
print("--- E: XCZU5EG, DVS128 slice, CNN data toggle 50 percent, SNN spike-rate sweep (activity set on the register nets)")
for r in ("0.02", "0.05", "0.1", "0.2", "0.5"): show("E_dvs288_cnn_x50", f"E_dvs288_snn_s{r}", f"spike rate {r}")
print("--- E: LIF included (compare with the same spike rate without LIF)")
for t_ in ("E_dvs288_snn_lif_s0.1", "E_dvs288_snn_s0.1"):
    if t_ in by: print(t_, "LUT", by[t_]["luts"], "dyn_W", by[t_]["dyn_W"])
print("--- F: CNN with DSP blocks (LUT, DSP, dynamic power); T* against the SNN of the same rate")
for t_ in ("E_dvs288_cnn_x50", "F_dvs288_cnn_dsp", "F_nmnist18_cnn_dsp"):
    if t_ in by: print(t_, "LUT", by[t_]["luts"], "DSP", by[t_]["dsps"], "dyn_W", by[t_]["dyn_W"])
for r in ("0.02", "0.05", "0.1", "0.2", "0.5"): show("F_dvs288_cnn_dsp", f"E_dvs288_snn_s{r}", f"DSP-CNN vs SNN rate {r}")
