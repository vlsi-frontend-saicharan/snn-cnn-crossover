"""R4 table: reads results/slices_r4.csv (timing-closed runs). For each size/kind keeps the run with WNS>=0 (smallest passing period).
E per output = dynamic power x period (nJ per cycle); SNN slice needs T cycles, so T* = E_cnn / E_snn_per_cycle. Tj > 100 C or margin < 0 is flagged."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "slices_r4.csv"))))
def num(v):
    try: return float(str(v).lstrip("<"))
    except (ValueError, TypeError): return None
best = {}
for r in rows:
    base = r["tag"].rsplit("_p", 1)[0]; w = num(r["wns"])
    ok = w is not None and w >= 0
    if base not in best or (ok and not best[base][1]) or (ok == best[base][1] and float(r["period"]) < float(best[base][0]["period"])): best[base] = (r, ok)
print(f"{'run':26s} {'period':>7s} {'timing':>7s} {'LUT':>6s} {'DSP':>4s} {'dyn W':>7s} {'E nJ/cycle':>10s} {'Tj C':>6s} {'margin':>7s} {'conf':>7s}")
E = {}
for base, (r, ok) in sorted(best.items()):
    e = num(r["dyn_W"]) * float(r["period"]); E[base] = e
    tj = num(r["tj_C"]); flag = "  <-- Tj high" if (tj is not None and tj >= 85) else ""
    if num(r["dyn_W"]) is not None and num(r["dyn_W"]) < 0.01: flag += "  (power < 10 mW: 3-decimal resolution, do not interpret)"
    print(f"{base:26s} {r['period']:>7s} {'MET' if ok else 'NOT MET':>7s} {r['luts']:>6s} {r['dsps']:>4s} {r['dyn_W']:>7s} {e:10.3f} {r['tj_C']:>6s} {r['margin_C']:>7s} {r['confidence']:>7s}{flag}")
print("\nT* = E_cnn / E_snn(per cycle)")
for nm in ("nmnist18", "dvs288", "conv576"):
    for ck, lab in ((f"R_{nm}_cnn_x50", "LUT-CNN"), (f"R_{nm}_cnn_dsp_x50", "DSP-CNN")):
        for s in ("0.05", "0.1", "0.2", "0.33", "0.5"):
            sk = f"R_{nm}_snn_s{s}"
            if ck in E and sk in E: print(f"  {nm:9s} {lab:8s} vs SNN spike rate {s:>4s}:  T* = {E[ck]/E[sk]:5.2f}")

print("\nAll attempts of runs whose timing is NOT MET (period ns, WNS ns):")
for base, (r, ok) in sorted(best.items()):
    if not ok:
        print(" ", base, [(x["period"], x["wns"]) for x in rows if x["tag"].rsplit("_p", 1)[0] == base])
