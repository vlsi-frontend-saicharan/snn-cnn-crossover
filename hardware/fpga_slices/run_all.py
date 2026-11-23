"""Runs the slice experiments one Vivado process at a time and appends to results/slices.csv.
Open the Vivado command prompt (so `vivado` is on PATH), cd to this folder, then:
    python run_all.py --sets A          # continuity with the paper (Zynq-7020, LUT multipliers)
    python run_all.py --sets B E F      # PYNQ-ZU device (xczu5eg)
Runs already in the CSV are skipped, so it can be restarted."""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "slices.csv")
CSV_R = os.path.join(HERE, "results", "slices_r4.csv")   # rectified runs (timing closed, thermal conditions stated)
HEADER = "tag,part,top,N,lif,maxdsp,xrate,srate,period,luts,ffs,dsps,wns,dyn_W,static_W,signals_W,logic_W,dsp_W,tj_C,margin_C,confidence\n"
Z7 = "xc7z020clg400-1"
ZU = "xczu5eg-sfvc784-1-e"
PERIOD = 4.0                      # 250 MHz, as in the paper
SIZES = [("nmnist18", 18), ("dvs288", 288), ("conv576", 576)]

def configs(sets):
    out = []
    for s in sets:
        if s == "A":              # same device and settings as the paper's Table VII, vectorless power
            for nm, n in SIZES:
                out.append((f"A_{nm}_cnn", Z7, "cnn_slice", n, 0, 0, "default", "default"))
                out.append((f"A_{nm}_snn", Z7, "snn_slice", n, 0, 0, "default", "default"))
        if s == "B":              # PYNQ-ZU device, LUT multipliers, vectorless power
            for nm, n in SIZES:
                out.append((f"B_{nm}_cnn_vl", ZU, "cnn_slice", n, 0, 0, "default", "default"))
                out.append((f"B_{nm}_snn_vl", ZU, "snn_slice", n, 0, 0, "default", "default"))
        if s == "E":              # activity set on the register nets (rate sweep), 288-input slice
            out.append(("E_dvs288_cnn_x50", ZU, "cnn_slice", 288, 0, 0, 50, "default"))
            for r in (0.02, 0.05, 0.1, 0.2, 0.5):
                out.append((f"E_dvs288_snn_s{r}", ZU, "snn_slice", 288, 0, 0, "default", r))
            out.append(("E_dvs288_snn_lif_s0.1", ZU, "snn_slice", 288, 1, 0, "default", 0.1))
        if s == "F":              # CNN multipliers forced into DSP blocks
            out.append(("F_nmnist18_cnn_dsp", ZU, "cnn_slice_dsp", 18, 0, -1, 50, "default"))
            out.append(("F_dvs288_cnn_dsp", ZU, "cnn_slice_dsp", 288, 0, -1, 50, "default"))
        if s == "R":              # R4 rectified: closed timing (period found by iteration), explicit activity, Tj reported
            for nm, n in SIZES:
                out.append((f"R_{nm}_cnn_x50", ZU, "cnn_slice", n, 0, 0, 50, "default"))
                out.append((f"R_{nm}_cnn_dsp_x50", ZU, "cnn_slice_dsp", n, 0, -1, 50, "default"))
                for r_ in (0.05, 0.1, 0.2, 0.33, 0.5):
                    out.append((f"R_{nm}_snn_s{r_}", ZU, "snn_slice", n, 0, 0, "default", r_))
    return out

def run_closed(viv, tag, part, top, n, lif, maxdsp, xr, sr, csv):
    """Start at 10 ns; if setup slack is negative, set period = period - WNS + 2 ns and rerun (max 3 tries). Every try is a CSV row tagged _p<period>."""
    period = 10.0
    for _ in range(3):
        t = f"{tag}_p{period:g}"
        cmd = [viv, "-mode", "batch", "-notrace", "-nojournal", "-nolog", "-source", os.path.join(HERE, "run_slice.tcl"),
               "-tclargs", part, top, str(n), str(lif), str(maxdsp), str(xr), str(sr), str(period), csv, t]
        print(">>", t, flush=True); subprocess.run(cmd, cwd=HERE)
        try: wns = float(open(csv).read().splitlines()[-1].split(",")[12])
        except Exception: return
        if wns >= 0: return
        period = round(period - wns + 2.0, 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sets", nargs="+", default=["A"])
    a = ap.parse_args()
    viv = shutil.which("vivado") or shutil.which("vivado.bat")
    if not viv: sys.exit("vivado not on PATH: use the Vivado command prompt")
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    if not os.path.exists(CSV): open(CSV, "w").write(HEADER)
    done = {l.split(",")[0] for l in open(CSV).read().splitlines()[1:]}
    if "R" in a.sets:
        if not os.path.exists(CSV_R): open(CSV_R, "w").write(HEADER)
        doneR = {l.split(",")[0].rsplit("_p", 1)[0] for l in open(CSV_R).read().splitlines()[1:]}
        for tag, part, top, n, lif, maxdsp, xr, sr in configs(["R"]):
            if tag in doneR: print("skip", tag); continue
            run_closed(viv, tag, part, top, n, lif, maxdsp, xr, sr, CSV_R)
        a.sets = [s for s in a.sets if s != "R"]
    for tag, part, top, n, lif, maxdsp, xr, sr in configs(a.sets):
        if tag in done: print("skip", tag); continue
        cmd = [viv, "-mode", "batch", "-notrace", "-nojournal", "-nolog", "-source", os.path.join(HERE, "run_slice.tcl"),
               "-tclargs", part, top, str(n), str(lif), str(maxdsp), str(xr), str(sr), str(PERIOD), CSV, tag]
        print(">>", tag, flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode: print("FAILED", tag, "code", r.returncode)

if __name__ == "__main__":
    main()
