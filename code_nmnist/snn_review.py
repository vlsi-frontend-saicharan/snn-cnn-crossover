"""nmnist_snn_cnn - Part 3: everything a reviewer asked for, on N-MNIST.
  1. more seeds (mean +- std, paired comparison)      -> use S.SEED = 0,1,2 with S.run_sweep()
  2. stronger CNN baselines that SEE the time bins     -> CNN-Stack, CNN-Stack-Small (2*T input channels)
  3. INT8 baseline + fairer energy model               -> review_seed(): int8 accuracy + op counts + 4 energy scenarios
  4. KD ablation (alpha, temperature)                  -> kd_ablation(T)
  5. paper-ready tables and plots                      -> paper_tables()
Saved in Google Drive next to the results so the exact code is kept.  Needs snn_sweep.py (Part 2) in the same folder.
"""
import os, json, glob, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import snn_sweep as S

ALL = ["CNN", "CNN-Small", "CNN-Stack", "CNN-Stack-Small", "DeepSNN", "LiteSNN", "KDSNN"]   # DeepSNN before KDSNN

# Horowitz 2014 (ISSCC), 45 nm, pJ:  fp32 mult 3.7, fp32 add 0.9 | int8 mult 0.2, int32 add 0.1
#   fp32 : MAC = 3.7 + 0.9 = 4.6 pJ,  AC (accumulate only) = 0.9 pJ
#   int8 : MAC = 0.2 + 0.1 = 0.3 pJ (8-bit weights/activations, 32-bit accumulator), AC = 0.1 pJ (32-bit accumulator)
E = {"fp32": {"mac": 4.6, "ac": 0.9}, "int8": {"mac": 0.3, "ac": 0.1}}
# "+upd": the SNN also pays one MAC-cost per neuron per time step for the membrane update (decay + add).
# Memory access is NOT counted for either side (state this as a limitation in the paper).


# ---------------- extra baseline: CNN that sees all time bins as channels ----------------
class StackCNN(nn.Module):
    snn_mode = False

    def __init__(self, ch, T):
        super().__init__()
        c1, c2, c3 = ch
        self.conv1 = nn.Conv2d(2 * T, c1, 3, padding=1)     # (2 polarities x T bins) input channels
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc = nn.Linear(c3 * 4 * 4, 10)

    def forward(self, x):                                   # x: (B, T, 2, 34, 34)
        h = x.flatten(1, 2)
        h = F.max_pool2d(F.relu(self.conv1(h)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        h = F.max_pool2d(F.relu(self.conv3(h)), 2)
        return self.fc(h.flatten(1))


S.CFG["CNN-Stack"] = ((64, 128, 256), "stack")
S.CFG["CNN-Stack-Small"] = ((16, 32, 64), "stack")
if getattr(S.build, "__name__", "") != "_build":            # patch once, even if this file is reloaded
    S._orig_build = S.build


def _build(name, T):
    ch, kind = S.CFG[name]
    if kind == "stack":
        return StackCNN(ch, T).to(S.dev)
    return S._orig_build(name, T)


S.build = _build


# ---------------- operation counts (no retraining needed) ----------------
@torch.no_grad()
def count_ops(model, te_dl, n=1000):
    """Per-sample counts. ANN: MACs. SNN: accumulates (spikes x fan-out) and neuron updates (neurons x time steps)."""
    model.eval(); tot = {"macs": 0.0, "ac": 0.0, "upd": 0.0}; hooks = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            fan = m.out_channels * k
            macs = out[0].numel() * m.in_channels * k
        else:
            fan = m.out_features; macs = m.in_features * m.out_features
        if model.snn_mode:
            tot["ac"] += inp[0].sum().item() * fan
            tot["upd"] += out.numel()
        else:
            tot["macs"] += macs * inp[0].shape[0]

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    seen = 0
    for x, y in te_dl:
        x, _ = S.prep(x, y); model(x); seen += len(x)
        if seen >= n: break
    for h in hooks: h.remove()
    return {k: v / seen for k, v in tot.items()}


def energy_uJ(r, prec, upd):
    e = E[prec]
    if r["snn"]:
        pj = r["ac"] * e["ac"] + (r["upd"] * e["mac"] if upd else 0.0)
    else:
        pj = r["macs"] * e["mac"]
    return pj * 1e-6


# ---------------- INT8 post-training quantization (simulated / fake-quant) ----------------
def _fq(t, scale):
    return torch.clamp(torch.round(t / scale), -128, 127) * scale


@torch.no_grad()
def quantize_model(model, calib_dl):
    """8-bit weights (per-output-channel, symmetric). CNNs also get 8-bit activations (per-tensor, calibrated on
    10 batches). SNN activations are already binary spikes, so only their weights are quantized."""
    q = copy.deepcopy(model).eval()
    layers = [m for m in q.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in layers:
        w = m.weight.data
        s = (w.abs().flatten(1).amax(1).clamp_min(1e-8) / 127).view(-1, *([1] * (w.dim() - 1)))
        m.weight.data = _fq(w, s)
    if not q.snn_mode:
        mx = {}

        def calib(m, i, o): mx[m] = max(mx.get(m, 0.0), o.abs().max().item())
        hs = [l.register_forward_hook(calib) for l in layers]
        for b, (x, y) in enumerate(calib_dl):
            x, _ = S.prep(x, y); q(x)
            if b >= 9: break
        for h in hs: h.remove()
        for l in layers:
            sc = max(mx[l], 1e-8) / 127
            l.register_forward_hook(lambda m, i, o, sc=sc: _fq(o, sc))
    return q


def review_seed():
    """For the current S.SEED: fp32 + INT8 accuracy and op counts of every trained (model, T). -> review.json"""
    d = S.sw_dir(); p = f"{d}/review.json"
    rv = json.load(open(p)) if os.path.exists(p) else {}
    for T in S.T_LIST:
        tr, te = S.get_loaders(T)
        for name in S.MODELS:
            key = f"{name}_T{T}"; fin = f"{d}/ckpt/{key}_final.pt"
            if key in rv: continue
            if not os.path.exists(fin):
                print(f"[{key}] not trained yet, skipping"); continue
            m = S.build(name, T); m.load_state_dict(torch.load(fin, map_location=S.dev, weights_only=False)); m.eval()
            c = count_ops(m, te); qm = quantize_model(m, tr)
            rv[key] = {"model": name, "T": T, "seed": S.SEED, "snn": bool(m.snn_mode),
                       "params": sum(x.numel() for x in m.parameters()),
                       "fp32_acc": round(S.accuracy(m, te), 2), "int8_acc": round(S.accuracy(qm, te), 2), **c}
            S.save_json(rv, p)
            print(f"[{key}] fp32 {rv[key]['fp32_acc']}%  int8 {rv[key]['int8_acc']}%")
    return rv


# ---------------- KD ablation ----------------
def kd_ablation(T=5, alphas=(0.3, 0.5, 0.7, 0.9), temps=(1.0, 2.0, 4.0, 8.0)):
    """Train KDSNN with different distillation weight alpha (temp 4) and temperature (alpha 0.7). Teacher: DeepSNN."""
    d = S.sw_dir(); p = f"{d}/ablation_kd_T{T}.json"
    ab = json.load(open(p)) if os.path.exists(p) else {}
    configs = [(a, 4.0) for a in alphas] + [(0.7, t) for t in temps if t != 4.0]
    tr, te = S.get_loaders(T)
    teacher = S.build("DeepSNN", T)
    teacher.load_state_dict(torch.load(f"{d}/ckpt/DeepSNN_T{T}_final.pt", map_location=S.dev, weights_only=False))
    teacher.eval()
    old = (S.KD_ALPHA, S.KD_T)
    try:
        for a, t in configs:
            key = f"KD_a{a}_t{t}"
            if key in ab: continue
            S.KD_ALPHA, S.KD_T = a, t
            hist = S.train_model(key, T, S.build("KDSNN", T), tr, te, teacher)
            ab[key] = {"alpha": a, "temp": t, "T": T, "seed": S.SEED, "test_acc": hist[-1]["test_acc"], "history": hist}
            S.save_json(ab, p)
    finally:
        S.KD_ALPHA, S.KD_T = old
    return ab


# ---------------- paper tables + plots ----------------
PAIRS = [("DeepSNN", "CNN"), ("DeepSNN", "CNN-Stack"), ("LiteSNN", "CNN-Small"), ("LiteSNN", "CNN-Stack-Small"),
         ("KDSNN", "CNN-Small"), ("KDSNN", "CNN-Stack-Small"), ("KDSNN", "LiteSNN")]


def paper_tables():
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except Exception:
        display = print
    try:
        from scipy import stats
    except Exception:
        stats = None
    out_dir = f"{S.ROOT}/paper_tables"; os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(f"{S.ROOT}/sweep_seed*/review.json")):
        rows += list(json.load(open(p)).values())
    if not rows:
        print("nothing yet - run R.review_seed() first"); return
    df = pd.DataFrame(rows)
    for prec in ("fp32", "int8"):
        for upd in (False, True):
            df[f"E_{prec}{'+upd' if upd else ''}"] = df.apply(lambda r: energy_uJ(r, prec, upd), axis=1)
    lat = []
    for p in glob.glob(f"{S.ROOT}/sweep_seed*/latency.json"):
        lat += list(json.load(open(p)).values())
    if lat:
        ld = pd.DataFrame(lat)
        ld["eager_ms"] = ld["eager_ms"].astype(float)
        ld["lat_ms"] = ld[["eager_ms", "fp16_ms", "compiled_ms"]].astype(float).min(axis=1)
        df = df.merge(ld.groupby(["model", "T"])[["eager_ms", "lat_ms"]].mean().reset_index(), on=["model", "T"], how="left")
    seeds = sorted(int(x) for x in df["seed"].unique()); order = [m for m in ALL if m in set(df["model"])]
    print("seeds:", seeds, "| mean ± std (sample std) over seeds when more than one seed")

    def fmt(nd):
        return lambda s: f"{s.mean():.{nd}f}" if len(s) == 1 else f"{s.mean():.{nd}f} ± {s.std(ddof=1):.{nd}f}"

    def tab(col, nd=2):
        return df.groupby(["model", "T"])[col].apply(fmt(nd)).unstack("T").reindex(order)

    tables = {"I_accuracy_fp32_%": tab("fp32_acc"), "II_accuracy_int8_%": tab("int8_acc")}
    for col, nm in [("E_fp32", "III_energy_uJ_fp32"), ("E_fp32+upd", "IV_energy_uJ_fp32_with_membrane_updates"),
                    ("E_int8", "V_energy_uJ_int8"), ("E_int8+upd", "VI_energy_uJ_int8_with_membrane_updates")]:
        tables[nm] = tab(col, 3)
    if "lat_ms" in df:
        tables["VII_latency_best_ms_bs1"] = tab("lat_ms", 3)
        tables["VIII_latency_eager_ms_bs1"] = tab("eager_ms", 3)
    for nm, t in tables.items():
        print(f"\n=== Table {nm} (rows: model, columns: time steps T) ==="); display(t)
        t.to_csv(f"{out_dir}/table_{nm}.csv")

    pr = []                                                  # paired comparisons over seeds
    for a, b in PAIRS:
        for T in sorted(df["T"].unique()):
            pa = df[(df.model == a) & (df["T"] == T)].set_index("seed")
            pb = df[(df.model == b) & (df["T"] == T)].set_index("seed")
            sd = pa.index.intersection(pb.index)
            if len(sd) == 0: continue
            dif = pa.loc[sd, "fp32_acc"] - pb.loc[sd, "fp32_acc"]
            pv = float(stats.ttest_rel(pa.loc[sd, "fp32_acc"], pb.loc[sd, "fp32_acc"]).pvalue) if (stats and len(sd) >= 3) else float("nan")
            pr.append({"T": T, "A": a, "B": b, "acc_diff_A-B": round(dif.mean(), 2), "A_wins": f"{int((dif > 0).sum())}/{len(sd)}",
                       "p_paired": round(pv, 3),
                       "energy_x_less_fp32": round((pb.loc[sd, "E_fp32"] / pa.loc[sd, "E_fp32"]).mean(), 1),
                       "energy_x_less_int8+upd": round((pb.loc[sd, "E_int8+upd"] / pa.loc[sd, "E_int8+upd"]).mean(), 1)})
    if pr:
        pdf = pd.DataFrame(pr)
        print("\n=== Table IX: paired comparison A vs B (accuracy difference, in how many seeds A is more accurate, energy ratio B/A) ===")
        display(pdf); pdf.to_csv(f"{out_dir}/table_IX_paired.csv", index=False)

    abl = []
    for p in sorted(glob.glob(f"{S.ROOT}/sweep_seed*/ablation_kd_T*.json")):
        abl += [{k: v for k, v in r.items() if k != "history"} for r in json.load(open(p)).values()]
    if abl:
        adf = pd.DataFrame(abl)
        t = adf.groupby(["T", "alpha", "temp"])["test_acc"].apply(fmt(2))
        print("\n=== Table X: KD ablation (KDSNN test accuracy %) ==="); display(t.to_frame("acc_%"))
        t.to_frame("acc_%").to_csv(f"{out_dir}/table_X_kd_ablation.csv")
        ref = df[df.model.isin(["LiteSNN", "KDSNN", "DeepSNN"])].groupby(["model", "T"])["fp32_acc"].apply(fmt(2)).unstack("T")
        print("reference from the main sweep (LiteSNN = no KD, KDSNN = alpha 0.7 / temp 4, DeepSNN = teacher):"); display(ref)

    sty = {"CNN": ("k", "s"), "CNN-Small": ("gray", "s"), "CNN-Stack": ("tab:orange", "^"), "CNN-Stack-Small": ("tab:brown", "^"),
           "DeepSNN": ("tab:blue", "o"), "LiteSNN": ("tab:green", "o"), "KDSNN": ("tab:red", "o")}
    g = df.groupby(["model", "T"]).agg(acc=("fp32_acc", "mean"), sd=("fp32_acc", "std"), e=("E_int8+upd", "mean"))
    if "lat_ms" in df: g["lat"] = df.groupby(["model", "T"])["lat_ms"].mean()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for m in order:
        if m not in g.index.get_level_values(0): continue
        s = g.loc[m]; c, mk = sty.get(m, (None, "o"))
        ax[0].errorbar(s["e"], s["acc"], yerr=s["sd"].fillna(0), marker=mk, color=c, label=m, capsize=2)
        for T, r in s.iterrows(): ax[0].annotate(f"T{T}", (r["e"], r["acc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        if "lat" in s:
            ax[1].errorbar(s["lat"], s["acc"], yerr=s["sd"].fillna(0), marker=mk, color=c, label=m, capsize=2)
            for T, r in s.iterrows(): ax[1].annotate(f"T{T}", (r["lat"], r["acc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].set_xlabel("energy per sample (uJ, INT8 + membrane updates)"); ax[0].set_ylabel("test accuracy (%)")
    ax[0].set_title("Accuracy vs energy"); ax[1].set_xlabel("latency (ms, batch 1, GPU)"); ax[1].set_title("Accuracy vs latency")
    for a_ in ax: a_.grid(alpha=.3); a_.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out_dir}/pareto.png", dpi=170); plt.show()
    df.to_csv(f"{out_dir}/all_runs.csv", index=False)
    print("saved all tables (CSV) and pareto.png in", out_dir)
