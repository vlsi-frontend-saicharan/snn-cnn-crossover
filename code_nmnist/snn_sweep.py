"""nmnist_snn_cnn - time-step sweep (T = 1,2,3,5) for CNN / CNN-Small / DeepSNN / LiteSNN / KDSNN.

This file is saved in Google Drive next to the results, so the exact code behind every number is kept.
Usage (from the notebook):  import snn_sweep as S ; S.run_sweep() ; S.run_latency() ; S.summarize()
Everything is resumable: re-running skips finished (model, T) pairs and resumes unfinished ones from the last epoch.
"""
import os, json, time, glob, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ---------------- CONFIG (change from the notebook, e.g. S.EPOCHS = 20) ----------------
TITLE = "nmnist_snn_cnn"
ROOT = f"/content/drive/MyDrive/{TITLE}"
SEED = 0                      # run again with SEED = 1, 2 for mean +- std
T_LIST = [1, 2, 3, 5]         # SNN time steps (time bins) to sweep
EPOCHS = 10
BATCH = 128
LR = 1e-3
BETA = 0.9                    # LIF membrane decay
KD_T, KD_ALPHA = 4.0, 0.7     # distillation temperature / weight
E_MAC, E_AC = 4.6, 0.9        # pJ per MAC / AC (Horowitz 2014, 45 nm, 32-bit float)
MODELS = ["CNN", "CNN-Small", "DeepSNN", "LiteSNN", "KDSNN"]   # DeepSNN must come before KDSNN
CFG = {                       # (channels of the 3 conv layers, is_snn)
    "CNN":       ((64, 128, 256), False),
    "CNN-Small": ((16, 32, 64),   False),   # same size as LiteSNN -> fair baseline
    "DeepSNN":   ((64, 128, 256), True),
    "LiteSNN":   ((16, 32, 64),   True),
    "KDSNN":     ((16, 32, 64),   True),    # LiteSNN trained with distillation from DeepSNN
}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sw_dir():
    d = f"{ROOT}/sweep_seed{SEED}"
    os.makedirs(f"{d}/ckpt", exist_ok=True)
    return d


def save_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)        # atomic: never a half-written file


# ---------------- MODELS ----------------
class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside spike in forward, fast-sigmoid surrogate gradient (slope 25) in backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (25.0 * x.abs() + 1.0) ** 2


spike_fn = FastSigmoidSpike.apply


class SNNNet(nn.Module):
    """3 conv blocks + FC with LIF neurons (threshold 1, subtract-reset).
    LATENCY TRICK: the net is feed-forward, so each conv is applied to ALL time steps in one batched call
    (time folded into the batch), and only the cheap element-wise LIF update loops over time.
    The output layer is a leaky readout whose time-average is computed in closed form (no loop).
    Mathematically identical to the usual step-by-step loop, but far fewer GPU kernel launches."""
    snn_mode = True

    def __init__(self, ch, T):
        super().__init__()
        c1, c2, c3 = ch
        self.T = T
        self.conv1 = nn.Conv2d(2, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc = nn.Linear(c3 * 4 * 4, 10)
        k = torch.arange(T, dtype=torch.float32)
        # mean_t of the leaky membrane = (1/T) * sum_k w_k * current_k,  w_k = (1 - beta^(T-k)) / (1 - beta)
        self.register_buffer("ro_w", (1 - BETA ** (T - k)) / (1 - BETA) / T)

    def forward(self, x):                              # x: (B, T, 2, 34, 34)
        B, T = x.shape[0], x.shape[1]
        h = x.flatten(0, 1)                            # (B*T, 2, 34, 34)
        for conv in (self.conv1, self.conv2, self.conv3):
            c = conv(h)                                # all time steps at once
            c = c.reshape(B, T, *c.shape[1:])
            mem = torch.zeros_like(c[:, 0])
            out = []
            for t in range(T):                         # cheap element-wise LIF loop
                mem = BETA * mem + c[:, t]
                s = spike_fn(mem - 1.0)
                mem = mem - s.detach()                 # subtract reset
                out.append(s)
            h = F.max_pool2d(torch.stack(out, 1).flatten(0, 1), 2)
        cur = self.fc(h.flatten(1)).reshape(B, T, 10)
        return (cur * self.ro_w.view(1, T, 1)).sum(1)


class CNNNet(nn.Module):
    """Same 3 conv blocks + FC with ReLU; input is the time-averaged event frame."""
    snn_mode = False

    def __init__(self, ch, T=None):
        super().__init__()
        c1, c2, c3 = ch
        self.conv1 = nn.Conv2d(2, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc = nn.Linear(c3 * 4 * 4, 10)

    def forward(self, x):
        h = x.mean(1)
        h = F.max_pool2d(F.relu(self.conv1(h)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        h = F.max_pool2d(F.relu(self.conv3(h)), 2)
        return self.fc(h.flatten(1))


def build(name, T):
    ch, is_snn = CFG[name]
    return (SNNNet(ch, T) if is_snn else CNNNet(ch)).to(dev)


# ---------------- DATA ----------------
def get_loaders(T):
    import tonic, tonic.transforms as Tr
    tf = Tr.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=T)
    tr = tonic.datasets.NMNIST(save_to="/content/data", train=True, transform=tf)
    te = tonic.datasets.NMNIST(save_to="/content/data", train=False, transform=tf)
    tr = tonic.DiskCachedDataset(tr, cache_path=f"/content/cache/train_T{T}")   # frames cached after 1st epoch
    te = tonic.DiskCachedDataset(te, cache_path=f"/content/cache/test_T{T}")
    tr_dl = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    te_dl = torch.utils.data.DataLoader(te, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    return tr_dl, te_dl


def prep(x, y):
    return x.to(dev).float().clamp(0, 1), y.to(dev)      # binary spike frames


# ---------------- TRAIN / EVAL ----------------
def kd_loss(s, t, y):
    soft = F.kl_div(F.log_softmax(s / KD_T, 1), F.softmax(t / KD_T, 1), reduction="batchmean") * KD_T ** 2
    return KD_ALPHA * soft + (1 - KD_ALPHA) * F.cross_entropy(s, y)


@torch.no_grad()
def accuracy(model, dl):
    model.eval(); c = n = 0
    for x, y in dl:
        x, y = prep(x, y)
        c += (model(x).argmax(1) == y).sum().item(); n += len(y)
    return 100.0 * c / n


def train_model(name, T, model, tr_dl, te_dl, teacher=None):
    d = sw_dir(); ck = f"{d}/ckpt/{name}_T{T}_ckpt.pt"; tag = f"{name} T={T}"
    torch.manual_seed(SEED); np.random.seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    start, hist = 0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, hist = s["epoch"], s["hist"]
        print(f"[{tag}] resumed from epoch {start}")
    for ep in range(start, EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = prep(x, y); out = model(x)
            if teacher is not None:
                with torch.no_grad():
                    tout = teacher(x)
                loss = kd_loss(out, tout, y)
            else:
                loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        acc = accuracy(model, te_dl)
        hist.append({"epoch": ep + 1, "loss": tot / len(tr_dl), "test_acc": acc})
        print(f"[{tag}] epoch {ep+1}/{EPOCHS}  loss {tot/len(tr_dl):.4f}  test acc {acc:.2f}%  ({time.time()-t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "hist": hist}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    torch.save(model.state_dict(), f"{d}/ckpt/{name}_T{T}_final.pt")
    return hist


@torch.no_grad()
def ops_per_sample(model, te_dl, n=1000):
    """CNN: MACs. SNN: accumulates = incoming spikes x fan-out (inputs are binary, so no multiplies)."""
    model.eval(); total = [0.0]; hooks = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            fan = m.out_channels * k
            macs = out[0].numel() * m.in_channels * k
        else:
            fan = m.out_features; macs = m.in_features * m.out_features
        total[0] += (inp[0].sum().item() * fan) if model.snn_mode else macs * inp[0].shape[0]

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    seen = 0
    for x, y in te_dl:
        x, _ = prep(x, y); model(x); seen += len(x)
        if seen >= n: break
    for h in hooks: h.remove()
    return total[0] / seen


def full_eval(name, T, model, te_dl):
    ops = ops_per_sample(model, te_dl)
    e_pj = ops * (E_AC if model.snn_mode else E_MAC)
    return {"model": name, "T": T, "seed": SEED,
            "params": sum(p.numel() for p in model.parameters()),
            "test_acc": round(accuracy(model, te_dl), 2),
            "time_steps": T if model.snn_mode else 1,
            "ops_per_sample": round(ops), "op_type": "AC" if model.snn_mode else "MAC",
            "energy_uJ": round(e_pj * 1e-6, 4)}


def run_sweep():
    """Train + evaluate every (model, T). Saves results.json after each pair."""
    d = sw_dir(); rp = f"{d}/results.json"
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    print("device:", dev, "| seed:", SEED, "| T_LIST:", T_LIST, "| epochs:", EPOCHS)
    for T in T_LIST:
        tr_dl, te_dl = get_loaders(T)
        for name in MODELS:
            key = f"{name}_T{T}"
            if key in res:
                print(f"[{key}] already done, skipping"); continue
            model = build(name, T); teacher = None
            if name == "KDSNN":
                teacher = build("DeepSNN", T)
                teacher.load_state_dict(torch.load(f"{d}/ckpt/DeepSNN_T{T}_final.pt", map_location=dev, weights_only=False))
                teacher.eval()
            hist = train_model(name, T, model, tr_dl, te_dl, teacher)
            res[key] = {**full_eval(name, T, model, te_dl), "history": hist}
            save_json(res, rp)
            print(f"[{key}] DONE ->", {k: v for k, v in res[key].items() if k != "history"})
    return res


# ---------------- LATENCY (batch size 1) ----------------
def _sync():
    if dev.type == "cuda": torch.cuda.synchronize()


@torch.no_grad()
def _time(fn, x, n, warm=30):
    for _ in range(warm): fn(x)
    _sync(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(x); _sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(ts)), 4)               # median = robust to Colab noise


def run_latency(n=300, compile_model=True):
    """Median per-sample latency (ms, batch 1) for every trained (model, T) in 3 modes:
       eager fp32 | fp16 | torch.compile with CUDA graphs ('reduce-overhead').  Saved to latency.json."""
    d = sw_dir(); lp = f"{d}/latency.json"
    lat = json.load(open(lp)) if os.path.exists(lp) else {}
    for T in T_LIST:
        _, te_dl = get_loaders(T)
        x = next(iter(te_dl))[0][:1].to(dev).float().clamp(0, 1)
        for name in MODELS:
            key = f"{name}_T{T}"
            if key in lat: continue
            fin = f"{d}/ckpt/{key}_final.pt"
            if not os.path.exists(fin):
                print(f"[{key}] not trained yet, skipping"); continue
            m = build(name, T); m.load_state_dict(torch.load(fin, map_location=dev, weights_only=False)); m.eval()
            r = {"model": name, "T": T, "seed": SEED, "eager_ms": _time(m, x, n), "fp16_ms": None, "compiled_ms": None}
            if dev.type == "cuda":
                try:
                    mh = copy.deepcopy(m).half(); r["fp16_ms"] = _time(mh, x.half(), n)
                except Exception as e:
                    print(f"[{key}] fp16 failed: {str(e)[:100]}")
                if compile_model:
                    try:
                        import torch._dynamo as dyn; dyn.reset()
                        cm = torch.compile(m, mode="reduce-overhead")
                        with torch.no_grad():
                            for _ in range(3): cm(x)      # triggers compilation
                        r["compiled_ms"] = _time(cm, x, n)
                    except Exception as e:
                        print(f"[{key}] compile failed (eager/fp16 still saved): {str(e)[:100]}")
            lat[key] = r; save_json(lat, lp)
            print(f"[{key}]", {k: v for k, v in r.items() if k.endswith("_ms")})
    return lat


# ---------------- SUMMARY: tables + plots over all seeds found in Drive ----------------
def summarize():
    import pandas as pd
    import matplotlib; import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except Exception:
        display = print
    rows, lrows = [], []
    for p in sorted(glob.glob(f"{ROOT}/sweep_seed*/results.json")):
        rows += [{k: v for k, v in r.items() if k != "history"} for r in json.load(open(p)).values()]
        lp = p.replace("results.json", "latency.json")
        if os.path.exists(lp): lrows += list(json.load(open(lp)).values())
    if not rows:
        print("no results yet - run S.run_sweep() first"); return
    df = pd.DataFrame(rows)
    seeds = sorted(df["seed"].unique()); print("seeds found:", seeds)
    order = [m for m in MODELS if m in set(df["model"])]

    def fmt(g):
        return f"{g.mean():.2f}" if len(g) == 1 else f"{g.mean():.2f} ± {g.std(ddof=0):.2f}"

    def table(frame, col):
        return frame.groupby(["model", "T"])[col].apply(fmt).unstack("T").reindex(order)

    out = {"accuracy_%": table(df, "test_acc"), "energy_uJ": table(df, "energy_uJ")}
    mean_tbl = {"accuracy_%": df.groupby(["model", "T"])["test_acc"].mean().unstack("model"),
                "energy_uJ": df.groupby(["model", "T"])["energy_uJ"].mean().unstack("model")}
    if lrows:
        ldf = pd.DataFrame(lrows)
        ldf["best_ms"] = ldf[["eager_ms", "fp16_ms", "compiled_ms"]].astype(float).min(axis=1)
        for c, nm in [("eager_ms", "latency_eager_ms"), ("fp16_ms", "latency_fp16_ms"),
                      ("compiled_ms", "latency_compiled_ms"), ("best_ms", "latency_best_ms")]:
            if ldf[c].notna().any():
                out[nm] = table(ldf.dropna(subset=[c]), c)
        mean_tbl["latency_best_ms"] = ldf.groupby(["model", "T"])["best_ms"].mean().unstack("model")
    for nm, t in out.items():
        print(f"\n=== {nm} (rows: model, columns: time steps T) ==="); display(t)
        t.to_csv(f"{ROOT}/sweep_{nm}.csv")
    df.to_csv(f"{ROOT}/sweep_all_runs.csv", index=False)

    print("\n=== winner per time step (by seed-mean) ===")
    for T in sorted(df["T"].unique()):
        line = f"T={T}:"
        a = mean_tbl["accuracy_%"].loc[T].dropna();  line += f"  best acc = {a.idxmax()} ({a.max():.2f}%)"
        e = mean_tbl["energy_uJ"].loc[T].dropna();   line += f" | lowest energy = {e.idxmin()} ({e.min():.2f} uJ)"
        if "latency_best_ms" in mean_tbl and T in mean_tbl["latency_best_ms"].index:
            l = mean_tbl["latency_best_ms"].loc[T].dropna()
            if len(l): line += f" | lowest latency = {l.idxmin()} ({l.min():.3f} ms)"
        print(line)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8)); ax = ax.ravel()
    style = {"CNN": ("k", "--"), "CNN-Small": ("gray", "--"), "DeepSNN": ("tab:blue", "-"),
             "LiteSNN": ("tab:green", "-"), "KDSNN": ("tab:red", "-")}
    panels = [("accuracy_%", "Accuracy (%)", False), ("energy_uJ", "Energy per sample (uJ, est.)", True),
              ("latency_best_ms", "Latency (ms, batch 1, best mode)", False)]
    for a_, (k, ttl, logy) in zip(ax, panels):
        if k not in mean_tbl: a_.set_visible(False); continue
        for m in order:
            if m in mean_tbl[k]:
                c, ls = style.get(m, (None, "-")); mean_tbl[k][m].plot(ax=a_, marker="o", color=c, ls=ls, label=m)
        a_.set_title(ttl); a_.set_xlabel("time steps T"); a_.set_xticks(sorted(df["T"].unique()))
        if logy: a_.set_yscale("log")
        a_.legend(); a_.grid(alpha=.3)
    if "latency_best_ms" in mean_tbl:                       # accuracy vs latency trade-off
        for m in order:
            if m in mean_tbl["latency_best_ms"]:
                c, ls = style.get(m, (None, "-"))
                ax[3].plot(mean_tbl["latency_best_ms"][m], mean_tbl["accuracy_%"][m], marker="o", color=c, ls=ls, label=m)
        ax[3].set_xlabel("latency (ms)"); ax[3].set_ylabel("accuracy (%)"); ax[3].set_title("Accuracy vs latency")
        ax[3].legend(); ax[3].grid(alpha=.3)
    else:
        ax[3].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{ROOT}/sweep_plots.png", dpi=150); plt.show()
    print("saved CSVs + sweep_plots.png in", ROOT)
