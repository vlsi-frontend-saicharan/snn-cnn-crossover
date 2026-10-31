"""snn_win.py - N-MNIST: can a LOW time-step SNN beat equal-size CNNs on accuracy, latency AND energy?

What this file does (one file, runs in Colab or PyCharm):
  * SNN students at T = 1 and 2 (widths S/M/L) trained with knowledge distillation from the DeepSNN (T=5) teachers
  * CNN baselines trained with the SAME recipe (same input, same epochs, same BatchNorm, same KD) -> the comparison is fair
  * input = event COUNTS (log1p) instead of clamped binary frames: binary clamping destroys information at T=1
  * BatchNorm is folded into the convs for inference, so it costs nothing in latency or energy
  * all data lives on the GPU as uint8 tensors -> training needs no DataLoader and is much faster
  * every experiment goes in its own folder  ROOT/win_runs/<RUN_TAG>/  together with a snapshot of THIS file,
    the config, the checkpoints, the per-run results, the test predictions, the latency file and the tables
  * summarize() prints WIN / TIE / LOSS for accuracy (McNemar), latency and energy for every SNN-vs-CNN pair

Typical use:
    import snn_win as W
    W.ROOT = "/content/drive/MyDrive/nmnist_snn_cnn_work"
    W.init_run("win_v1", note="count input + BN + KD from DeepSNN T5")
    W.teacher_logits()
    W.run_plan(seeds=[0])
    W.measure_latency()
    W.summarize()
"""
import os, sys, json, time, glob, copy, shutil, hashlib, datetime, platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================ CONFIG (edit from the notebook / PyCharm) ============================================
ROOT = "/content/drive/MyDrive/nmnist_snn_cnn_work"   # holds sweep_seed*/ (old results) and win_runs/ (new results)
OLD_ROOT = None            # folder with sweep_seed*/ckpt/DeepSNN_T5_final.pt (the teachers); None -> same as ROOT
RUN_TAG = "win_v1"         # one folder per experiment:  ROOT/win_runs/<RUN_TAG>/
RUN_NOTE = ""
DATA_DIR = "/content/data"         # tonic raw N-MNIST download
CACHE_DIR = "/content/cache_win"   # uint8 tensors built from the raw data (rebuilt after a disconnect, takes a few minutes)

EPOCHS = 20                # SAME for SNNs and CNNs
BATCH = 128
LR = 2e-3
WD = 1e-4
BETA = 0.9                 # LIF membrane decay (irrelevant at T=1)
KD_ALPHA, KD_TEMP = 0.7, 4.0
TEACHER_SEEDS = [0, 1, 2]  # DeepSNN T=5 checkpoints averaged as the teacher (missing seeds are skipped)
E_MAC, E_AC = 4.6, 0.9     # pJ, 32-bit float, 45 nm (Horowitz 2014). MAC = 3.7 mult + 0.9 add, AC = 0.9 add
WIDTHS = {"S": (16, 32, 64), "M": (32, 64, 128), "L": (64, 128, 256)}   # S = LiteSNN size, L = DeepSNN size

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_plan():
    """(kind, width, T, kd). Ordered so each width finishes its key comparison first; ablations last.
    CNN baselines use T=1: a CNN only sees the time-summed frame, so T does not change what it sees."""
    p = []
    for w in "SML":
        p += [("CNN", w, 1, True), ("SNN", w, 1, True), ("SNN", w, 2, True)]
    for w in "SML":
        p.append(("CNN", w, 1, False))
    p += [("SNN", "M", 1, False), ("SNN", "M", 2, False)]
    return p


def spec_name(spec):
    kind, w, T, kd = spec
    return f"{kind}-{w}_T{T}_{'kd' if kd else 'nokd'}"


# ============================================ run folder, code snapshot ============================================
def run_dir():
    return f"{ROOT}/win_runs/{RUN_TAG}"


def _p(*a):
    return os.path.join(run_dir(), *a)


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)          # atomic: never a half-written file


def init_run(tag=None, note=None):
    """Create the run folder, store a snapshot of this code (named by its hash) and the config, register in RUNS_INDEX.csv."""
    global RUN_TAG, RUN_NOTE
    if tag: RUN_TAG = tag
    if note is not None: RUN_NOTE = note
    for d in ("ckpt", "preds", "tables"):
        os.makedirs(_p(d), exist_ok=True)
    os.makedirs(f"{ROOT}/win_runs/_teacher_cache", exist_ok=True)
    src = os.path.abspath(__file__)
    h = hashlib.md5(open(src, "rb").read()).hexdigest()[:8]
    snap = _p(f"code_snapshot_snn_win_{h}.py")
    if not os.path.exists(snap):
        shutil.copyfile(src, snap)
    cfg = dict(run_tag=RUN_TAG, note=RUN_NOTE, code_hash=h, epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD, beta=BETA,
               kd_alpha=KD_ALPHA, kd_temp=KD_TEMP, teacher_seeds=TEACHER_SEEDS, widths=WIDTHS, e_mac_pJ=E_MAC, e_ac_pJ=E_AC,
               input_encoding="log1p(event counts per time bin)", plan=[spec_name(s) for s in default_plan()],
               torch=torch.__version__, python=platform.python_version(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               time=datetime.datetime.now().isoformat(timespec="seconds"))
    _save(cfg, _p("config.json"))
    with open(_p("history_log.txt"), "a") as f:
        f.write(f"{cfg['time']}  code {h}  epochs {EPOCHS}  kd_alpha {KD_ALPHA}  kd_temp {KD_TEMP}  lr {LR}\n")
    idx = f"{ROOT}/win_runs/RUNS_INDEX.csv"
    if not os.path.exists(idx):
        open(idx, "w").write("run_tag,created,code_hash,note\n")
    if f"\n{RUN_TAG}," not in "\n" + open(idx).read():
        open(idx, "a").write(f"{RUN_TAG},{cfg['time']},{h},\"{RUN_NOTE}\"\n")
    print(f"run folder: {run_dir()}\ncode snapshot: {os.path.basename(snap)}")
    return cfg


# ============================================ data: uint8 tensors on the GPU ============================================
_DATA = {}


def _build_tensors(T, split):
    import tonic, tonic.transforms as Tr
    ds = tonic.datasets.NMNIST(save_to=DATA_DIR, train=(split == "train"),
                               transform=Tr.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=T))
    dl = torch.utils.data.DataLoader(ds, batch_size=500, shuffle=False, num_workers=2)
    xs, ys, mx = [], [], 0
    for x, y in dl:
        mx = max(mx, int(x.max())); xs.append(x.clamp(0, 255).to(torch.uint8)); ys.append(y)
    print(f"  built T={T} {split}: max event count in one pixel/bin = {mx} (clipped at 255)")
    return torch.cat(xs), torch.cat(ys)


def get_data(T):
    """{'train': (X uint8 (N,T,2,34,34), Y), 'test': (...)} on the GPU. Cached on local disk after the first build."""
    if T in _DATA:
        return _DATA[T]
    d = {}
    for split in ("train", "test"):
        f = f"{CACHE_DIR}/nmnist_T{T}_{split}.pt"
        if os.path.exists(f):
            x, y = torch.load(f, weights_only=False)
        else:
            x, y = _build_tensors(T, split)
            os.makedirs(CACHE_DIR, exist_ok=True); torch.save((x, y), f)
        d[split] = (x.to(dev), y.to(dev))
    _DATA[T] = d
    return d


def _enc(xb):
    return torch.log1p(xb.float())          # event counts -> compressed analog values; the first layer is the encoding layer


# ============================================ teacher logits (DeepSNN T=5, trained earlier) ============================================
def teacher_logits():
    """Average the logits of the DeepSNN T=5 teachers (seeds in TEACHER_SEEDS) on train and test, once, and cache them.
    The teachers see exactly what they were trained on: binary frames from 5 time bins."""
    f = f"{ROOT}/win_runs/_teacher_cache/dsnn_t5_logits_seeds{''.join(map(str, TEACHER_SEEDS))}.pt"
    if os.path.exists(f):
        out = torch.load(f, map_location=dev, weights_only=False)
        print(f"teacher (cached) seeds {out['seeds']}: test acc {out['acc_test']:.2f}%")
        return out
    sys.path.insert(0, ROOT)
    import snn_sweep as TS                               # the Part 2 file: builds the old DeepSNN architecture
    base = OLD_ROOT or ROOT
    ms, used = [], []
    for s in TEACHER_SEEDS:
        p = f"{base}/sweep_seed{s}/ckpt/DeepSNN_T5_final.pt"
        if os.path.exists(p):
            m = TS.build("DeepSNN", 5)
            m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)); ms.append(m.eval()); used.append(s)
    if not ms:
        raise FileNotFoundError(f"no DeepSNN_T5_final.pt found under {base}/sweep_seed*/ckpt/ - needed as the KD teacher")
    d5 = get_data(5); out = {"seeds": used}
    with torch.no_grad():
        for split in ("train", "test"):
            X, Y = d5[split]; L = []
            for i in range(0, len(X), 250):
                xb = (X[i:i + 250] > 0).float()          # binary frames, same as the old clamp(0,1)
                L.append(torch.stack([m(xb) for m in ms]).mean(0))
            out[split] = torch.cat(L).float(); out[f"labels_{split}"] = Y
            out[f"acc_{split}"] = (out[split].argmax(1) == Y).float().mean().item() * 100
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out.items()}, f)
    _DATA.pop(5, None); torch.cuda.empty_cache()
    print(f"teacher seeds {used}: train acc {out['acc_train']:.2f}%  test acc {out['acc_test']:.2f}%")
    if out["acc_test"] < 98.0:
        print("!!! teacher test accuracy is low - check that the DeepSNN T=5 checkpoints are the right ones")
    return out


# ============================================ models ============================================
class _Spike(torch.autograd.Function):
    """Heaviside spike, fast-sigmoid surrogate gradient (slope 25)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (25.0 * x.abs() + 1.0) ** 2


spike_fn = _Spike.apply


class Net(nn.Module):
    """3 x (conv3x3 -> BN -> activation -> maxpool2) + FC.  spiking=True: LIF neurons over T bins; False: ReLU CNN.
    SNN: all T bins go through each conv in ONE batched call (feed-forward net), only the cheap LIF update loops over T.
    Readout = time-average of a leaky membrane, computed in closed form. At T=1 a layer is just spike = (BN(conv(x)) > 1)."""

    def __init__(self, ch, T, spiking):
        super().__init__()
        c1, c2, c3 = ch
        self.T, self.spiking = T, spiking
        self.conv1 = nn.Conv2d(2, c1, 3, padding=1, bias=False);  self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(c3)
        self.fc = nn.Linear(c3 * 4 * 4, 10)
        if spiking:
            k = torch.arange(T, dtype=torch.float32)
            self.register_buffer("ro_w", (1 - BETA ** (T - k)) / (1 - BETA) / T)

    def _blocks(self):
        return ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3))

    def _lif(self, c):                                   # c: (B, T, C, H, W) -> spikes (B*T, C, H, W)
        T = c.shape[1]
        if T == 1:
            return spike_fn(c[:, 0] - 1.0)
        mem = torch.zeros_like(c[:, 0]); out = []
        for t in range(T):
            mem = BETA * mem + c[:, t]
            s = spike_fn(mem - 1.0)
            mem = mem - s.detach()                        # subtract reset
            out.append(s)
        return torch.stack(out, 1).flatten(0, 1)

    def forward(self, x):                                # x: (B, T, 2, 34, 34), already encoded
        if not self.spiking:
            h = x.mean(1)
            for conv, bn in self._blocks():
                h = F.max_pool2d(F.relu(bn(conv(h))), 2)
            return self.fc(h.flatten(1))
        B, T = x.shape[0], x.shape[1]
        h = x.flatten(0, 1)
        for conv, bn in self._blocks():
            c = bn(conv(h))
            h = F.max_pool2d(self._lif(c.reshape(B, T, *c.shape[1:])), 2)
        cur = self.fc(h.flatten(1)).reshape(B, T, 10)
        return (cur * self.ro_w.view(1, T, 1)).sum(1)

    def fuse(self):
        """Fold BatchNorm into the convs (inference copy). Same outputs, no BN cost."""
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        m = copy.deepcopy(self).eval()
        for i in (1, 2, 3):
            setattr(m, f"conv{i}", fuse_conv_bn_eval(getattr(m, f"conv{i}"), getattr(m, f"bn{i}")))
            setattr(m, f"bn{i}", nn.Identity())
        return m


def build(spec):
    kind, w, T, kd = spec
    return Net(WIDTHS[w], T, kind == "SNN").to(dev)


# ============================================ training / evaluation ============================================
def kd_loss(s, t, y):
    soft = F.kl_div(F.log_softmax(s / KD_TEMP, 1), F.softmax(t / KD_TEMP, 1), reduction="batchmean") * KD_TEMP ** 2
    return KD_ALPHA * soft + (1 - KD_ALPHA) * F.cross_entropy(s, y)


@torch.no_grad()
def predict(model, X, bs=1000):
    model.eval()
    return torch.cat([model(_enc(X[i:i + bs])).argmax(1) for i in range(0, len(X), bs)])


def accuracy(model, X, Y):
    return 100.0 * (predict(model, X) == Y).float().mean().item()


@torch.no_grad()
def op_stats(model, X, n=1000):
    """Per-sample operation statistics of a (fused, eval) model on n real test samples."""
    model.eval(); st, hooks = {}, []

    def mk(name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1]; fan = m.out_channels * k; dense = out.numel() * m.in_channels * k
            else:
                fan = m.out_features; dense = out.numel() * m.in_features
            s = st.setdefault(name, {"dense": 0.0, "nnz_fan": 0.0, "neurons": 0.0})
            s["dense"] += dense                              # all multiply-accumulates if nothing is skipped
            s["nnz_fan"] += (x != 0).sum().item() * fan      # operations if every zero input is skipped
            s["neurons"] += out.numel()
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(mk(name)))
    seen = 0
    for i in range(0, min(n, len(X)), 500):
        xb = X[i:i + 500]; model(_enc(xb)); seen += len(xb)
    for h in hooks: h.remove()
    return {k: {a: b / seen for a, b in v.items()} for k, v in st.items()}


def energy_from_stats(st, spiking, e_mac=E_MAC, e_ac=E_AC):
    """Energy per sample in uJ.
    SNN  E_main  : conservative. Encoding layer (conv1, analog count input) = dense MACs; layers 2-4 = accumulates x incoming spikes;
                   plus one MAC-cost membrane update per spiking neuron per time step.
    SNN  E_evt   : same but the encoding layer skips zero inputs (event-driven first layer).
    CNN  E_main  : dense MACs.    CNN E_strict: MACs with every zero input skipped (the strongest fair baseline).
    Memory access is NOT counted for either side."""
    L = [st[n] for n in ("conv1", "conv2", "conv3", "fc")]
    if spiking:
        rest = sum(l["nnz_fan"] for l in L[1:]) * e_ac + sum(l["neurons"] for l in L[:3]) * e_mac
        main = (L[0]["dense"] * e_mac + rest) * 1e-6
        evt = (L[0]["nnz_fan"] * e_mac + rest) * 1e-6
        return {"E_main_uJ": main, "E_strict_uJ": main, "E_evt_uJ": evt}
    dense = sum(l["dense"] for l in L) * e_mac * 1e-6
    sparse = sum(l["nnz_fan"] for l in L) * e_mac * 1e-6
    return {"E_main_uJ": dense, "E_strict_uJ": sparse, "E_evt_uJ": sparse}


def train_one(spec, seed):
    kind, w, T, kd = spec
    key = f"{spec_name(spec)}_s{seed}"
    rp = _p("results.json"); res = _load(rp)
    if key in res:
        print(f"[{key}] already done, skipping"); return res[key]
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    d = get_data(T); Xtr, Ytr = d["train"]; Xte, Yte = d["test"]
    TL = None
    if kd:
        t = teacher_logits(); TL = t["train"].to(dev)
        assert torch.equal(t["labels_train"].to(dev), Ytr), "teacher and student datasets are not index-aligned"
    model = build(spec)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ck = _p("ckpt", f"{key}_ckpt.pt"); start, hist = 0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, hist = s["epoch"], s["hist"]; print(f"[{key}] resumed from epoch {start}")
    N = len(Xtr)
    for ep in range(start, EPOCHS):
        model.train(); t0 = time.time()
        g = torch.Generator(device=dev); g.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N, device=dev, generator=g); tot = torch.zeros((), device=dev); nb = 0
        for i in range(0, N - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]; x = _enc(Xtr[idx]); y = Ytr[idx]
            out = model(x)
            loss = kd_loss(out, TL[idx], y) if kd else F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.detach(); nb += 1
        sched.step()
        acc = accuracy(model, Xte, Yte)                      # monitoring only; the LAST epoch is reported (no best-epoch picking)
        hist.append({"epoch": ep + 1, "loss": float(tot) / nb, "test_acc": acc})
        print(f"[{key}] epoch {ep + 1}/{EPOCHS}  loss {float(tot) / nb:.4f}  test acc {acc:.2f}%  ({time.time() - t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "hist": hist}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    torch.save(model.state_dict(), _p("ckpt", f"{key}_final.pt"))
    fused = model.fuse()
    acc_f = accuracy(fused, Xte, Yte)
    if abs(acc_f - hist[-1]["test_acc"]) > 0.1:
        print(f"!!! fused accuracy {acc_f:.2f} differs from unfused {hist[-1]['test_acc']:.2f}")
    np.save(_p("preds", f"{key}.npy"), predict(fused, Xte).cpu().numpy().astype(np.uint8))
    lab = _p("preds", "labels_test.npy")
    if not os.path.exists(lab):
        np.save(lab, Yte.cpu().numpy().astype(np.uint8))
    en = energy_from_stats(op_stats(fused, Xte), kind == "SNN")
    res = _load(rp)
    res[key] = {"key": key, "spec": spec_name(spec), "kind": kind, "width": w, "T": T, "kd": kd, "seed": seed,
                "params": sum(p.numel() for p in fused.parameters()), "epochs": EPOCHS, "acc": round(acc_f, 2),
                **{k: round(v, 4) for k, v in en.items()}, "history": hist}
    _save(res, rp)
    print(f"[{key}] DONE  acc {acc_f:.2f}%  E_main {en['E_main_uJ']:.3f} uJ  E_strict {en['E_strict_uJ']:.3f} uJ")
    return res[key]


def run_plan(seeds=(0,), plan=None):
    """Train every (spec, seed). Resumable: finished runs are skipped, unfinished ones continue from their last epoch."""
    for seed in seeds:
        for spec in (plan or default_plan()):
            train_one(spec, seed)


# ============================================ latency: batch size 1, round-robin, eager + CUDA graph ============================================
def _median_ms(fn, iters, warm):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _make_graph(model, x):
    static = x.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): model(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(static)
    return g, static, out


@torch.no_grad()
def measure_latency(rounds=5, iters=300, warm=100, force=False):
    """Median latency per sample (batch 1, real test input, BN folded). All models are measured in round-robin so GPU clock
    drift hits them equally. eager = normal PyTorch; graph = CUDA graph replay (removes Python/launch overhead)."""
    res = _load(_p("results.json")); lp = _p("latency.json"); lat = _load(lp)
    best = {}
    for k, r in res.items():
        if r["spec"] not in best or r["seed"] < res[best[r["spec"]]]["seed"]:
            best[r["spec"]] = k
    todo = [n for n in best if force or n not in lat]
    if not todo:
        print("latency already measured for all specs"); return lat
    items = {}
    for n in todo:
        r = res[best[n]]
        m = Net(WIDTHS[r["width"]], r["T"], r["kind"] == "SNN").to(dev)
        m.load_state_dict(torch.load(_p("ckpt", f"{best[n]}_final.pt"), map_location=dev, weights_only=False))
        m = m.fuse().eval(); x = _enc(get_data(r["T"])["test"][0][:1]); g = None
        try:
            g = _make_graph(m, x)
        except Exception as e:
            print(f"[{n}] CUDA graph failed ({str(e)[:80]}), eager only")
        items[n] = (m, x, g)
    E = {n: [] for n in items}; G = {n: [] for n in items}
    for _ in range(rounds):
        for n, (m, x, g) in items.items():
            E[n].append(_median_ms(lambda: m(x), iters, warm))
            if g is not None:
                G[n].append(_median_ms(g[0].replay, iters, warm))
    gpu = torch.cuda.get_device_name(0)
    for n in items:
        lat[n] = {"eager_ms": round(float(np.median(E[n])), 4), "graph_ms": round(float(np.median(G[n])), 4) if G[n] else None,
                  "rounds": rounds, "iters": iters, "gpu": gpu}
        print(f"{n:22s} eager {lat[n]['eager_ms']:.3f} ms   graph {lat[n]['graph_ms']}")
    _save(lat, lp)
    return lat


# ============================================ verdict: SNN vs CNN, three criteria ============================================
def _md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:                                    # tabulate not installed
        return df.to_string(index=False)


def mcnemar(pa, pb, y):
    from scipy.stats import binomtest
    ca, cb = pa == y, pb == y
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return p, n10, n01


def compare(res, lat, labels, preds_dir, snn, cnn, band=0.05):
    """WIN/TIE/LOSS of SNN spec vs CNN spec. accuracy: per-seed McNemar p<0.05; latency: graph (else eager) ratio, +-5% = TIE;
    energy: CNN with zero-skipping (strict) vs SNN conservative, +-5% = TIE."""
    a = {r["seed"]: r for r in res.values() if r["spec"] == snn}
    b = {r["seed"]: r for r in res.values() if r["spec"] == cnn}
    seeds = sorted(set(a) & set(b))
    if not seeds or snn not in lat or cnn not in lat:
        return None
    wins = losses = 0; diffs = []
    for s in seeds:
        pa = np.load(f"{preds_dir}/{a[s]['key']}.npy"); pb = np.load(f"{preds_dir}/{b[s]['key']}.npy")
        p, _, _ = mcnemar(pa, pb, labels)
        d = a[s]["acc"] - b[s]["acc"]; diffs.append(d)
        if p < 0.05:
            wins += d > 0; losses += d < 0
    acc_v = "WIN" if wins > losses else "LOSS" if losses > wins else "TIE"
    la = lat[snn]["graph_ms"] or lat[snn]["eager_ms"]; lb = lat[cnn]["graph_ms"] or lat[cnn]["eager_ms"]
    lr = la / lb
    lat_v = "WIN" if lr < 1 - band else "LOSS" if lr > 1 + band else "TIE"
    ea = float(np.mean([a[s]["E_main_uJ"] for s in seeds])); eb = float(np.mean([b[s]["E_strict_uJ"] for s in seeds]))
    er = eb / ea
    en_v = "WIN" if er > 1 + band else "LOSS" if er < 1 - band else "TIE"
    vs = [acc_v, lat_v, en_v]
    overall = ("CLEAN WIN" if all(v == "WIN" for v in vs)
               else "NO LOSS + energy win" if ("LOSS" not in vs and en_v == "WIN") else "MIXED")
    return {"SNN": snn, "CNN": cnn, "seeds": len(seeds), "acc_diff_pts": round(float(np.mean(diffs)), 2),
            "acc_sig_wins/losses": f"{int(wins)}/{int(losses)}", "ACC": acc_v,
            "latency_ratio_snn/cnn": round(lr, 3), "LAT": lat_v, "energy_ratio_cnn/snn": round(er, 2), "ENERGY": en_v, "overall": overall}


def summarize():
    import pandas as pd
    res = _load(_p("results.json")); lat = _load(_p("latency.json"))
    if not res:
        print("no results yet"); return None
    try:
        from IPython.display import display
    except Exception:
        display = print
    rows = []
    for spec in sorted({r["spec"] for r in res.values()}):
        rs = [r for r in res.values() if r["spec"] == spec]; a = np.array([r["acc"] for r in rs])
        l = lat.get(spec, {})
        rows.append({"spec": spec, "kind": rs[0]["kind"], "width": rs[0]["width"], "T": rs[0]["T"], "kd": rs[0]["kd"],
                     "params": rs[0]["params"], "seeds": len(rs),
                     "acc_%": f"{a.mean():.2f}" if len(a) == 1 else f"{a.mean():.2f} ± {a.std(ddof=1):.2f}",
                     "latency_eager_ms": l.get("eager_ms"), "latency_graph_ms": l.get("graph_ms"),
                     "E_main_uJ": round(float(np.mean([r["E_main_uJ"] for r in rs])), 3),
                     "E_strict_uJ": round(float(np.mean([r["E_strict_uJ"] for r in rs])), 3)})
    tab = pd.DataFrame(rows).sort_values(["width", "kind", "T", "kd"]).reset_index(drop=True)
    print("=== accuracy / latency / energy (last-epoch test accuracy, mean ± std over seeds) ===")
    display(tab); tab.to_csv(_p("tables", "accuracy_latency_energy.csv"), index=False)
    labp = _p("preds", "labels_test.npy"); vd = None
    if os.path.exists(labp) and lat:
        labels = np.load(labp); out = []
        snns = [s for s in tab.spec if s.startswith("SNN")]; cnns = [s for s in tab.spec if s.startswith("CNN")]
        for sn in snns:
            for cn in cnns:
                if sn.split("_")[-1] != cn.split("_")[-1]:
                    continue                                     # same recipe only (kd vs kd, nokd vs nokd)
                v = compare(res, lat, labels, _p("preds"), sn, cn)
                if v: out.append(v)
        if out:
            vd = pd.DataFrame(out)
            vd["same_width"] = vd.apply(lambda r: r.SNN.split("-")[1][0] == r.CNN.split("-")[1][0], axis=1)
            vd = vd.sort_values(["same_width", "overall"], ascending=[False, True]).reset_index(drop=True)
            print("\n=== SNN vs CNN verdicts (same training recipe). ACC = McNemar on the same test samples; "
                  "ENERGY compares against the CNN with zero-skipping ===")
            display(vd); vd.to_csv(_p("tables", "verdicts.csv"), index=False)
            sw = vd[vd.same_width]                                   # headline = same width only (a bigger SNN vs a smaller CNN is not a fair win)
            wins = sw[sw.overall == "CLEAN WIN"]; nl = sw[sw.overall == "NO LOSS + energy win"]
            print(f"\nSAME-WIDTH pairs only.  CLEAN WIN (accuracy, latency and energy all WIN): {len(wins)} pair(s)")
            for _, r in wins.iterrows(): print("   ", r.SNN, "beats", r.CNN)
            print(f"no loss on any criterion and an energy win: {len(nl)} pair(s)")
            for _, r in nl.iterrows(): print("   ", r.SNN, "vs", r.CNN, "->", r.ACC, "acc /", r.LAT, "latency /", r.ENERGY, "energy")
    elif not lat:
        print("\n(run measure_latency() to get latency and the SNN-vs-CNN verdicts)")
    cfg = _load(_p("config.json"))
    with open(_p("RESULTS_SUMMARY.md"), "w") as f:
        f.write(f"# {RUN_TAG}\n\n{RUN_NOTE}\n\ncode hash {cfg.get('code_hash')}  |  epochs {EPOCHS}  |  KD alpha {KD_ALPHA} temp {KD_TEMP}  |  "
                f"gpu {cfg.get('gpu')}\n\n" + _md(tab) + "\n" + ("\n\n" + _md(vd) if vd is not None else ""))
    print("\nsaved tables + RESULTS_SUMMARY.md in", _p("tables"), "and", run_dir())
    return tab, vd


# ============================================ command line (PyCharm) ============================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="N-MNIST low-T SNN vs CNN win experiment")
    ap.add_argument("--root", default=ROOT); ap.add_argument("--tag", default=RUN_TAG); ap.add_argument("--note", default="")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--data-dir", default="./data"); ap.add_argument("--cache-dir", default="./cache_win")
    a = ap.parse_args()
    ROOT, EPOCHS, DATA_DIR, CACHE_DIR = a.root, a.epochs, a.data_dir, a.cache_dir
    init_run(a.tag, a.note); teacher_logits()
    run_plan(seeds=[int(s) for s in a.seeds.split(",")]); measure_latency(); summarize()
