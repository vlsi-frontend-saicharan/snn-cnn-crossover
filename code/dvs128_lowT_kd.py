"""
DVS128 Gesture - low-timestep SNN + knowledge distillation (code version: lowT_kd_v1, 2026-10-19)

Question tested: can a KD-trained SNN at T=1 or T=2 match/beat the CNN on
accuracy, measured GPU latency and measured GPU energy, under ONE protocol?

Controls (so a win is clean, not a KD or capacity artefact):
  cnn3 plain / kd  : the paper's 3-block CNN (stacked frames as channels)
  cnn4 plain / kd  : 4-block CNN, architecture-matched to the SNN (ReLU instead of LIF)
All models: same data, same augmentation, same epochs, same optimiser, same seeds.
Outputs: <ROOT>/v2_lowT_kd/{results,ckpt,measure,ops,summary,logs,code,cache}/ + README.md
"""
import os, sys, json, time, math, shutil, hashlib, datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

VERSION = 'lowT_kd_v1'
ROOT = os.environ.get('DVS_ROOT', '/content/drive/MyDrive/dvs128_gesture_snn_cnn')
EXP = os.environ.get('DVS_EXP', os.path.join(ROOT, 'v2_lowT_kd'))  # permanent (Drive); override with env DVS_EXP
LOCAL = os.environ.get('DVS_LOCAL', '/content/dvs_v2_local')  # scratch (local disk)
PACKED = os.path.join(ROOT, 'packed.npz')
TEACHERS = [('snn_deep_T16', 93.75), ('snn_deep_T16_s1', 94.79), ('snn_deep_T16_s2', 94.10)]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
E_MAC, E_AC = 4.6, 0.9                                      # pJ, 45 nm (Horowitz 2014)
SHIFTS = [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (-3, -3), (3, -3), (-3, 3)]
CFG = dict(tag='v1', epochs=100, bs=32, lr=1e-3, wd=0.01, warmup=5, ls=0.1, drop=0.3,
           kd_alpha=0.7, kd_T=4.0, aug=True)
torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------- specs / ids
def S(arch, T, K, w, kd):
    return dict(arch=arch, T=T, K=K, w=w, kd=kd)          # B = T*K temporal bins


def sid(s):
    return f"{s['arch']}_T{s['T']}K{s['K']}_w{s['w']:g}_{'kd' if s['kd'] else 'plain'}"


def rid(s, seed):
    return f"{CFG['tag']}_{sid(s)}_s{seed}"


SPECS = {}
for _s in [S('cnn3', 1, 4, 1.0, False), S('cnn3', 1, 4, 1.0, True)]:
    SPECS[sid(_s)] = _s
for _w in (1.0, 0.5):
    for _s in [S('cnn4', 1, 4, _w, False), S('cnn4', 1, 4, _w, True), S('cnn4', 1, 8, _w, True)]:
        SPECS[sid(_s)] = _s
for _T, _K in ((1, 4), (1, 8), (2, 2), (2, 4)):
    for _w in (1.0, 0.5):
        _s = S('snn4', _T, _K, _w, True)
        SPECS[sid(_s)] = _s
for _T, _K, _w in ((1, 4, 1.0), (1, 4, 0.5), (2, 2, 1.0)):
    _s = S('snn4', _T, _K, _w, False)
    SPECS[sid(_s)] = _s


def baselines_for(s):
    B = s['T'] * s['K']
    ids = ['cnn3_T1K4_w1_plain', 'cnn3_T1K4_w1_kd',
           f"cnn4_T1K{B}_w{s['w']:g}_plain", f"cnn4_T1K{B}_w{s['w']:g}_kd"]
    return [i for i in ids if i in SPECS]


# ----------------------------------------------------------------- files (Drive-safe)
def _mk(p):
    try:
        os.makedirs(p, exist_ok=True)
    except OSError as e:
        print('WARN: cannot create', p, '->', e)


def _drive_copy(local_path, rel):
    dst = os.path.join(EXP, rel)
    for _ in range(3):
        try:
            _mk(os.path.dirname(dst))
            shutil.copyfile(local_path, dst)
            return True
        except Exception:
            time.sleep(2)
    print('WARN: Drive copy failed, kept local only:', rel)
    return False


def save_json(rel, obj):
    p = os.path.join(LOCAL, rel)
    _mk(os.path.dirname(p))
    with open(p, 'w') as f:
        json.dump(obj, f)
    _drive_copy(p, rel)


def read_json(rel):
    for base in (EXP, LOCAL):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def done(r):
    return read_json(f'results/{r}.json') is not None


def ckpt_file(r):
    for base in (LOCAL, EXP):
        p = os.path.join(base, 'ckpt', f'{r}.pth')
        if os.path.exists(p):
            return p
    return None


def sync_to_drive():
    n = 0
    for dp, _, fs in os.walk(LOCAL):
        for fn in fs:
            lp = os.path.join(dp, fn)
            rel = os.path.relpath(lp, LOCAL)
            if not os.path.exists(os.path.join(EXP, rel)):
                n += _drive_copy(lp, rel)
    print('sync_to_drive: copied', n, 'files')


def code_sha():
    try:
        return hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:8]
    except Exception:
        return 'unknown'


def setup():
    for d in ('results', 'ckpt', 'measure', 'ops', 'summary', 'logs', 'code', 'cache'):
        _mk(os.path.join(EXP, d)); _mk(os.path.join(LOCAL, d))
    try:
        _drive_copy(__file__, f'code/dvs128_{VERSION}.py')
    except Exception:
        pass
    readme = f"""# DVS128 low-T KD experiment ({VERSION})
code/      dvs128_{VERSION}.py  = the exact code that produced these results (also paste into PyCharm)
results/   one JSON per run: <tag>_<arch>_T<T>K<K>_w<width>_<kd|plain>_s<seed>.json  (final-epoch test acc + test predictions)
ckpt/      final weights of each run (.pth)
measure/   latency + NVML energy JSONs (tag = screen | final)
ops/       op-count estimate + measured spike rates per model
summary/   tables written by report()
logs/      runs.csv (one line per finished run)
cache/     teacher logits bank
Naming: arch snn4 = 4-block spiking net; cnn3 = paper 3-block CNN; cnn4 = 4-block CNN matched to snn4.
T = SNN timesteps; K = temporal bins stacked into channels per timestep; B = T*K bins of 16//B frames each.
kd = trained with distillation from 3 Deep-SNN T=16 teachers (snn_deep_T16, _s1, _s2); plain = CE only.
Protocol: {json.dumps(CFG)}
Reported accuracy = FINAL epoch (never best). Energy = NVML board power minus idle, batch 64.
"""
    p = os.path.join(LOCAL, 'README.md')
    open(p, 'w').write(readme)
    _drive_copy(p, 'README.md')


# ----------------------------------------------------------------- data
_D = {}


def data():
    if not _D:
        z = np.load(PACKED)
        for k in ('tr_snn', 'te_snn'):
            _D[k] = torch.from_numpy(z[k]).to(DEV)                    # float16 (N,16,2,48,48)
        for k in ('tr_y', 'te_y'):
            _D[k] = torch.from_numpy(np.asarray(z[k])).long().to(DEV)
    return _D


def make_inputs(raw, s):
    """raw (N,16,2,H,W) float -> model input. Same prep as the paper: sum groups, log1p, per-sample max."""
    N, _, C, H, W = raw.shape
    B = s['T'] * s['K']
    x = raw.reshape(N, B, 16 // B, C, H, W).sum(2)
    x = torch.log1p(x)
    x = x / x.flatten(1).amax(1).clamp_min(1e-6).view(N, 1, 1, 1, 1)
    if s['arch'] == 'snn4':
        return x.reshape(N, s['T'], s['K'] * C, H, W)
    return x.flatten(1, 2)                                              # (N, 2B, H, W)


def shift_xy(x, dx, dy):
    if dx == 0 and dy == 0:
        return x
    H, W = x.shape[-2:]
    out = torch.zeros_like(x)
    ys, ye, xs, xe = max(0, dy), H + min(0, dy), max(0, dx), W + min(0, dx)
    out[..., ys:ye, xs:xe] = x[..., ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def apply_shifts(raw, sidx):
    out = raw.clone()
    for k in torch.unique(sidx).tolist():
        if k == 0:
            continue
        m = sidx == k
        out[m] = shift_xy(raw[m], *SHIFTS[k])
    return out


# ----------------------------------------------------------------- models
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors
        return g / (1 + (math.pi * v) ** 2)                             # arctan surrogate


class LIFSeq(nn.Module):
    def __init__(self, beta=0.5, thr=1.0):
        super().__init__()
        self.beta, self.thr = beta, thr

    def forward(self, x):                                               # (B,T,...) -> spikes
        v = torch.zeros_like(x[:, 0])
        out = []
        for t in range(x.shape[1]):
            v = self.beta * v + x[:, t]
            s = SpikeFn.apply(v - self.thr)
            v = v - s.detach() * self.thr                               # soft reset
            out.append(s)
        return torch.stack(out, 1)


def ch(c, w):
    return max(4, int(round(c * w)))


class DeepSNN(nn.Module):
    """4 x [conv3x3 s1 -> BN -> LIF -> maxpool2] -> dropout -> FC, logits averaged over T (teacher = in_ch 2, w 1)."""
    def __init__(self, in_ch=2, w=1.0, ncls=11, drop=0.3):
        super().__init__()
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        cin = in_ch
        for c in [ch(c, w) for c in (32, 64, 128, 128)]:
            self.convs.append(nn.Conv2d(cin, c, 3, 1, 1, bias=False))
            self.bns.append(nn.BatchNorm2d(c))
            cin = c
        self.lif, self.drop, self.fc = LIFSeq(), nn.Dropout(drop), nn.Linear(cin * 9, ncls)

    def forward(self, x):                                               # (B,T,C,H,W)
        B, T = x.shape[:2]
        for conv, bn in zip(self.convs, self.bns):
            h = bn(conv(x.flatten(0, 1)))
            x = self.lif(h.view(B, T, *h.shape[1:]))
            y = F.max_pool2d(x.flatten(0, 1), 2)
            x = y.view(B, T, *y.shape[1:])
        return self.fc(self.drop(x.flatten(2))).mean(1)


class CNN(nn.Module):
    """cnn3: paper baseline (3 blocks + FC128). cnn4: same layout as DeepSNN with ReLU."""
    def __init__(self, arch, in_ch, w=1.0, ncls=11, drop=0.3):
        super().__init__()
        chans = (32, 64, 128) if arch == 'cnn3' else (32, 64, 128, 128)
        bias = arch == 'cnn3'
        L, cin = [], in_ch
        for c in [ch(c, w) for c in chans]:
            L += [nn.Conv2d(cin, c, 3, 1, 1, bias=bias), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*L)
        if arch == 'cnn3':
            hid = ch(128, w)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(cin * 36, hid), nn.ReLU(),
                                      nn.Dropout(drop), nn.Linear(hid, ncls))
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(cin * 9, ncls))

    def forward(self, x):
        return self.head(self.features(x))


def build(s):
    if s['arch'] == 'snn4':
        return DeepSNN(2 * s['K'], s['w'], drop=CFG['drop'])
    return CNN(s['arch'], 2 * s['T'] * s['K'], s['w'], drop=CFG['drop'])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- teacher / KD bank
@torch.no_grad()
def evaluate(model, s, split='te', bs=64):
    model.eval()
    d = data()
    X, Y = d[f'{split}_snn'], d[f'{split}_y']
    P = []
    for i in range(0, len(Y), bs):
        P.append(model(make_inputs(X[i:i + bs].float(), s)).argmax(1))
    P = torch.cat(P)
    return (P == Y).float().mean().item() * 100, P.cpu().tolist()


TSPEC = dict(arch='snn4', T=16, K=1, w=1.0, kd=False)


def load_teacher(name, expected):
    p = os.path.join(ROOT, 'checkpoints', f'{name}.pth')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    m = DeepSNN(2, 1.0)
    try:
        m.load_state_dict(sd)
    except Exception:
        print('checkpoint keys:', list(sd.keys())[:14])
        print('my keys        :', list(m.state_dict().keys())[:14])
        raise
    m.to(DEV).eval()
    acc, _ = evaluate(m, TSPEC, 'te', bs=32)
    print(f'teacher {name}: test acc {acc:.2f}  (stored {expected})')
    if abs(acc - expected) > 1.0:
        raise RuntimeError(f'teacher {name} does not reproduce its stored accuracy -> preprocessing/model mismatch; '
                           'send this output before training anything')
    return m


@torch.no_grad()
def build_bank(force=False):
    d = data()
    N = d['tr_y'].shape[0]
    key = hashlib.md5(('|'.join(n for n, _ in TEACHERS) + str(SHIFTS) + str(N)).encode()).hexdigest()[:8]
    rel = f'cache/teacher_bank_{key}.pt'
    for base in (EXP, LOCAL):
        p = os.path.join(base, rel)
        if os.path.exists(p) and not force:
            print('teacher bank loaded from', p)
            return torch.load(p, map_location=DEV, weights_only=False)
    ts = [load_teacher(n, a) for n, a in TEACHERS]
    bank = torch.zeros(len(SHIFTS), N, 11, device=DEV)
    t0 = time.time()
    for si, (dx, dy) in enumerate(SHIFTS):
        for i in range(0, N, 32):
            x = make_inputs(shift_xy(d['tr_snn'][i:i + 32].float(), dx, dy), TSPEC)
            bank[si, i:i + 32] = sum(t(x) for t in ts) / len(ts)
        print(f'  bank shift {si + 1}/{len(SHIFTS)}  {time.time() - t0:.0f}s')
    p = os.path.join(LOCAL, rel)
    _mk(os.path.dirname(p))
    torch.save(bank, p)
    _drive_copy(p, rel)
    return bank


# ----------------------------------------------------------------- training
def lr_at(e, E, base, warm):
    if e < warm:
        return base * (e + 1) / warm
    return 0.5 * base * (1 + math.cos(math.pi * (e - warm) / max(1, E - warm)))


def train_one(s, seed, bank=None, epochs=None, persist=True, verbose=True):
    r = rid(s, seed)
    if persist and done(r):
        return read_json(f'results/{r}.json')
    E = epochs or CFG['epochs']
    d = data()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = build(s).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], betas=(0.9, 0.999))
    N, bs, a, Tk = d['tr_y'].shape[0], CFG['bs'], CFG['kd_alpha'], CFG['kd_T']
    if s['kd'] and bank is None:
        raise ValueError('kd run needs bank=D.build_bank()')
    hist, t0 = [], time.time()
    for e in range(E):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(e, E, CFG['lr'], CFG['warmup'])
        model.train()
        perm = torch.randperm(N, generator=g).to(DEV)
        tl = nb = 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            if CFG['aug']:
                sidx = torch.randint(0, len(SHIFTS), (len(idx),), generator=g).to(DEV)
            else:
                sidx = torch.zeros(len(idx), dtype=torch.long, device=DEV)
            x = make_inputs(apply_shifts(d['tr_snn'][idx].float(), sidx), s)
            y = d['tr_y'][idx]
            out = model(x)
            ce = F.cross_entropy(out, y, label_smoothing=CFG['ls'])
            if s['kd']:
                zt = bank[sidx, idx]
                kd = F.kl_div(F.log_softmax(out / Tk, 1), F.softmax(zt / Tk, 1), reduction='batchmean') * Tk * Tk
                loss = a * kd + (1 - a) * ce
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        if (e + 1) % 20 == 0 or e == E - 1:
            acc, _ = evaluate(model, s)
            hist.append([e + 1, round(tl / max(nb, 1), 4), round(acc, 2)])
            if verbose:
                print(f'   {r} ep{e + 1:3d} loss {tl / max(nb, 1):.3f} test {acc:.2f}  ({time.time() - t0:.0f}s)')
    acc, preds = evaluate(model, s)
    res = dict(rid=r, spec=s, seed=seed, final_test_acc=acc, preds=preds, epochs=E, hist=hist,
               train_seconds=round(time.time() - t0, 1), params=nparams(model), cfg=CFG,
               version=VERSION, code_sha=code_sha(), created=datetime.datetime.now().isoformat(timespec='seconds'))
    if persist:
        p = os.path.join(LOCAL, 'ckpt', f'{r}.pth')
        _mk(os.path.dirname(p))
        torch.save(dict(model=model.state_dict(), spec=s, seed=seed, final_test_acc=acc, version=VERSION), p)
        _drive_copy(p, f'ckpt/{r}.pth')
        save_json(f'results/{r}.json', res)
        lp = os.path.join(LOCAL, 'logs', 'runs.csv')
        _mk(os.path.dirname(lp))
        open(lp, 'a').write(f"{res['created']},{r},{acc:.2f},{res['train_seconds']},{VERSION},{res['code_sha']}\n")
        _drive_copy(lp, 'logs/runs.csv')
    return res


def run_all(ids=None, seeds=(0,), bank=None):
    setup()
    ids = ids or list(SPECS)
    if bank is None and any(SPECS[i]['kd'] for i in ids):
        bank = build_bank()
    for sd in seeds:
        for i in ids:
            r = rid(SPECS[i], sd)
            if done(r):
                print(f'skip (done) {r}')
                continue
            res = train_one(SPECS[i], sd, bank)
            print(f'DONE {r}: final test acc {res["final_test_acc"]:.2f}  params {res["params"]}  {res["train_seconds"]:.0f}s')
    sync_to_drive()


# ----------------------------------------------------------------- op-count estimate (paper method)
@torch.no_grad()
def op_count(model, s):
    mods = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    macs, rate_sum, n, cur = {}, {m: 0.0 for m in mods}, [0], [1]
    hs = []

    def fwd_hook(m, inp, out):
        if m in macs:
            return
        k = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] if isinstance(m, nn.Conv2d) else m.in_features
        macs[m] = out.numel() * k / cur[0]                              # per sample, includes the T repeats

    def pre_hook(m, inp):
        rate_sum[m] += inp[0].float().mean().item() * cur[0]

    model.eval()
    d = data()
    for m in mods:
        hs += [m.register_forward_hook(fwd_hook), m.register_forward_pre_hook(pre_hook)]
    X = d['te_snn']
    for i in range(0, len(X), 64):
        cur[0] = min(64, len(X) - i)
        model(make_inputs(X[i:i + 64].float(), s))
        n[0] += cur[0]
    for h in hs:
        h.remove()
    snn = s['arch'] == 'snn4'
    mac = ac = 0.0
    rates = []
    for j, m in enumerate(mods):
        rate = rate_sum[m] / n[0]
        if snn and j >= 1:                                              # input is binary spikes -> AC
            ac += macs[m] * rate
            rates.append(round(rate, 4))
        else:
            mac += macs[m]
    return dict(mac=mac, ac=ac, est_uJ=(mac * E_MAC + ac * E_AC) * 1e-6, rates=rates)


def get_ops(s, seed=0):
    rel = f"ops/{CFG['tag']}_{sid(s)}.json"
    o = read_json(rel)
    if o is None:
        o = op_count(load_model(s, seed), s)
        save_json(rel, o)
    return o


def load_model(s, seed=0):
    p = ckpt_file(rid(s, seed))
    if p is None:
        raise FileNotFoundError(f'no checkpoint for {rid(s, seed)}')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    m = build(s)
    m.load_state_dict(ck['model'])
    return m.to(DEV).eval()


# ----------------------------------------------------------------- latency + NVML energy (paper protocol, interleaved)
def _sync():
    if DEV == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def _lat_run(model, x, reps):
    if DEV == 'cuda':
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            model(x)
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / reps
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    return (time.perf_counter() - t) * 1000 / reps


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None, None


def _idle(nv, h, sec=3.0):
    v, t0 = [], time.time()
    while time.time() - t0 < sec:
        v.append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.05)
    return float(np.mean(v))


@torch.no_grad()
def _energy_run(model, x, seconds, nv, h, idle):
    for _ in range(10):
        model(x)
    _sync()
    p, n, t0 = [], 0, time.time()
    while time.time() - t0 < seconds:
        model(x)
        _sync()
        p.append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
        n += 1
    el = time.time() - t0
    return (float(np.mean(p)) - idle) * el / n / x.shape[0] * 1e3        # mJ per sample


def measure_suite(ids, seed=0, seconds=10, repeats=5, tag='final'):
    """Interleaved: each repeat visits every model once (limits GPU-state drift between models)."""
    d = data()
    models = {i: load_model(SPECS[i], seed) for i in ids}
    x1 = {i: make_inputs(d['te_snn'][:1].float(), SPECS[i]) for i in ids}
    x64 = {i: make_inputs(d['te_snn'][:64].float(), SPECS[i]) for i in ids}
    nv, h = _nvml()
    if nv is None:
        print('NOTE: NVML not available -> latency only (pip install pynvml, GPU runtime needed)')
    with torch.no_grad():
        for i in ids:
            for _ in range(30):
                models[i](x1[i]); models[i](x64[i])
    L1, L64, EN = {i: [] for i in ids}, {i: [] for i in ids}, {i: [] for i in ids}
    for r in range(repeats):
        idle = _idle(nv, h) if nv else None
        for i in ids:
            L1[i].append(_lat_run(models[i], x1[i], 100))
            L64[i].append(_lat_run(models[i], x64[i], 30))
            if nv:
                EN[i].append(_energy_run(models[i], x64[i], seconds, nv, h, idle))
        print(f'  measure repeat {r + 1}/{repeats} done')
    ms = lambda v: [float(np.mean(v)), float(np.std(v))] if len(v) else None
    out = dict(tag=tag, seed=seed, seconds=seconds, repeats=repeats, device=torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu',
               created=datetime.datetime.now().isoformat(timespec='seconds'), version=VERSION, code_sha=code_sha(),
               models={i: dict(lat_bs1_ms=ms(L1[i]), lat_bs64_ms=ms(L64[i]), energy_mJ=ms(EN[i])) for i in ids})
    save_json(f"measure/{CFG['tag']}_{tag}_{datetime.datetime.now().strftime('%m%d_%H%M')}.json", out)
    return out


def load_measures(tag=None):
    js = []
    for base in (LOCAL, EXP):
        dp = os.path.join(base, 'measure')
        if not os.path.isdir(dp):
            continue
        for fn in os.listdir(dp):
            if not fn.startswith(CFG['tag'] + '_') or (tag and f'_{tag}_' not in fn):
                continue
            try:
                js.append(json.load(open(os.path.join(dp, fn))))
            except Exception:
                pass
    M = {}
    for j in sorted(js, key=lambda j: j.get('created', '')):          # newest measurement wins
        M.update(j['models'])
    return M


# ----------------------------------------------------------------- stats + reporting
def mcnemar_p(pa, pb, y):
    from scipy.stats import binomtest, chi2
    a, b = np.array(pa) == y, np.array(pb) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    if n < 25:
        return float(binomtest(min(n01, n10), n, 0.5).pvalue), n01, n10
    return float(chi2.sf((abs(n01 - n10) - 1) ** 2 / n, 1)), n01, n10


def accs(i, seeds):
    out = {}
    for sd in seeds:
        j = read_json(f'results/{rid(SPECS[i], sd)}.json')
        if j:
            out[sd] = j
    return out


def _ms(v):
    v = np.array(v, dtype=float)
    return f'{v.mean():.2f}±{(v.std(ddof=1) if len(v) > 1 else 0):.2f}'


def report(ids=None, seeds=(0, 1, 2)):
    ids = ids or list(SPECS)
    M = load_measures()
    rows = []
    for i in ids:
        R = accs(i, seeds)
        if not R:
            continue
        a = [R[k]['final_test_acc'] for k in R]
        try:
            o = get_ops(SPECS[i])
            est = f"{o['est_uJ']:.1f}"
        except Exception:
            est = '-'
        m = M.get(i, {})
        f = lambda k, p=3: ('-' if not m.get(k) else f'{m[k][0]:.{p}f}')
        rows.append((np.mean(a), f"{i:26s} n={len(a)} acc {_ms(a):>11s} params {list(R.values())[0]['params']:>8d} "
                                  f"est {est:>7s}uJ  lat1 {f('lat_bs1_ms')}ms  lat64 {f('lat_bs64_ms')}ms  E {f('energy_mJ', 4)}mJ"))
    print(f"results tag: {CFG['tag']}   (ratios/estimates from seed 0; acc = mean±std of final epoch over seeds)")
    for _, line in sorted(rows, key=lambda r: -r[0]):
        print(line)
    p = os.path.join(LOCAL, 'summary', f"report_{datetime.datetime.now().strftime('%m%d_%H%M')}.txt")
    _mk(os.path.dirname(p))
    open(p, 'w').write('\n'.join(l for _, l in sorted(rows, key=lambda r: -r[0])))
    _drive_copy(p, os.path.relpath(p, LOCAL))


def shortlist(seed=0, n_snn=5):
    base = [i for i in SPECS if SPECS[i]['arch'] != 'snn4' and done(rid(SPECS[i], seed))]
    snn = sorted([(read_json(f'results/{rid(SPECS[i], seed)}.json')['final_test_acc'], i)
                  for i in SPECS if SPECS[i]['arch'] == 'snn4' and done(rid(SPECS[i], seed))], reverse=True)
    return base + [i for _, i in snn[:n_snn]]


def verdict(ids=None, seeds=(0, 1, 2)):
    """SNN vs 4 controls. Ratios are SNN/baseline (<1 = SNN better). CLEAN = acc>0 and lat1,lat64,energy all <1."""
    M = load_measures()
    y = data()['te_y'].cpu().numpy()
    ids = [i for i in (ids or list(SPECS)) if SPECS[i]['arch'] == 'snn4']
    clean, best = [], []
    for i in ids:
        A = accs(i, seeds)
        if not A:
            continue
        print(f'\n{i}  acc {_ms([A[k]["final_test_acc"] for k in A])} (n={len(A)})')
        for b in baselines_for(SPECS[i]):
            Bb = accs(b, seeds)
            common = [k for k in A if k in Bb]
            if not common:
                continue
            da = np.mean([A[k]['final_test_acc'] for k in common]) - np.mean([Bb[k]['final_test_acc'] for k in common])
            ps = [mcnemar_p(A[k]['preds'], Bb[k]['preds'], y)[0] for k in common]
            better = [k for k, p in zip(common, ps) if p < 0.05 and A[k]['final_test_acc'] > Bb[k]['final_test_acc']]
            ms_, mb = M.get(i, {}), M.get(b, {})
            ratio = lambda key: (ms_[key][0] / mb[key][0]) if ms_.get(key) and mb.get(key) else None
            r1, r64, re = ratio('lat_bs1_ms'), ratio('lat_bs64_ms'), ratio('energy_mJ')
            fa = da > 0
            fl = r1 is not None and r64 is not None and r1 < 1 and r64 < 1
            fe = re is not None and re < 1
            fmt = lambda v: '  -  ' if v is None else f'{v:5.2f}'
            print(f"  vs {b:24s} dAcc {da:+6.2f}  McNemar p {['%.3f' % p for p in ps]} sig+ {len(better)}/{len(common)}  "
                  f"lat1 {fmt(r1)} lat64 {fmt(r64)} energy {fmt(re)}  -> ACC{'+' if fa else '-'} LAT{'+' if fl else '-'} EN{'+' if fe else '-'}")
            best.append((int(fa) + int(fl) + int(fe), i, b))
            if fa and fl and fe:
                clean.append((i, b))
    print('\n==== CLEAN WINS (acc + latency + energy, per baseline) ====')
    if clean:
        for i, b in clean:
            print(f'  {i}  beats  {b}')
        print('A clean win vs the KD baselines (cnn3_..._kd and cnn4_..._kd) means the gain is not just distillation.')
    else:
        print('  none under this protocol. Closest (axes won of 3):')
        for sc, i, b in sorted(best, reverse=True)[:5]:
            print(f'  {sc}/3  {i} vs {b}')


# ----------------------------------------------------------------- self-test (run first)
def selftest():
    setup()
    d = data()
    print('data', {k: tuple(v.shape) for k, v in d.items()}, 'device', DEV)
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 1, 1, 4, 4)
    y = shift_xy(x, 1, 0)
    assert torch.equal(y[..., 1:], x[..., :-1]) and y[..., 0].abs().sum() == 0
    y = shift_xy(x, 0, -1)
    assert torch.equal(y[..., :-1, :], x[..., 1:, :]) and y[..., -1, :].abs().sum() == 0
    old_tag = CFG['tag']
    CFG['tag'] = 'selftest'
    try:
        fake_bank = torch.randn(len(SHIFTS), d['tr_y'].shape[0], 11, device=DEV)
        for i in ('snn4_T1K4_w0.5_kd', 'snn4_T2K2_w1_kd', 'cnn3_T1K4_w1_plain', 'cnn4_T1K8_w0.5_kd'):
            s = SPECS[i]
            m = build(s).to(DEV)
            out = m(make_inputs(d['tr_snn'][:8].float(), s))
            assert out.shape == (8, 11), out.shape
            out.sum().backward()
            r = train_one(s, 0, fake_bank, epochs=1, persist=False, verbose=False)
            print(f'  ok {i:24s} params {nparams(m):>8d}  1-epoch acc {r["final_test_acc"]:.1f}')
        s = SPECS['snn4_T1K4_w0.5_kd']
        o = op_count(build(s).to(DEV), s)
        print('  op_count', {k: (round(v, 3) if not isinstance(v, list) else v) for k, v in o.items()})
        o = op_count(build(SPECS['cnn3_T1K4_w1_plain']).to(DEV), SPECS['cnn3_T1K4_w1_plain'])
        print('  cnn3 MACs (paper: 27.13M, 124.81 uJ):', round(o['mac'] / 1e6, 2), 'M', round(o['est_uJ'], 2), 'uJ')
        print('  lat test', round(_lat_run(build(s).to(DEV).eval(), make_inputs(d['te_snn'][:1].float(), s), 5), 3), 'ms')
    finally:
        CFG['tag'] = old_tag
    print('checking teachers (must reproduce 93.75 / 94.79 / 94.10):')
    for n, a in TEACHERS:
        load_teacher(n, a)
    print('\nSELFTEST PASSED')


if __name__ == '__main__':
    print(__doc__)
