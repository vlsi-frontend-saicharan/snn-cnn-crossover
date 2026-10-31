"""
nmnist_resnet.py  --  SECOND ARCHITECTURE FAMILY on N-MNIST (add-on to snn_win.py)           [nmnist_resnet_v1]

Same question as dvs128_resnet.py: where is the crossover timestep T* of a spiking RESIDUAL network against its ReLU twin?
Reuses snn_win.py unchanged (data cache, count input log1p, recipe: AdamW lr 2e-3, wd 1e-4, cosine, batch 128, 20 epochs,
last-epoch accuracy, LIF beta 0.9 + fast-sigmoid surrogate, BN folding, op-count energy incl. membrane term, McNemar).
No distillation (plain cross-entropy for SNN and CNN).  3 seeds.

Models (about 0.16 M parameters, 4 residual blocks 16-32-64-64, each followed by max-pool 2: 34 -> 2):
  RSNN-R_T<T>_nokd  spiking ResNet, T in {1,2,3,5}, one time bin per step, shortcut added to the membrane input of the 2nd LIF
  RCNN-R_T1_nokd    ReLU ResNet on the time-summed frame (the N-MNIST CNN of the paper)
Results go to  <ROOT>/win_runs/resnet_family_nmnist/  (results.json, preds/, ckpt/, tables/, latency+energy json, README).
"""
import os, sys, json, time, copy, shutil, datetime, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get('NM_ROOT', '/content/drive/MyDrive/nmnist_snn_cnn')
sys.path.insert(0, ROOT)
import snn_win as W

W.ROOT = ROOT
TAG = 'resnet_family_nmnist'
NOTE = 'second architecture family: spiking ResNet vs ReLU ResNet, no KD, count input, same recipe as snn_win'
VERSION = 'nmnist_resnet_v1'
CH = (16, 32, 64, 64)
TS = (1, 2, 3, 5)


# ------------------------------------------------------------------ model
class RBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        if cin != cout:
            self.sc = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
            self.bs = nn.BatchNorm2d(cout)
        else:
            self.sc = None
            self.bs = None


class ResNet2(nn.Module):
    """spiking=True: input (B,T,2,34,34) -> LIF over T, leaky-membrane readout (closed form, as in snn_win.Net).
    spiking=False: input (B,T,2,34,34) -> time mean -> ReLU ResNet."""
    def __init__(self, T, spiking):
        super().__init__()
        self.T, self.spiking = T, spiking
        self.stem = nn.Conv2d(2, CH[0], 3, 1, 1, bias=False)          # first Conv2d in modules()
        self.bn0 = nn.BatchNorm2d(CH[0])
        cins = [CH[0], CH[0], CH[1], CH[2]]
        self.blocks = nn.ModuleList([RBlock(cins[k], CH[k]) for k in range(4)])
        self.fc = nn.Linear(CH[3] * 2 * 2, 10)                          # 34 -> 17 -> 8 -> 4 -> 2
        if spiking:
            k = torch.arange(T, dtype=torch.float32)
            self.register_buffer('ro_w', (1 - W.BETA ** (T - k)) / (1 - W.BETA) / T)

    def _act(self, c):                                                  # c: (B*T,C,H,W) spiking, (B,C,H,W) ReLU
        if not self.spiking:
            return F.relu(c)
        T = self.T
        B = c.shape[0] // T
        c = c.reshape(B, T, *c.shape[1:])
        if T == 1:
            return W.spike_fn(c[:, 0] - 1.0)
        mem = torch.zeros_like(c[:, 0])
        out = []
        for t in range(T):
            mem = W.BETA * mem + c[:, t]
            s = W.spike_fn(mem - 1.0)
            mem = mem - s.detach()
            out.append(s)
        return torch.stack(out, 1).flatten(0, 1)

    def forward(self, x):
        B = x.shape[0]
        h = x.flatten(0, 1) if self.spiking else x.mean(1)
        h = self._act(self.bn0(self.stem(h)))
        for blk in self.blocks:
            a = self._act(blk.b1(blk.c1(h)))
            z = blk.b2(blk.c2(a))
            s = h if blk.sc is None else blk.bs(blk.sc(h))
            h = F.max_pool2d(self._act(z + s), 2)
        if not self.spiking:
            return self.fc(h.flatten(1))
        cur = self.fc(h.flatten(1)).reshape(B, self.T, 10)
        return (cur * self.ro_w.view(1, self.T, 1)).sum(1)

    def fuse(self):
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        m = copy.deepcopy(self).eval()
        m.stem = fuse_conv_bn_eval(m.stem, m.bn0)
        m.bn0 = nn.Identity()
        for blk in m.blocks:
            blk.c1 = fuse_conv_bn_eval(blk.c1, blk.b1); blk.b1 = nn.Identity()
            blk.c2 = fuse_conv_bn_eval(blk.c2, blk.b2); blk.b2 = nn.Identity()
            if blk.sc is not None:
                blk.sc = fuse_conv_bn_eval(blk.sc, blk.bs); blk.bs = nn.Identity()
        return m.eval()


# ------------------------------------------------------------------ plug into snn_win (done once)
if not getattr(W, '_resnet_patched', False):
    W._orig_build, W._orig_op_stats, W._orig_energy = W.build, W.op_stats, W.energy_from_stats

    def _build(spec):
        kind, w, T, kd = spec
        if kind in ('RSNN', 'RCNN'):
            return ResNet2(T, kind == 'RSNN').to(W.dev)
        return W._orig_build(spec)

    @torch.no_grad()
    def _op_stats(model, X, n=1000):
        st = W._orig_op_stats(model, X, n)
        if isinstance(model, ResNet2):
            st['__meta__'] = {'spiking': float(model.spiking)}
        return st

    def _energy(st, spiking, e_mac=W.E_MAC, e_ac=W.E_AC):
        meta = st.pop('__meta__', None)
        if meta is None:
            return W._orig_energy(st, spiking, e_mac, e_ac)
        layers = {k: v for k, v in st.items()}
        if meta['spiking'] > 0:
            first = layers['stem']
            rest = sum(v['nnz_fan'] for k, v in layers.items() if k != 'stem') * e_ac
            spk = sum(v['neurons'] for k, v in layers.items() if k == 'stem' or k.endswith('.c1') or k.endswith('.c2'))
            memb = spk * e_mac                                      # one MAC-cost membrane update per spiking neuron and step
            return {'E_main_uJ': (first['dense'] * e_mac + rest + memb) * 1e-6,
                    'E_strict_uJ': (first['dense'] * e_mac + rest + memb) * 1e-6,
                    'E_evt_uJ': (first['nnz_fan'] * e_mac + rest + memb) * 1e-6}
        dense = sum(v['dense'] for v in layers.values()) * e_mac * 1e-6
        sparse = sum(v['nnz_fan'] for v in layers.values()) * e_mac * 1e-6
        return {'E_main_uJ': dense, 'E_strict_uJ': sparse, 'E_evt_uJ': sparse}

    W.build, W.op_stats, W.energy_from_stats = _build, _op_stats, _energy
    W._resnet_patched = True

build = W.build

SPECS = {'RCNN': ('RCNN', 'R', 1, False)}
for _T in TS:
    SPECS[f'RSNN{_T}'] = ('RSNN', 'R', _T, False)
PLAN = [SPECS['RCNN']] + [SPECS[f'RSNN{t}'] for t in TS]


def setup():
    W.init_run(TAG, NOTE)
    p = W._p('README_resnet.md')
    open(p, 'w').write(f"""# N-MNIST second architecture family (residual network)   [{VERSION}]
Spiking ResNet (RSNN-R_T<T>_nokd) vs ReLU ResNet on the time-summed frame (RCNN-R_T1_nokd); ~0.16 M parameters; no distillation.
Recipe = snn_win.py: AdamW lr {W.LR}, wd {W.WD}, cosine, batch {W.BATCH}, {W.EPOCHS} epochs, last-epoch accuracy, 3 seeds, count input log1p, LIF beta {W.BETA}.
results.json  one entry per (model, seed): acc, params, E_main_uJ (SNN: first layer dense MACs + ACs x spikes + membrane updates; CNN: dense MACs),
              E_strict_uJ (CNN with zero-skipping = generous baseline), E_evt_uJ.
preds/<model>_s<seed>.npy  test predictions;  ckpt/*_final.pt  weights;  resnet_latency_energy.json  batch-1 latency + NVML energy;
tables/resnet_family_models.csv, resnet_family_snn_vs_cnn.csv   final tables with T* for estimate / GPU latency / GPU energy.
Code: code_snapshot_* (snn_win.py) and nmnist_resnet.py in this folder.
""")
    try:
        shutil.copyfile(os.path.abspath(__file__), W._p('nmnist_resnet_snapshot.py'))
    except Exception:
        pass


def train(seeds=(0,), plan=None):
    """Resumable (finished runs skipped, unfinished ones continue from their last epoch)."""
    setup()
    W.run_plan(seeds=list(seeds), plan=plan or PLAN)


def status():
    res = W._load(W._p('results.json'))
    for s in (0, 1, 2):
        d = [r for r in res.values() if r['seed'] == s and r['kind'] in ('RSNN', 'RCNN')]
        print(f'seed {s}: {len(d)}/{len(PLAN)} runs done')


# ------------------------------------------------------------------ smoke test (1 epoch, builds the data caches too)
def smoke():
    old, oldtag = W.EPOCHS, W.RUN_TAG
    W.EPOCHS = 1
    try:
        W.init_run('smoke_resnet', note='smoke test, safe to delete')
        for spec in (SPECS['RCNN'], SPECS['RSNN2']):
            m = build(spec)
            x = W._enc(W.get_data(spec[2])['test'][0][:4])
            print(spec, 'input', tuple(x.shape), 'output', tuple(m(x).shape), 'params', sum(p.numel() for p in m.parameters()))
            fm = m.eval().fuse()
            print('   fold diff', float((m(x) - fm(x)).abs().max()))
            r = W.train_one(spec, 99)
            print('   1-epoch acc', r['acc'], ' E_main', r['E_main_uJ'], ' E_strict', r['E_strict_uJ'])
        print('SMOKE OK')
    finally:
        W.EPOCHS = old
        W.RUN_TAG = oldtag


# ------------------------------------------------------------------ latency + NVML energy (BN folded, round-robin)
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
def _energy_mJ(m, x64, seconds, nv, h, idle):
    for _ in range(10):
        m(x64)
    torch.cuda.synchronize()
    p, n, t0 = [], 0, time.time()
    while time.time() - t0 < seconds:
        m(x64)
        torch.cuda.synchronize()
        p.append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
        n += 1
    return (float(np.mean(p)) - idle) * (time.time() - t0) / n / x64.shape[0] * 1e3


@torch.no_grad()
def measure(rounds=5, iters=300, warm=100, e_seconds=10, e_repeats=3):
    res = W._load(W._p('results.json'))
    best = {}
    for k, r in res.items():
        if r['kind'] in ('RSNN', 'RCNN') and (r['spec'] not in best or r['seed'] < res[best[r['spec']]]['seed']):
            best[r['spec']] = k
    items = {}
    for n, k in best.items():
        r = res[k]
        m = build((r['kind'], r['width'], r['T'], r['kd']))
        m.load_state_dict(torch.load(W._p('ckpt', f'{k}_final.pt'), map_location=W.dev, weights_only=False))
        m = m.fuse().eval()
        Xte = W.get_data(r['T'])['test'][0]
        x1, x64 = W._enc(Xte[:1]), W._enc(Xte[:64])
        g = None
        try:
            g = W._make_graph(m, x1)
        except Exception as e:
            print(f'[{n}] CUDA graph failed ({str(e)[:80]}), eager only')
        items[n] = (m, x1, x64, g)
        print(f'  loaded {n}  (seed {r["seed"]})')
    E = {n: [] for n in items}
    G = {n: [] for n in items}
    for _ in range(rounds):
        for n, (m, x1, x64, g) in items.items():
            E[n].append(W._median_ms(lambda: m(x1), iters, warm))
            if g is not None:
                G[n].append(W._median_ms(g[0].replay, iters, warm))
    nv, h = _nvml()
    EN = {n: [] for n in items}
    if nv is None:
        print('NOTE: pynvml unavailable -> no measured energy (pip install pynvml)')
    else:
        names = list(items)
        for rep in range(e_repeats):
            k = rep % len(names)
            for n in names[k:] + names[:k]:
                time.sleep(8)
                idle = _idle(nv, h)
                EN[n].append(_energy_mJ(items[n][0], items[n][2], e_seconds, nv, h, idle))
            print(f'  energy repeat {rep + 1}/{e_repeats}')
    out = {n: dict(eager_ms=round(float(np.median(E[n])), 4), graph_ms=round(float(np.median(G[n])), 4) if G[n] else None,
                   energy_mJ=[float(np.mean(EN[n])), float(np.std(EN[n]))] if EN[n] else None,
                   gpu=torch.cuda.get_device_name(0), version=VERSION,
                   created=datetime.datetime.now().isoformat(timespec='seconds')) for n in items}
    W._save(out, W._p('resnet_latency_energy.json'))
    for n, v in out.items():
        print(f'{n:22s} eager {v["eager_ms"]:.3f} ms  graph {v["graph_ms"]}  energy {v["energy_mJ"]}')
    return out


# ------------------------------------------------------------------ tables with T* and McNemar
def _mcnemar(pa, pb, y):
    """Paper rule: exact binomial if fewer than 25 discordant items, else chi-square with continuity correction."""
    from scipy.stats import binomtest, chi2
    ca, cb = pa == y, pb == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    if n < 25:
        return float(binomtest(min(n01, n10), n, 0.5).pvalue)
    return float(chi2.sf((abs(n01 - n10) - 1) ** 2 / n, 1))


def _rat(a, b):
    return None if (a is None or b is None or b == 0) else a / b


def table():
    import csv
    res = W._load(W._p('results.json'))
    lat = W._load(W._p('resnet_latency_energy.json'))
    labels = np.load(W._p('preds', 'labels_test.npy'))
    by = {}
    for r in res.values():
        if r['kind'] in ('RSNN', 'RCNN'):
            by.setdefault(r['spec'], []).append(r)
    rows = {}
    for spec, rs in by.items():
        a = np.array([r['acc'] for r in rs])
        l = lat.get(spec, {})
        rows[spec] = dict(spec=spec, kind=rs[0]['kind'], T=rs[0]['T'], n=len(rs), acc=float(a.mean()),
                          sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0, params=rs[0]['params'],
                          E_main_uJ=float(np.mean([r['E_main_uJ'] for r in rs])),
                          E_strict_uJ=float(np.mean([r['E_strict_uJ'] for r in rs])),
                          lat_graph_ms=l.get('graph_ms'), energy_mJ=(l.get('energy_mJ') or [None])[0])
    cnn = [r for r in rows.values() if r['kind'] == 'RCNN']
    if not cnn:
        print('no finished RCNN runs yet')
        return
    c = cnn[0]
    f = lambda v, p=3: '  -  ' if v is None else f'{v:.{p}f}'
    print('N-MNIST ResNet family (acc = last epoch, mean+-std over n seeds; E in uJ; lat = batch-1 CUDA-graph ms; E mJ = NVML)')
    print(f"{'model':18s}{'n':>2s}{'acc %':>13s}{'params':>8s}{'E_main':>9s}{'E_strict':>9s}{'lat ms':>8s}{'E mJ':>8s}")
    for r in sorted(rows.values(), key=lambda r: (r['kind'] != 'RCNN', r['T'])):
        print(f"{r['spec']:18s}{r['n']:2d}{r['acc']:7.2f}+-{r['sd']:4.2f}{r['params']:8d}{r['E_main_uJ']:9.2f}{r['E_strict_uJ']:9.2f}"
              f"{f(r['lat_graph_ms']):>8s}{f(r['energy_mJ']):>8s}")
    comp = []
    for r in [x for x in rows.values() if x['kind'] == 'RSNN']:
        A = {x['seed']: x for x in by[r['spec']]}
        C = {x['seed']: x for x in by[c['spec']]}
        ks = sorted(set(A) & set(C))
        ps = [_mcnemar(np.load(W._p('preds', f"{A[k]['key']}.npy")), np.load(W._p('preds', f"{C[k]['key']}.npy")), labels) for k in ks]
        up = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['acc'] > C[k]['acc'])
        dn = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['acc'] < C[k]['acc'])
        T = r['T']
        er_dense, er_strict = r['E_main_uJ'] / c['E_main_uJ'], r['E_main_uJ'] / c['E_strict_uJ']
        lr_, mr = _rat(r['lat_graph_ms'], c['lat_graph_ms']), _rat(r['energy_mJ'], c['energy_mJ'])
        comp.append(dict(snn=r['spec'], cnn=c['spec'], dAcc=round(r['acc'] - c['acc'], 2), snn_better=up, cnn_better=dn,
                         p_per_seed=[round(p, 4) for p in ps], est_ratio_dense_cnn=round(er_dense, 3), Tstar_est_dense=round(T / er_dense, 2),
                         est_ratio_zero_skip_cnn=round(er_strict, 3), Tstar_est_zero_skip=round(T / er_strict, 2),
                         lat_ratio=None if lr_ is None else round(lr_, 3), Tstar_lat=None if lr_ is None else round(T / lr_, 2),
                         energy_ratio=None if mr is None else round(mr, 3), Tstar_energy=None if mr is None else round(T / mr, 2)))
    print('\nSNN vs CNN (ratios SNN/CNN, <1 = SNN cheaper; T* = T / ratio; McNemar per seed, k/3 seeds with p<0.05)')
    for x in comp:
        print(f"{x['snn']:18s} dAcc {x['dAcc']:+6.2f}  SNN-better {x['snn_better']}/3 CNN-better {x['cnn_better']}/3  "
              f"est(dense CNN) {x['est_ratio_dense_cnn']:5.2f} (T* {x['Tstar_est_dense']:5.2f})  est(zero-skip CNN) {x['est_ratio_zero_skip_cnn']:5.2f} "
              f"(T* {x['Tstar_est_zero_skip']:5.2f})  lat {f(x['lat_ratio'], 2)} (T* {f(x['Tstar_lat'], 2)})  E {f(x['energy_ratio'], 2)} (T* {f(x['Tstar_energy'], 2)})")
    for nm, data in (('resnet_family_models.csv', list(rows.values())), ('resnet_family_snn_vs_cnn.csv', comp)):
        if data:
            p = W._p('tables', nm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(data[0]))
                wr.writeheader()
                wr.writerows(data)
    return rows, comp


# ------------------------------------------------------------------ package with recognisable names
def package():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    path = f'/content/nmnist_resnet_family_{stamp}.zip'
    base = W.run_dir()
    n = 0
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(base):
            for fn in fs:
                if fn.endswith('_ckpt.pt'):
                    continue                                  # resume files (with optimizer state) are not needed
                full = os.path.join(dp, fn)
                z.write(full, os.path.join('nmnist_resnet_family', os.path.relpath(full, base)))
                n += 1
    print(f'{os.path.basename(path)}: {n} files, {os.path.getsize(path) / 1e6:.1f} MB')
    return path
