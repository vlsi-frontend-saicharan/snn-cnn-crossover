"""
dvs128_resnet.py  --  SECOND ARCHITECTURE FAMILY for the crossover-timestep study (DVS128 Gesture)   [resnet_v1]

Why: the main study uses ONE family (plain stack conv3x3-BN-act-pool).  This file adds a small RESIDUAL network and asks the same
question: where is the crossover timestep T* for a ResNet-style SNN against its matched ReLU ResNet?

Models (width 1.0, about 0.16 M parameters, 4 pooling stages: 48 -> 3):
  stem conv3x3(2K ch -> 16)-BN-act, then 4 residual blocks with channels 16, 32, 64, 64, each followed by max-pool 2.
  block: act( BN(conv3x3( act(BN(conv3x3(x))) )) + shortcut(x) ),  shortcut = identity or conv1x1-BN when channels change.
  rsnn = spiking ResNet: act = LIF (beta 0.5, threshold 1, soft reset, arctan surrogate), the shortcut is added to the MEMBRANE
         input of the second LIF (Spiking-ResNet style), so every conv input after the stem is binary -> counted as AC.
  rcnn = ReLU ResNet with B stacked bins as input channels.
Grid (same recipe as the main study: 100 epochs, AdamW lr 1e-3, wd 1e-2, label smoothing 0.1, +-3 px shifts, final-epoch accuracy):
  rsnn T in {2,4,8,16}, one bin per step;   rcnn B in {1,2,4,8,16};   3 seeds.
Everything reuses dvs128_lowT_kd.py and dvs128_proto.py (same data, training loop, op-count estimate, BN-folded CUDA-graph latency,
NVML energy, McNemar test).  File names carry the tag 'resnet', so nothing mixes with the main grid ('proto') or the KD runs ('v1').

NOT counted (same simplifications as the main paper): identity-shortcut additions and membrane updates; memory traffic.
Outputs go to <EXP>/{results,ckpt,measure,ops,summary,logs}  ->  results/resnet_<arch>_T<T>K<K>_w1_plain_s<seed>.json etc.
"""
import os, sys, json, time, copy, datetime, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/content')
import dvs128_lowT_kd as D
import dvs128_proto as P

D.CFG['tag'] = 'resnet'
VERSION = 'resnet_v1'
WIDTH = 1.0
CH = (16, 32, 64, 64)
TS = (2, 4, 8, 16)
BS = (1, 2, 4, 8, 16)


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


class ResNetNet(nn.Module):
    """spiking=True: input (B,T,C,H,W), LIF activations, logits averaged over T.  spiking=False: input (B,C,H,W), ReLU."""
    def __init__(self, in_ch, w=1.0, spiking=False, ncls=11, drop=0.3):
        super().__init__()
        self.spiking = spiking
        c = [D.ch(x, w) for x in CH]
        self.stem = nn.Conv2d(in_ch, c[0], 3, 1, 1, bias=False)      # registered first -> first Conv2d in modules()
        self.bn0 = nn.BatchNorm2d(c[0])
        cins = [c[0], c[0], c[1], c[2]]
        self.blocks = nn.ModuleList([RBlock(cins[k], c[k]) for k in range(4)])
        self.lif = D.LIFSeq()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(c[3] * 9, ncls)

    def _cv(self, m, x):
        if not self.spiking:
            return m(x)
        B, T = x.shape[:2]
        y = m(x.flatten(0, 1))
        return y.reshape(B, T, *y.shape[1:])

    def _act(self, x):
        return self.lif(x) if self.spiking else F.relu(x)

    def _pool(self, x):
        if not self.spiking:
            return F.max_pool2d(x, 2)
        B, T = x.shape[:2]
        y = F.max_pool2d(x.flatten(0, 1), 2)
        return y.reshape(B, T, *y.shape[1:])

    def _block(self, blk, x):
        h = self._act(self._cv(blk.b1, self._cv(blk.c1, x)))
        h = self._cv(blk.b2, self._cv(blk.c2, h))
        s = x if blk.sc is None else self._cv(blk.bs, self._cv(blk.sc, x))
        return self._act(h + s)

    def forward(self, x):
        x = self._act(self._cv(self.bn0, self._cv(self.stem, x)))
        for blk in self.blocks:
            x = self._pool(self._block(blk, x))
        if self.spiking:
            return self.fc(self.drop(x.flatten(2))).mean(1)
        return self.fc(self.drop(x.flatten(1)))


# ------------------------------------------------------------------ plug into the existing code (done once)
if not getattr(D, '_resnet_patched', False):
    D._orig_build, D._orig_make_inputs, P._orig_fold = D.build, D.make_inputs, P.fold

    def _build(s):
        if s['arch'] == 'rsnn':
            return ResNetNet(2 * s['K'], s['w'], True, drop=D.CFG['drop'])
        if s['arch'] == 'rcnn':
            return ResNetNet(2 * s['T'] * s['K'], s['w'], False, drop=D.CFG['drop'])
        return D._orig_build(s)

    def _make_inputs(raw, s):
        if s['arch'] == 'rsnn':                                   # same layout as snn4: (N,T,K*2,H,W)
            return D._orig_make_inputs(raw, dict(s, arch='snn4'))
        return D._orig_make_inputs(raw, s)                        # rcnn: arch != snn4 -> stacked channels (N,2B,H,W)

    def _fold(model):
        if not isinstance(model, ResNetNet):
            return P._orig_fold(model)
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        m = copy.deepcopy(model).eval()
        m.stem = fuse_conv_bn_eval(m.stem, m.bn0)
        m.bn0 = nn.Identity()
        for blk in m.blocks:
            blk.c1 = fuse_conv_bn_eval(blk.c1, blk.b1); blk.b1 = nn.Identity()
            blk.c2 = fuse_conv_bn_eval(blk.c2, blk.b2); blk.b2 = nn.Identity()
            if blk.sc is not None:
                blk.sc = fuse_conv_bn_eval(blk.sc, blk.bs); blk.bs = nn.Identity()
        return m.eval()

    D.build, D.make_inputs, P.fold = _build, _make_inputs, _fold
    D._resnet_patched = True


def specs():
    out = {}
    for B in BS:
        s = D.S('rcnn', 1, B, WIDTH, False)
        out[D.sid(s)] = s
    for T in TS:
        s = D.S('rsnn', T, 1, WIDTH, False)
        out[D.sid(s)] = s
    return out


SP = specs()
D.SPECS.update(SP)
P.SP.update(SP)
ALL = list(SP)


def snn_id(T):
    return D.sid(D.S('rsnn', T, 1, WIDTH, False))


def cnn_id(B):
    return D.sid(D.S('rcnn', 1, B, WIDTH, False))


# ------------------------------------------------------------------ setup / training (resumable)
README = f"""# DVS128 Gesture - second architecture family (residual network)   [{VERSION}]
What: spiking ResNet (rsnn) versus ReLU ResNet (rcnn), width 1.0 (~0.16 M parameters), same recipe as the main study.
Naming: results/resnet_<arch>_T<T>K<K>_w1_plain_s<seed>.json   arch rsnn = spiking, rcnn = ReLU;
        rsnn: T = timesteps (K=1); rcnn: K = number of stacked bins B (T=1).   One JSON per run (final-epoch test accuracy + predictions).
ckpt/resnet_*.pth  final weights.   ops/resnet_stats_*.json  op-count estimate and spike rates.
measure/proto_pm_<date>_resnet.json  batch-1 latency (eager, CUDA graph, BN folded) and NVML energy.
summary/resnet_family_models.csv  and  resnet_family_snn_vs_cnn.csv  the final tables (T* for estimate / GPU latency / GPU energy).
Protocol: {json.dumps(D.CFG)}   Reported accuracy = FINAL epoch, 3 seeds, no distillation.
Code: code/dvs128_resnet.py (this experiment), dvs128_lowT_kd.py, dvs128_proto.py (shared training / measurement code).
"""


def setup():
    for d in ('results', 'ckpt', 'measure', 'ops', 'summary', 'logs', 'code'):
        D._mk(os.path.join(D.EXP, d))
        D._mk(os.path.join(D.LOCAL, d))
    p = os.path.join(D.LOCAL, 'README_resnet.md')
    open(p, 'w').write(README)
    D._drive_copy(p, 'README_resnet.md')
    for f in (__file__,):
        try:
            D._drive_copy(f, 'code/dvs128_resnet.py')
        except Exception:
            pass


def train(seeds=(0,), ids=None):
    """Resumable: finished runs are skipped. Seed-major order, so seed 0 of every model finishes first."""
    setup()
    for sd in seeds:
        for i in (ids or ALL):
            r = D.rid(SP[i], sd)
            if D.done(r):
                print(f'skip (done) {r}')
                continue
            res = D.train_one(SP[i], sd, None)
            print(f'DONE {r}: final test acc {res["final_test_acc"]:.2f}  params {res["params"]}  {res["train_seconds"]:.0f}s')
    D.sync_to_drive()


def status():
    n = 0
    for sd in (0, 1, 2):
        d = [i for i in ALL if D.done(D.rid(SP[i], sd))]
        n += len(d)
        print(f'seed {sd}: {len(d)}/{len(ALL)} runs done')
    print('total', n, '/', 3 * len(ALL))


# ------------------------------------------------------------------ estimates + measurement
def stats(i, seed=0):
    rel = f'ops/resnet_stats_{i}.json'
    o = D.read_json(rel)
    if o is None:
        s = SP[i]
        s2 = dict(s, arch='snn4' if s['arch'] == 'rsnn' else 'cnn4')     # reuse the main study's estimator unchanged
        o = P._stats(D.load_model(s, seed), s2)
        D.save_json(rel, o)
    return o


def measure_all(rounds=15, e_repeats=3, e_seconds=10):
    """BN-folded, interleaved batch-1 latency (eager + CUDA graph) and NVML energy; seed-0 checkpoints."""
    return P.measure(ids=ALL, seed=0, rounds=rounds, energy=True, e_seconds=e_seconds, e_repeats=e_repeats, tag='resnet')


def _rat(a, b):
    return None if (a is None or b is None or b == 0) else a / b


def table(seeds=(0, 1, 2)):
    M = P.load_pm()
    y = D.data()['te_y'].cpu().numpy()
    rows = {}
    for i in ALL:
        A = D.accs(i, seeds)
        if not A:
            continue
        s, st, m = SP[i], stats(i), M.get(i, {})
        a = [A[k]['final_test_acc'] for k in sorted(A)]
        rows[i] = dict(id=i, kind='SNN' if s['arch'] == 'rsnn' else 'CNN', T_or_B=s['T'] if s['arch'] == 'rsnn' else s['K'],
                       n=len(a), acc=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                       params=A[sorted(A)[0]]['params'], est_uJ=st['est_uJ'], lat_graph_ms=m.get('lat_graph_ms'),
                       energy_mJ=(m.get('energy_mJ') or [None])[0], rates=st.get('rates'))
    cnn = [r for r in rows.values() if r['kind'] == 'CNN']
    if not cnn:
        print('no finished CNN runs yet')
        return
    best = max(cnn, key=lambda r: r['acc'])
    f = lambda v, p=3: '  -  ' if v is None else f'{v:.{p}f}'
    print('ResNet family, DVS128 (acc = final epoch, mean+-std over n seeds; est = op-count uJ; lat = batch-1 CUDA-graph ms; E = NVML mJ/sample)')
    print(f"{'model':6s}{'T/B':>4s}{'n':>2s}{'acc %':>13s}{'params':>8s}{'est uJ':>8s}{'lat ms':>8s}{'E mJ':>8s}")
    for r in sorted(rows.values(), key=lambda r: (r['kind'] != 'CNN', r['T_or_B'])):
        tb = ('T' if r['kind'] == 'SNN' else 'B') + str(r['T_or_B'])
        print(f"{r['kind']:6s}{tb:>4s}{r['n']:2d}{r['acc']:7.2f}+-{r['sd']:4.2f}{r['params']:8d}{r['est_uJ']:8.1f}{f(r['lat_graph_ms']):>8s}{f(r['energy_mJ']):>8s}")
    print(f"best CNN (chosen on test accuracy, as in the paper): {best['id']}  {best['acc']:.2f}")
    comp = []
    for r in [x for x in rows.values() if x['kind'] == 'SNN']:
        for tag, c in (('best_CNN', best), ('matched_B=T', rows.get(cnn_id(r['T_or_B'])))):
            if c is None:
                continue
            A, C = D.accs(r['id'], seeds), D.accs(c['id'], seeds)
            ks = [k for k in A if k in C]
            ps = [D.mcnemar_p(A[k]['preds'], C[k]['preds'], y)[0] for k in ks]
            up = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['final_test_acc'] > C[k]['final_test_acc'])
            dn = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['final_test_acc'] < C[k]['final_test_acc'])
            T = r['T_or_B']
            er, lr_, mr = r['est_uJ'] / c['est_uJ'], _rat(r['lat_graph_ms'], c['lat_graph_ms']), _rat(r['energy_mJ'], c['energy_mJ'])
            comp.append(dict(snn=r['id'], vs=tag, cnn=c['id'], dAcc=round(r['acc'] - c['acc'], 2), snn_better=up, cnn_better=dn,
                             p_per_seed=[round(p, 4) for p in ps], est_ratio=round(er, 3), Tstar_est=round(T / er, 2),
                             lat_ratio=None if lr_ is None else round(lr_, 3), Tstar_lat=None if lr_ is None else round(T / lr_, 2),
                             energy_ratio=None if mr is None else round(mr, 3), Tstar_energy=None if mr is None else round(T / mr, 2)))
    print('\nSNN vs CNN (ratios SNN/CNN, <1 = SNN cheaper; T* = T / ratio; McNemar per seed, k/3 seeds with p<0.05)')
    for c in comp:
        print(f"{c['snn']:22s} vs {c['vs']:11s} dAcc {c['dAcc']:+6.2f}  SNN-better {c['snn_better']}/3 CNN-better {c['cnn_better']}/3  "
              f"est {c['est_ratio']:6.2f} (T* {c['Tstar_est']:5.2f})  lat {f(c['lat_ratio'], 2)} (T* {f(c['Tstar_lat'], 2)})  "
              f"E {f(c['energy_ratio'], 2)} (T* {f(c['Tstar_energy'], 2)})")
    import csv
    for nm, data in (('resnet_family_models.csv', list(rows.values())), ('resnet_family_snn_vs_cnn.csv', comp)):
        if not data:
            continue
        p = os.path.join(D.LOCAL, 'summary', nm)
        D._mk(os.path.dirname(p))
        with open(p, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
        D._drive_copy(p, f'summary/{nm}')
    return rows, comp


# ------------------------------------------------------------------ cross-width tests on the MAIN grid (plain stack nets)
def cross_width(seeds=(0, 1, 2)):
    """Does a small SNN at T=16 beat a LARGER best CNN?  (the estimated-energy picture of the accuracy-energy plot)
    Uses the finished main-grid runs (tag 'proto').  Paired McNemar per seed, as everywhere in the paper."""
    old = D.CFG['tag']
    D.CFG['tag'] = 'proto'
    try:
        y = D.data()['te_y'].cpu().numpy()
        M = P.load_pm()
        sn = lambda w, T: D.sid(D.S('snn4', T, 1, w, False))
        cn = lambda w, B: D.sid(D.S('cnn4', 1, B, w, False))
        pairs = [('SNN-S T16 vs CNN-M (B2)', sn(.5, 16), cn(1.0, 2)), ('SNN-S T16 vs CNN-L (B2)', sn(.5, 16), cn(2.0, 2)),
                 ('SNN-M T16 vs CNN-L (B2)', sn(1.0, 16), cn(2.0, 2)), ('SNN-S T8 vs CNN-M (B2)', sn(.5, 8), cn(1.0, 2))]
        out = []
        print('pair                      dAcc   SNN-better  CNN-better  p per seed                est SNN/CNN   GPU-energy SNN/CNN')
        for name, a, b in pairs:
            A, B = D.accs(a, seeds), D.accs(b, seeds)
            ks = [k for k in A if k in B]
            if not ks:
                print(name, ': results not found (check DVS_EXP points to the folder of the main grid)')
                continue
            da = np.mean([A[k]['final_test_acc'] - B[k]['final_test_acc'] for k in ks])
            ps = [D.mcnemar_p(A[k]['preds'], B[k]['preds'], y)[0] for k in ks]
            up = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['final_test_acc'] > B[k]['final_test_acc'])
            dn = sum(1 for k, p in zip(ks, ps) if p < .05 and A[k]['final_test_acc'] < B[k]['final_test_acc'])
            er = P.stats(a)['est_uJ'] / P.stats(b)['est_uJ']
            ea, eb = (M.get(a, {}).get('energy_mJ') or [None])[0], (M.get(b, {}).get('energy_mJ') or [None])[0]
            gr = None if (ea is None or eb is None) else ea / eb
            print(f'{name:25s}{da:+6.2f}   {up}/3         {dn}/3        {[round(p, 3) for p in ps]}   {er:5.2f}        {"-" if gr is None else f"{gr:5.2f}"}')
            out.append(dict(pair=name, dAcc=round(float(da), 2), snn_better=up, cnn_better=dn, p_per_seed=[round(p, 4) for p in ps],
                            est_ratio=round(er, 3), gpu_energy_ratio=None if gr is None else round(gr, 3)))
        if out:
            import csv
            p = os.path.join(D.LOCAL, 'summary', 'cross_width_tests.csv')
            D._mk(os.path.dirname(p))
            with open(p, 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(out[0]))
                wr.writeheader()
                wr.writerows(out)
            D._drive_copy(p, 'summary/cross_width_tests.csv')
        return out
    finally:
        D.CFG['tag'] = old


# ------------------------------------------------------------------ smoke test (run first, ~2 min)
def smoke():
    setup()
    for i in (snn_id(2), cnn_id(2)):
        s = SP[i]
        r = D.train_one(s, 99, None, epochs=1, persist=False, verbose=False)
        m = D.build(s).to(D.DEV).eval()
        fm = fold(m)
        x = D.make_inputs(D.data()['te_snn'][:4].float(), s)
        with torch.no_grad():
            d = (m(x) - fm(x)).abs().max().item()
        g = P._graph(fm, x[:1])
        st = P._stats(m, dict(s, arch='snn4' if s['arch'] == 'rsnn' else 'cnn4'))
        print(f'{i}: input {tuple(x.shape)}  1-epoch acc {r["final_test_acc"]:.1f}  params {r["params"]}  fold diff {d:.1e}  '
              f'graph {"ok" if g else "FAILED (eager only)"}')
        print('   est uJ', round(st['est_uJ'], 2), ' rates', st['rates'][:4], '...')
    print('SMOKE OK')


fold = P.fold


# ------------------------------------------------------------------ package everything with recognisable names
def package():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    name = f'dvs128_resnet_family_{stamp}.zip'
    path = os.path.join('/content', name)
    n = 0
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in (D.LOCAL,):
            for dp, _, fs in os.walk(base):
                for fn in fs:
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, base)
                    if 'resnet' in fn or 'cross_width' in fn or rel.startswith('logs'):
                        z.write(full, os.path.join('dvs128_resnet_family', rel))
                        n += 1
        z.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dvs128_resnet.py'), 'dvs128_resnet_family/code/dvs128_resnet.py')
    print(f'{name}: {n} files, {os.path.getsize(path) / 1e6:.2f} MB')
    return path
