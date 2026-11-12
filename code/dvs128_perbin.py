"""
dvs128_perbin.py  --  PER-BIN ReLU CNN baseline for the crossover-timestep study (DVS128 Gesture)   [perbin_v1]

Question (reviewer): the information-matched CNN stacks B bins as channels, so it has a different compute structure from the SNN.
Here the baseline has EXACTLY the SNN's structure: the same 4-block conv net is applied to each of the B bins separately
(one bin per step, shared weights, BatchNorm over bins), ReLU instead of LIF, and the logits are averaged over the B bins.
Same parameters, same MACs-per-bin layout, same input, same recipe (100 epochs, AdamW 1e-3, wd 1e-2, label smoothing 0.1,
+-3 px shifts, final-epoch accuracy, no distillation) as the 'proto' grid.  Only the activation differs (ReLU vs spike).

arch name: 'pcnn'  (T = B bins, K = 1).   Results: <EXP>/results/pbin_pcnn_T<B>K1_w<w>_plain_s<seed>.json
Compared against the finished 'proto' runs (snn4 at the same T, cnn4 stacked) which are read from the same Drive folder.
"""
import os, sys, copy, json, datetime, csv
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/content')
import dvs128_lowT_kd as D
import dvs128_proto as P

D.CFG['tag'] = 'pbin'
VERSION = 'perbin_v1'
TS = (2, 4, 8, 16)
WIDTHS = {'S': 0.5, 'M': 1.0, 'L': 2.0}


class PerBinCNN(D.DeepSNN):
    """DeepSNN with ReLU in place of the LIF layer: same conv/BN/pool/FC, input (N,B,2,H,W), logits averaged over B."""
    def __init__(self, in_ch=2, w=1.0, ncls=11, drop=0.3):
        super().__init__(in_ch, w, ncls, drop)
        self.lif = nn.ReLU()                                      # elementwise, works on (N,B,C,H,W) as is


if not getattr(D, '_perbin_patched', False):
    D._pb_build, D._pb_make = D.build, D.make_inputs

    def _build(s):
        if s['arch'] == 'pcnn':
            return PerBinCNN(2 * s['K'], s['w'], drop=D.CFG['drop'])
        return D._pb_build(s)

    def _make(raw, s):
        if s['arch'] == 'pcnn':
            return D._pb_make(raw, dict(s, arch='snn4'))          # identical input tensor to the SNN
        return D._pb_make(raw, s)

    D.build, D.make_inputs = _build, _make                        # P.fold already handles DeepSNN subclasses
    D._perbin_patched = True


def specs(widths=('M',)):
    out = {}
    for k in widths:
        for T in TS:
            s = D.S('pcnn', T, 1, WIDTHS[k], False)
            out[D.sid(s)] = s
    return out


SP = specs(('S', 'M', 'L'))
D.SPECS.update(SP)
P.SP.update(SP)


def pid(T, w=1.0):
    return D.sid(D.S('pcnn', T, 1, w, False))


def _main_id(kind, T, w):                       # ids of the finished 'proto' grid
    return D.sid(D.S('snn4', T, 1, w, False)) if kind == 'snn' else D.sid(D.S('cnn4', 1, T, w, False))


def setup():
    for d in ('results', 'ckpt', 'measure', 'ops', 'summary', 'logs', 'code'):
        D._mk(os.path.join(D.EXP, d)); D._mk(os.path.join(D.LOCAL, d))
    try:
        D._drive_copy(__file__, 'code/dvs128_perbin.py')
    except Exception:
        pass


def train(seeds=(0, 1, 2), widths=('M',)):
    """Resumable (finished runs are skipped). Default: width M, T in {2,4,8,16}, 3 seeds = 12 runs."""
    setup()
    ids = list(specs(widths))
    for sd in seeds:
        for i in ids:
            r = D.rid(SP[i], sd)
            if D.done(r):
                print('skip (done)', r); continue
            res = D.train_one(SP[i], sd, None)
            print(f'DONE {r}: acc {res["final_test_acc"]:.2f}  params {res["params"]}  {res["train_seconds"]:.0f}s')
    D.sync_to_drive()


def status(widths=('M',)):
    ids = list(specs(widths))
    for sd in (0, 1, 2):
        print(f'seed {sd}:', sum(D.done(D.rid(SP[i], sd)) for i in ids), '/', len(ids))


def smoke():
    setup()
    s = SP[pid(4)]
    r = D.train_one(s, 99, None, epochs=1, persist=False, verbose=False)
    m = D.build(s).to(D.DEV).eval()
    fm = P.fold(m)
    x = D.make_inputs(D.data()['te_snn'][:4].float(), s)
    with torch.no_grad():
        d = (m(x) - fm(x)).abs().max().item()
    xs = D.make_inputs(D.data()['te_snn'][:4].float(), SP[D.sid(D.S('pcnn', 4, 1, 1.0, False))])
    print('input', tuple(x.shape), '1-epoch acc', round(r['final_test_acc'], 1), 'params', r['params'], 'fold diff', f'{d:.1e}')
    snn = D.build(D.S('snn4', 4, 1, 1.0, False))
    print('params equal to SNN of same width:', D.nparams(snn) == r['params'])
    print('SMOKE OK')


# ---------------------------------------------------------------- stats (own file names, 'pbin')
def stats(i, seed=0):
    rel = f'ops/pbin_stats_{i}.json'
    o = D.read_json(rel)
    if o is None:
        o = P._stats(D.load_model(SP[i], seed), SP[i])     # zero-skipping + dense count, ReLU net
        D.save_json(rel, o)
    return o


def measure_all(widths=('M',), rounds=15, e_repeats=3, e_seconds=10):
    """Latency (BN folded, CUDA graph, batch 1) and NVML energy for pcnn and for the matching snn4 / cnn4 in ONE interleaved session."""
    ids = list(specs(widths))
    old = D.CFG['tag']
    out = None
    try:
        # the main-grid checkpoints carry the tag 'proto'; load them under that tag, measure everything together
        for i in ids:
            pass
        main = [_main_id(k, T, WIDTHS[w]) for w in widths for T in TS for k in ('snn', 'cnn')]
        P.SP.update(D.SPECS)
        out = _measure_mixed(ids, main, rounds, e_repeats, e_seconds)
    finally:
        D.CFG['tag'] = old
    return out


def _measure_mixed(pb_ids, main_ids, rounds, e_repeats, e_seconds):
    import time
    models, x1, x64, graphs = {}, {}, {}, {}
    d = D.data()
    for tag, ids in (('pbin', pb_ids), ('proto', main_ids)):
        D.CFG['tag'] = tag
        for i in ids:
            s = P.SP[i] if i in P.SP else D.SPECS[i]
            fm = P.fold(D.load_model(s, 0))
            models[i] = fm
            x1[i] = D.make_inputs(d['te_snn'][:1].float(), s)
            x64[i] = D.make_inputs(d['te_snn'][:64].float(), s)
            with torch.no_grad():
                for _ in range(30):
                    fm(x1[i]); fm(x64[i])
            graphs[i] = P._graph(fm, x1[i])
    ids = list(models)
    LE, LG = {i: [] for i in ids}, {i: [] for i in ids}
    for r in range(rounds):
        k = r % len(ids)
        for i in ids[k:] + ids[:k]:
            LE[i].append(D._lat_run(models[i], x1[i], 100))
            if graphs[i] is not None:
                LG[i].append(P._replay(graphs[i], 100))
    EN = {i: [] for i in ids}
    nv, h = D._nvml()
    if nv is None:
        print('NOTE: NVML unavailable -> no energy (GPU runtime + pip install pynvml)')
    else:
        for r in range(e_repeats):
            k = r % len(ids)
            for i in ids[k:] + ids[:k]:
                time.sleep(8)
                idle = D._idle(nv, h)
                EN[i].append(D._energy_run(models[i], x64[i], e_seconds, nv, h, idle))
            print(f'  energy repeat {r + 1}/{e_repeats}')
    med = lambda v: float(np.median(v)) if len(v) else None
    ms = lambda v: [float(np.mean(v)), float(np.std(v))] if len(v) else None
    out = dict(tag='pbin', seed=0, created=datetime.datetime.now().isoformat(timespec='seconds'), version=VERSION,
               device=torch.cuda.get_device_name(0) if D.DEV == 'cuda' else 'cpu', folded_bn=True,
               models={i: dict(lat_eager_ms=med(LE[i]), lat_graph_ms=med(LG[i]), energy_mJ=ms(EN[i])) for i in ids})
    D.CFG['tag'] = 'pbin'
    D.save_json(f"measure/pbin_pm_{datetime.datetime.now().strftime('%m%d_%H%M')}.json", out)
    return out


def _load_pbin_measure():
    M = {}
    for base in (D.LOCAL, D.EXP):
        dp = os.path.join(base, 'measure')
        if os.path.isdir(dp):
            for fn in sorted(os.listdir(dp)):
                if fn.startswith('pbin_pm_'):
                    try:
                        M.update(json.load(open(os.path.join(dp, fn)))['models'])
                    except Exception:
                        pass
    return M


def _accs(tag, sid_, seeds):
    old = D.CFG['tag']
    D.CFG['tag'] = tag
    try:
        return D.accs(sid_, seeds)
    finally:
        D.CFG['tag'] = old


def _est(tag, i):
    if tag == 'pbin':
        return stats(i)
    D.CFG['tag'] = 'proto'
    try:
        return P.stats(i)
    finally:
        D.CFG['tag'] = 'pbin'


def table(w=1.0, seeds=(0, 1, 2)):
    """Prints: accuracy of SNN(T), per-bin ReLU CNN(B=T), stacked CNN(B=T); McNemar per seed; energy/latency ratios; T* values."""
    M = _load_pbin_measure()
    y = D.data()['te_y'].cpu().numpy()
    rows, comp = {}, []
    wn = {v: k for k, v in WIDTHS.items()}[w]
    print(f'width {wn}:  acc = final epoch mean+-std over seeds; est = zero-skipping op-count uJ (CNNs) / AC-based (SNN); lat ms = batch-1 CUDA graph; E = NVML mJ/sample')
    print(f"{'model':9s}{'T/B':>4s}{'n':>2s}{'acc %':>13s}{'est uJ':>9s}{'lat ms':>8s}{'E mJ':>8s}")
    for T in TS:
        for kind, tag, i in (('SNN', 'proto', _main_id('snn', T, w)), ('CNN-bin', 'pbin', pid(T, w)), ('CNN-stk', 'proto', _main_id('cnn', T, w))):
            A = _accs(tag, i, seeds)
            if not A:
                continue
            a = [A[k]['final_test_acc'] for k in sorted(A)]
            st = _est(tag, i)
            m = M.get(i, {})
            rows[(kind, T)] = dict(acc=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, n=len(a), est=st['est_uJ'],
                                   lat=m.get('lat_graph_ms'), E=(m.get('energy_mJ') or [None])[0], A=A, tag=tag, id=i)
            r = rows[(kind, T)]
            f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
            print(f"{kind:9s}{T:4d}{r['n']:2d}{r['acc']:7.2f}+-{r['sd']:4.2f}{r['est']:9.1f}{f(r['lat']):>8s}{f(r['E']):>8s}")
    print('\nSNN(T) vs per-bin ReLU CNN(B=T)  (ratio SNN/CNN; T* = T / ratio; McNemar per seed, uncorrected)')
    for T in TS:
        a, b = rows.get(('SNN', T)), rows.get(('CNN-bin', T))
        if not a or not b:
            continue
        ks = [k for k in a['A'] if k in b['A']]
        ps = [D.mcnemar_p(a['A'][k]['preds'], b['A'][k]['preds'], y)[0] for k in ks]
        up = sum(1 for k, p in zip(ks, ps) if p < .05 and a['A'][k]['final_test_acc'] > b['A'][k]['final_test_acc'])
        dn = sum(1 for k, p in zip(ks, ps) if p < .05 and a['A'][k]['final_test_acc'] < b['A'][k]['final_test_acc'])
        rat = lambda x, z: None if (x is None or z in (None, 0)) else x / z
        er, lr_, gr = a['est'] / b['est'], rat(a['lat'], b['lat']), rat(a['E'], b['E'])
        ts = lambda r_: '-' if r_ is None else f'{T / r_:.2f}'
        print(f"T={T:2d}: dAcc {a['acc'] - b['acc']:+5.2f}  SNN-better {up}/{len(ks)}  CNN-better {dn}/{len(ks)}  p {[round(p, 3) for p in ps]}  "
              f"est {er:5.2f} (T* {T / er:5.2f})  lat {'-' if lr_ is None else f'{lr_:5.2f}'} (T* {ts(lr_)})  E {'-' if gr is None else f'{gr:5.2f}'} (T* {ts(gr)})")
        comp.append(dict(T=T, dAcc=round(a['acc'] - b['acc'], 2), snn_better=up, cnn_better=dn, est_ratio=round(er, 3), Tstar_est=round(T / er, 2),
                         lat_ratio=lr_, energy_ratio=gr))
    best = max((r for k, r in rows.items() if k[0] == 'CNN-bin'), key=lambda r: r['acc'], default=None)
    if best:
        print(f"\nbest per-bin CNN (chosen on test accuracy, like the paper): {best['id']}  {best['acc']:.2f}+-{best['sd']:.2f}")
    if comp:
        p = os.path.join(D.LOCAL, 'summary', 'perbin_vs_snn.csv')
        D._mk(os.path.dirname(p))
        with open(p, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(comp[0])); wr.writeheader(); wr.writerows(comp)
        D._drive_copy(p, 'summary/perbin_vs_snn.csv')
    return rows, comp


# ---------------------------------------------------------------- identical-accuracy check (no training needed)
def verify_identical(root=None):
    """Recomputes accuracy from the stored predictions of EVERY results JSON and prints the discordant-pair counts for the two
    pairs that looked suspicious (SNN-M T=16 vs SNN-L T=16; CNN-M B=8 vs B=16).  Use it to put the explanation in the paper."""
    y = D.data()['te_y'].cpu().numpy()
    root = root or os.path.join(D.EXP, 'results')
    R = {}
    for fn in sorted(os.listdir(root)):
        if fn.endswith('.json'):
            try:
                j = json.load(open(os.path.join(root, fn)))
            except Exception:
                continue
            if 'preds' in j:
                R[fn[:-5]] = j
    bad = 0
    for k, j in R.items():
        acc = 100 * float((np.asarray(j['preds']) == y).mean())
        if abs(acc - j['final_test_acc']) > 0.01:
            bad += 1
            print('MISMATCH', k, acc, j['final_test_acc'])
    print(f'{len(R)} result files, accuracy recomputed from preds: {bad} mismatches')
    def pair(a, b):
        for sd in (0, 1, 2):
            ka, kb = f'proto_{a}_s{sd}', f'proto_{b}_s{sd}'
            if ka in R and kb in R:
                pa, pb = np.asarray(R[ka]['preds']), np.asarray(R[kb]['preds'])
                ca, cb = pa == y, pb == y
                print(f'  seed {sd}: acc {100 * ca.mean():.2f} vs {100 * cb.mean():.2f}   A right/B wrong {int((ca & ~cb).sum())}   '
                      f'A wrong/B right {int((~ca & cb).sum())}   identical predictions {int((pa == pb).sum())}/{len(y)}')
    print('SNN-M T16 (A) vs SNN-L T16 (B):'); pair(_main_id('snn', 16, 1.0), _main_id('snn', 16, 2.0))
    print('CNN-M B8 (A) vs CNN-M B16 (B):');   pair(_main_id('cnn', 8, 1.0), _main_id('cnn', 16, 1.0))
    print('(different architectures, so different predictions => equal means are a coincidence of the same number of correct samples)')
