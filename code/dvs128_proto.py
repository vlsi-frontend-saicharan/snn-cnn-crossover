"""
dvs128_proto.py  --  DVS128 Gesture, SAME protocol grid as the N-MNIST study (add-on to dvs128_lowT_kd.py).

Rows (every one trained with the identical recipe, final-epoch accuracy, 3 seeds, no KD):
  widths   S=0.5  M=1.0  L=2.0            (4-block net: S=(16,32,64,64) M=(32,64,128,128) L=(64,128,256,256))
  SNN      T in {2,4,8,16}, one temporal bin per step (K=1)
  CNN      input = B stacked bins, B in {1,2,4,8,16}  (B=1 -> summed frame, B=T -> same information as an SNN with T steps)
Fair pairs: SNN(w,T) vs CNN(w, B=T) [same information, same width] and vs CNN(w, B=1) [summed frame].

Measured (all models BatchNorm-folded, interleaved round-robin, rotated start order):
  GPU latency bs1 median: eager + CUDA-graph replay;  GPU energy/sample above idle (NVML, own idle reading per model)
Estimated:
  energy  Horowitz 45 nm (module constants E_MAC, E_AC).  SNN = first-layer MACs + ACs x measured spike rate.
          CNN shown two ways: dense, and zero-skipping (the fair one).  Memory traffic NOT counted.
  accelerator latency  cycle model: UNITS=256 units @ 200 MHz. CNN: dense / zero-skip. SNN: first layer dense,
          later layers = ops x spike rate, spike-handling cost c = 1, 2, 3 cycles per op.
Results go to <EXP>/{results,ckpt,measure,ops,summary} with tag 'proto' (never mixes with the v1 KD runs).
"""
import os, sys, json, time, copy, datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/content')
import dvs128_lowT_kd as D

D.CFG['tag'] = 'proto'
VERSION = 'proto_v1'
WIDTHS = {'S': 0.5, 'M': 1.0, 'L': 2.0}
TS = (2, 4, 8, 16)
BS = (1, 2, 4, 8, 16)
UNITS, F_MHZ = 256, 200.0


def specs():
    out = {}
    for w in WIDTHS.values():
        for B in BS:
            s = D.S('cnn4', 1, B, w, False)
            out[D.sid(s)] = s
        for T in TS:
            s = D.S('snn4', T, 1, w, False)
            out[D.sid(s)] = s
    return out


SP = specs()
D.SPECS.update(SP)
ALL = list(SP)
WN = {v: k for k, v in WIDTHS.items()}


def snn_ids():
    return [i for i in ALL if SP[i]['arch'] == 'snn4']


def cnn_ids():
    return [i for i in ALL if SP[i]['arch'] == 'cnn4']


def cnn_stack(s):
    return D.sid(D.S('cnn4', 1, s['T'], s['w'], False))


def cnn_sum(s):
    return D.sid(D.S('cnn4', 1, 1, s['w'], False))


# ------------------------------------------------------------------ training
def train(seeds=(0,), ids=None):
    """Resumable: finished runs are skipped. Order = seed-major, so seed 0 of everything finishes first."""
    D.run_all(ids=ids or ALL, seeds=seeds)


# ------------------------------------------------------------------ estimates (ops, energy, accelerator latency)
@torch.no_grad()
def _stats(model, s):
    mods = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    macs, nz, n, cur, hs = {}, {m: 0.0 for m in mods}, [0], [1], []

    def fh(m, inp, out):
        if m in macs:
            return
        k = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] if isinstance(m, nn.Conv2d) else m.in_features
        macs[m] = out.numel() * k / cur[0]                      # per sample, includes the T repeats

    def ph(m, inp):
        nz[m] += (inp[0] != 0).float().mean().item() * cur[0]   # non-zero fraction of the layer input

    for m in mods:
        hs += [m.register_forward_hook(fh), m.register_forward_pre_hook(ph)]
    model.eval()
    X = D.data()['te_snn']
    for i in range(0, len(X), 64):
        cur[0] = min(64, len(X) - i)
        model(D.make_inputs(X[i:i + 64].float(), s))
        n[0] += cur[0]
    for h in hs:
        h.remove()
    M = [macs[m] for m in mods]
    Z = [nz[m] / n[0] for m in mods]
    if s['arch'] == 'snn4':
        ac = sum(a * b for a, b in zip(M[1:], Z[1:]))
        out = dict(est_uJ=(M[0] * D.E_MAC + ac * D.E_AC) * 1e-6, rates=[round(z, 4) for z in Z[1:]],
                   us={f'c{c}': (M[0] + c * ac) / UNITS / F_MHZ for c in (1, 2, 3)})
    else:
        skip = sum(a * b for a, b in zip(M, Z))
        out = dict(est_uJ=skip * D.E_MAC * 1e-6, est_dense_uJ=sum(M) * D.E_MAC * 1e-6, rates=[round(z, 4) for z in Z],
                   us=dict(skip=skip / UNITS / F_MHZ, dense=sum(M) / UNITS / F_MHZ))
    return out


def stats(i, seed=0):
    rel = f"ops/proto_stats_{i}.json"
    o = D.read_json(rel)
    if o is None:
        o = _stats(D.load_model(SP[i], seed), SP[i])
        D.save_json(rel, o)
    return o


# ------------------------------------------------------------------ measurement (BN folded, interleaved)
def fold(model):
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m = copy.deepcopy(model).eval()
    if isinstance(m, D.DeepSNN):
        for k in range(len(m.convs)):
            m.convs[k] = fuse_conv_bn_eval(m.convs[k], m.bns[k])
            m.bns[k] = nn.Identity()
    else:
        L, out, k = list(m.features), [], 0
        while k < len(L):
            if isinstance(L[k], nn.Conv2d) and k + 1 < len(L) and isinstance(L[k + 1], nn.BatchNorm2d):
                out.append(fuse_conv_bn_eval(L[k], L[k + 1]))
                k += 2
            else:
                out.append(L[k])
                k += 1
        m.features = nn.Sequential(*out)
    return m.eval()


def _graph(model, x):
    if D.DEV != 'cuda':
        return None
    try:
        xs = x.clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(5):
                model(xs)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            model(xs)
        return g
    except Exception as e:
        print('  graph capture failed:', repr(e)[:120])
        return None


def _replay(g, reps=100):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def measure(ids=None, seed=0, rounds=15, energy=True, e_seconds=10, e_repeats=3, tag='m1'):
    ids = ids or ALL
    d = D.data()
    models, x1, x64, graphs = {}, {}, {}, {}
    for i in ids:
        raw = D.load_model(SP[i], seed)
        fm = fold(raw)
        x1[i] = D.make_inputs(d['te_snn'][:1].float(), SP[i])
        x64[i] = D.make_inputs(d['te_snn'][:64].float(), SP[i])
        with torch.no_grad():
            a, b = raw(x64[i]), fm(x64[i])
        agree = (a.argmax(1) == b.argmax(1)).float().mean().item() * 100
        print(f'  fold check {i:24s} max|dlogit| {(a - b).abs().max().item():.2e}  argmax agree {agree:.0f}%')
        models[i] = fm
        with torch.no_grad():
            for _ in range(30):
                fm(x1[i]); fm(x64[i])
        graphs[i] = _graph(fm, x1[i])
    LE, LG = {i: [] for i in ids}, {i: [] for i in ids}
    for r in range(rounds):
        k = r % len(ids)
        for i in ids[k:] + ids[:k]:
            LE[i].append(D._lat_run(models[i], x1[i], 100))
            if graphs[i] is not None:
                LG[i].append(_replay(graphs[i], 100))
        if (r + 1) % 5 == 0:
            print(f'  latency round {r + 1}/{rounds}')
    EN = {i: [] for i in ids}
    nv, h = D._nvml() if energy else (None, None)
    if energy and nv is None:
        print('NOTE: NVML unavailable -> no energy (pip install pynvml; GPU runtime needed)')
    if nv:
        for r in range(e_repeats):
            k = r % len(ids)
            for i in ids[k:] + ids[:k]:
                time.sleep(8)                                    # fall back to idle clocks
                idle = D._idle(nv, h)                            # idle reading right before THIS model
                EN[i].append(D._energy_run(models[i], x64[i], e_seconds, nv, h, idle))
            print(f'  energy repeat {r + 1}/{e_repeats}')
    med = lambda v: float(np.median(v)) if len(v) else None
    ms = lambda v: [float(np.mean(v)), float(np.std(v))] if len(v) else None
    out = dict(tag=tag, seed=seed, created=datetime.datetime.now().isoformat(timespec='seconds'), version=VERSION,
               device=torch.cuda.get_device_name(0) if D.DEV == 'cuda' else 'cpu', folded_bn=True,
               models={i: dict(lat_eager_ms=med(LE[i]), lat_graph_ms=med(LG[i]), energy_mJ=ms(EN[i])) for i in ids})
    D.save_json(f"measure/proto_pm_{datetime.datetime.now().strftime('%m%d_%H%M')}_{tag}.json", out)
    return out


def load_pm():
    js = []
    for base in (D.LOCAL, D.EXP):
        dp = os.path.join(base, 'measure')
        if os.path.isdir(dp):
            for fn in os.listdir(dp):
                if fn.startswith('proto_pm_'):
                    try:
                        js.append(json.load(open(os.path.join(dp, fn))))
                    except Exception:
                        pass
    M = {}
    for j in sorted(js, key=lambda j: j.get('created', '')):
        M.update(j['models'])
    return M


# ------------------------------------------------------------------ tables
def _row(i, seeds, M):
    R = D.accs(i, seeds)
    if not R:
        return None
    a = [R[k]['final_test_acc'] for k in R]
    st = stats(i)
    m = M.get(i, {})
    s = SP[i]
    us = st['us']
    if s['arch'] == 'snn4':
        accel = f"{us['c1']:.0f}-{us['c3']:.0f}"
    else:
        accel = f"{us['skip']:.0f}-{us['dense']:.0f}"
    return dict(id=i, kind='SNN' if s['arch'] == 'snn4' else 'CNN', width=WN[s['w']], T=s['T'] if s['arch'] == 'snn4' else None,
                B=s['K'] if s['arch'] != 'snn4' else s['T'], n=len(a), acc=float(np.mean(a)),
                sd=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, params=list(R.values())[0]['params'],
                est_uJ=st['est_uJ'], est_dense_uJ=st.get('est_dense_uJ'), accel_us=accel,
                lat_eager=m.get('lat_eager_ms'), lat_graph=m.get('lat_graph_ms'),
                e_mJ=(m.get('energy_mJ') or [None])[0], epochs=list(R.values())[0]['epochs'])


def table(seeds=(0, 1, 2), ids=None):
    M = load_pm()
    rows = [r for r in (_row(i, seeds, M) for i in (ids or ALL)) if r]
    f = lambda v, p=3: '-' if v is None else f'{v:.{p}f}'
    print('DVS128 Gesture  (acc = final epoch, mean±std over n seeds; est = op-count; accel = cycle model us (CNN: skip-dense, SNN: c1-c3))')
    print(f"{'model':5s} {'W':1s} {'T/B':>4s} {'n':>1s} {'acc %':>12s} {'params':>8s} {'estE uJ':>8s} {'(dense)':>8s} {'accel us':>10s} "
          f"{'eager ms':>8s} {'graph ms':>8s} {'E mJ':>7s}")
    for r in sorted(rows, key=lambda r: (r['kind'] != 'CNN', 'SML'.index(r['width']), r['T'] or r['B'])):
        tb = f"T{r['T']}" if r['kind'] == 'SNN' else f"B{r['B']}"
        print(f"{r['kind']:5s} {r['width']:1s} {tb:>4s} {r['n']:1d} {r['acc']:6.2f}±{r['sd']:4.2f} {r['params']:8d} {r['est_uJ']:8.1f} "
              f"{f(r['est_dense_uJ'], 1):>8s} {r['accel_us']:>10s} {f(r['lat_eager']):>8s} {f(r['lat_graph']):>8s} {f(r['e_mJ'], 3):>7s}")
    p = os.path.join(D.LOCAL, 'summary', f"proto_table_{datetime.datetime.now().strftime('%m%d_%H%M')}.json")
    D._mk(os.path.dirname(p))
    json.dump(rows, open(p, 'w'), indent=1)
    D._drive_copy(p, os.path.relpath(p, D.LOCAL))
    return rows


def compare(seeds=(0, 1, 2)):
    """SNN(w,T) vs same-width CNN with T stacked bins (same information) and vs summed-frame CNN. Ratios SNN/CNN (<1 = SNN better)."""
    M = load_pm()
    y = D.data()['te_y'].cpu().numpy()
    print('SNN vs CNN. dAcc in points; McNemar p per seed (exact if <25 discordant); ratios SNN/CNN: <1 SNN better')
    for i in snn_ids():
        A = D.accs(i, seeds)
        if not A:
            continue
        for tag, b in (('stack', cnn_stack(SP[i])), ('sum', cnn_sum(SP[i]))):
            B = D.accs(b, seeds)
            ks = [k for k in A if k in B]
            if not ks:
                continue
            da = np.mean([A[k]['final_test_acc'] - B[k]['final_test_acc'] for k in ks])
            ps = [D.mcnemar_p(A[k]['preds'], B[k]['preds'], y)[0] for k in ks]
            sig = sum(1 for k, p in zip(ks, ps) if p < 0.05 and A[k]['final_test_acc'] > B[k]['final_test_acc'])
            ma, mb = M.get(i, {}), M.get(b, {})
            rt = lambda key, ix=None: None if not (ma.get(key) and mb.get(key)) else (
                (ma[key][0] / mb[key][0]) if ix == 0 else ma[key] / mb[key])
            rg, re = rt('lat_graph_ms'), rt('energy_mJ', 0)
            ea, eb = stats(i)['est_uJ'], stats(b)['est_uJ']
            fm = lambda v: '  -  ' if v is None else f'{v:5.2f}'
            print(f"{i:20s} vs {b:20s} [{tag:5s}] dAcc {da:+6.2f} p {['%.3f' % p for p in ps]} sig+ {sig}/{len(ks)}  "
                  f"est {ea / eb:5.2f}  graph-lat {fm(rg)}  energy {fm(re)}")


# N-MNIST rows copied from your finished study (acc mean±std, est uJ, GPU graph latency ms, accel latency us).
# None = not printed yet -> shows '--'. Re-run your N-MNIST accelerator cell with SNN-S/CNN-S/CNN-L included to fill them.
NMNIST = [
    dict(kind='CNN', width='L', T=None, acc=98.77, sd=0.08, est_uJ=115.9, lat_graph=0.135, accel_us=None),
    dict(kind='CNN', width='S', T=None, acc=98.55, sd=0.14, est_uJ=7.3, lat_graph=0.067, accel_us=None),
    dict(kind='SNN', width='L', T=3, acc=98.87, sd=0.07, est_uJ=27.0, lat_graph=0.366, accel_us='234-530'),
    dict(kind='SNN', width='L', T=2, acc=98.65, sd=0.06, est_uJ=17.0, lat_graph=0.250, accel_us='135-292'),
    dict(kind='SNN', width='S', T=1, acc=94.57, sd=0.62, est_uJ=1.8, lat_graph=0.082, accel_us=None),
]


def _pick(rows):
    """Compact set for the paper: best-accuracy CNN per width, every SNN T for L, best SNN for S and M."""
    out = []
    for w in 'SML':
        c = [r for r in rows if r['kind'] == 'CNN' and r['width'] == w]
        if c:
            out.append(max(c, key=lambda r: r['acc']))
    out += sorted([r for r in rows if r['kind'] == 'SNN' and r['width'] == 'L'], key=lambda r: r['T'])
    for w in 'SM':
        c = [r for r in rows if r['kind'] == 'SNN' and r['width'] == w]
        if c:
            out.append(max(c, key=lambda r: r['acc']))
    return out


def combined(rows=None, seeds=(0, 1, 2)):
    """Common table for both datasets -> CSV + LaTeX (written to <EXP>/summary and printed)."""
    rows = rows or [r for r in (_row(i, seeds, load_pm()) for i in ALL) if r]
    dv = _pick(rows)
    f = lambda v, p: '--' if v is None else f'{v:.{p}f}'
    lines, csv = [], ['dataset,model,width,T_or_B,acc_mean,acc_std,est_uJ,gpu_graph_ms,accel_us']
    for name, rs in (('N-MNIST', NMNIST), ('DVS128', dv)):
        for r in rs:
            tb = '--' if r['kind'] == 'CNN' and name == 'N-MNIST' else (f"T={r['T']}" if r['kind'] == 'SNN' else f"B={r['B']}")
            lg = r.get('lat_graph')
            lines.append(f"{name} & {r['kind']}-{r['width']} & {tb} & {r['acc']:.2f}$\\pm${r['sd']:.2f} & {r['est_uJ']:.1f} & "
                         f"{f(lg, 3)} & {r['accel_us'] or '--'} \\\\")
            csv.append(f"{name},{r['kind']},{r['width']},{tb},{r['acc']:.2f},{r['sd']:.2f},{r['est_uJ']:.1f},{f(lg, 3)},{r['accel_us'] or ''}")
    tex = ("\\begin{table}[t]\n\\caption{Same protocol on both datasets: accuracy (last epoch, mean$\\pm$std over seeds), estimated energy per "
           "inference (op-count), measured GPU latency (batch 1, CUDA-graph) and estimated accelerator latency (cycle model).}\n"
           "\\label{tab:common}\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llccccc}\n\\toprule\n"
           "Data & Model & T/B & Acc.\\,(\\%) & Est.\\,E\\,($\\mu$J) & GPU\\,(ms) & Accel.\\,($\\mu$s)\\\\\n\\midrule\n"
           + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    for nm, txt in (('combined.csv', '\n'.join(csv)), ('combined.tex', tex)):
        p = os.path.join(D.LOCAL, 'summary', nm)
        D._mk(os.path.dirname(p))
        open(p, 'w').write(txt)
        D._drive_copy(p, f'summary/{nm}')
    print('\n'.join(csv))
    print('\n' + tex)


# ------------------------------------------------------------------ 2-minute smoke test (run this first)
def smoke():
    D.setup()
    for i in ('snn4_T4K1_w0.5_plain', 'cnn4_T1K4_w0.5_plain'):
        s = SP[i]
        r = D.train_one(s, 99, None, epochs=1, persist=False, verbose=False)
        m = D.build(s).to(D.DEV).eval()
        fm = fold(m)
        x = D.make_inputs(D.data()['te_snn'][:4].float(), s)
        with torch.no_grad():
            d = (m(x) - fm(x)).abs().max().item()
        g = _graph(fm, x[:1])
        print(f'{i}: 1-epoch acc {r["final_test_acc"]:.1f}  params {r["params"]}  fold diff {d:.1e}  graph {"ok" if g else "FAILED (eager only)"}')
        st = _stats(m, s)
        print('   est uJ', round(st['est_uJ'], 2), 'accel us', {k: round(v, 1) for k, v in st['us'].items()})
    print('SMOKE OK')
    try:
        D._drive_copy(__file__, 'code/dvs128_proto.py')
    except Exception:
        pass
