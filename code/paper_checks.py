# paper_checks.py  -- run AFTER the same setup you used for dvs128_proto (Drive mounted, files in /content).
# Prints every number the paper still needs. Paste the full output back.
import os, sys, glob, hashlib, json
sys.path.insert(0, '/content')
import numpy as np
import dvs128_proto as P
D = P.D
y = D.data()['te_y'].cpu().numpy()
SEEDS = (0, 1, 2)
W = {'S': 0.5, 'M': 1.0, 'L': 2.0}
BEST = {'S': 1, 'M': 2, 'L': 2}                      # best CNN B per width (Table II)
cid = lambda w, B: D.sid(D.S('cnn4', 1, B, W[w], False))
sid_ = lambda w, T: D.sid(D.S('snn4', T, 1, W[w], False))
R = lambda i: D.accs(i, SEEDS)


def p3(i, j):
    A, B = R(i), R(j)
    ks = [k for k in SEEDS if k in A and k in B]
    out = []
    for k in ks:
        p, n01, n10 = D.mcnemar_p(A[k]['preds'], B[k]['preds'], y)
        out.append((k, p, n01, n10, A[k]['final_test_acc'], B[k]['final_test_acc']))
    return out


print('=== 1. Table II(b): every SNN vs best CNN of its width (k = seeds with p<0.05) ===')
for w in 'SML':
    for T in (2, 4, 8, 16):
        r = p3(sid_(w, T), cid(w, BEST[w]))
        up = sum(1 for _, p, a, b, x, z in r if p < .05 and x > z)
        dn = sum(1 for _, p, a, b, x, z in r if p < .05 and x < z)
        print(f'{w} T={T:2d} vs CNN-{w}: SNN-better {up}/3  CNN-better {dn}/3  p={["%.3f" % q[1] for q in r]}')

print('\n=== 2. Identical-accuracy pairs: discordant counts and checkpoint hashes ===')
pairs = [('SNN-M T16 vs SNN-L T16', sid_('M', 16), sid_('L', 16)),
         ('CNN-M B8 vs CNN-M B16', cid('M', 8), cid('M', 16))]
for name, i, j in pairs:
    A, B = R(i), R(j)
    txt = []
    for k in SEEDS:
        a, b = np.array(A[k]['preds']), np.array(B[k]['preds'])
        dis = int((a != b).sum())                        # items where the two models predict different labels
        txt.append(f'seed{k}: acc {A[k]["final_test_acc"]:.2f}/{B[k]["final_test_acc"]:.2f}, differ on {dis}/{len(y)}')
        for base in (D.LOCAL, D.EXP):
            pa, pb = [os.path.join(base, 'ckpt', f'{D.rid(D.SPECS[z], k)}.pth') for z in (i, j)]
            if os.path.exists(pa) and os.path.exists(pb):
                h = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]
                txt[-1] += f', ckpt md5 {h(pa)} vs {h(pb)}'
                break
    print(name); [print('  ', t) for t in txt]

print('\n=== 3. Control (B=16 CNN) McNemar, SNN-L T=16 vs CNN-L B=16 etc. (for the "3 of 3" claim) ===')
for w in 'SML':
    r = p3(sid_(w, 16), cid(w, 16))
    print(w, ['%.4f' % q[1] for q in r], 'sig', sum(q[1] < .05 for q in r), '/3')

print('\n=== 4. N-MNIST: find stored predictions (SNN-L T=3 and CNN-L) ===')
roots = ['/content/drive/MyDrive', '/content']
hits = []
for r in roots:
    for p in glob.glob(r + '/**/*', recursive=True):
        n = os.path.basename(p).lower()
        if ('nmnist' in p.lower() or 'n-mnist' in p.lower() or 'n_mnist' in p.lower()) and \
           n.endswith(('.json', '.npy', '.npz', '.pt', '.pth', '.csv')) and os.path.getsize(p) < 5e7:
            hits.append(p)
for p in sorted(hits)[:80]:
    print(' ', p)
print('If a json/npy with per-sample predictions (10000 labels per seed) is listed above, paste the lines back;')
print('otherwise run the N-MNIST McNemar cell you used before (it produced the table with A_right_B_wrong) for the pair')
print('A=SNN-L at T=3, B=CNN-L, seeds 0,1,2 and paste its output.')
