"""Checks hwmodel.py without torch: random network, int model vs float model, mem file shapes. Run: python selftest_hwmodel.py"""
import numpy as np, os, tempfile
import hwmodel as H
rng = np.random.default_rng(0)
def rand_params():
    p = {}
    cin = 2
    for i, c in enumerate(H.CH):
        p[f"w{i+1}"] = rng.normal(0, 1, (c, cin, 3, 3)) * (1.0 / np.sqrt(cin * 9)) * (1.3 if i else 0.9)
        p[f"b{i+1}"] = rng.normal(0, 0.05, c); cin = c
    p["wfc"] = rng.normal(0, 1, (10, 4096)) / np.sqrt(4096) * 3; p["bfc"] = rng.normal(0, 0.1, 10)
    return p
def rand_counts(N, T):
    return np.minimum(rng.poisson(0.35, (N, T, 2, 34, 34)), 19).astype(np.uint8)
p = rand_params()
for T in (1, 2, 3):
    c = rand_counts(40, T)
    q = H.quantize_snn(p, T); st = {}
    li = H.forward_int(q, c, st); lf = H.float_snn(p, c)
    agree = (li.argmax(1) == lf.argmax(1)).mean()
    corr = np.corrcoef(li.reshape(-1) / 256.0 * 1.0, lf.reshape(-1))[0, 1]
    print(f"SNN T={T}: argmax agreement int vs float {agree:.2f}, logit correlation {corr:.3f}, events/layer {[round(e) for e in st['events']]}, max|acc| {st['maxabs']} (<2^23={2**23})")
    assert st["maxabs"] < 2 ** 23 and st["maxabs_logit"] < 2 ** 39
c = rand_counts(40, 1)
sa = H.calibrate_cnn(p, c); q = H.quantize_cnn(p, sa); st = {}
li = H.forward_int(q, c, st); lf = H.float_cnn(p, c)
print(f"CNN: agreement {(li.argmax(1) == lf.argmax(1)).mean():.2f}, correlation {np.corrcoef(li.reshape(-1), lf.reshape(-1))[0,1]:.3f}, max|acc| {st['maxabs']}")
assert st["maxabs"] < 2 ** 23
d = tempfile.mkdtemp()
q = H.quantize_snn(p, 3); c = rand_counts(3, 3); lg = H.forward_int(q, c)
cyc = H.write_mem(q, c, lg, d)
for f in sorted(os.listdir(d)):
    ls = open(f"{d}/{f}").read().split("\n")[:-1]
    print(f"{f:14s} lines {len(ls):6d}  hex digits/line {len(ls[0])}")
print("expected cycles:", cyc)
