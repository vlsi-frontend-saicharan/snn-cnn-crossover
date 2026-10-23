"""hwmodel.py - bit-exact integer model of the FPGA core (nmnist_core.v) + quantiser + .mem writers."""
import os, json
import numpy as np

BETA = 0.9
BETA_Q = 230
SH = 16
XS = 16.0
ACCW = 24
LOGW = 40
PH = (17, 8, 4)
LIM = (34, 16, 8)
CH = (64, 128, 256)


def lut():
    return np.round(XS * np.log1p(np.arange(256))).astype(np.int64)


def _conv(x, w):
    N, C, H, W = x.shape
    O = w.shape[0]
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)
    out = cols @ w.reshape(O, C * 9).astype(np.float64).T
    return out.reshape(N, H, W, O).transpose(0, 3, 1, 2)


def _conv_int(x, w):
    return np.rint(_conv(x, w)).astype(np.int64)


def _pool(x, ph):
    N, C = x.shape[:2]
    return x[:, :, :2 * ph, :2 * ph].reshape(N, C, ph, 2, ph, 2).max(axis=(3, 5))


def _absmax_scale(w, per_channel=True):
    if per_channel:
        m = np.abs(w).reshape(w.shape[0], -1).max(1)
    else:
        m = np.array([np.abs(w).max()])
    return np.maximum(m, 1e-12) / 127.0


def _qw(w, s):
    sh = (-1,) + (1,) * (w.ndim - 1)
    return np.clip(np.rint(w / s.reshape(sh)), -127, 127).astype(np.int64)


def float_snn(p, counts, T=None):
    N, T = counts.shape[:2]
    x = np.log1p(counts.astype(np.float64))
    mem = [0, 0, 0]
    logits = 0
    ro = (1 - BETA ** (T - np.arange(T))) / (1 - BETA) / T
    for t in range(T):
        h = x[:, t]
        for i in range(3):
            c = _conv(h, p[f"w{i+1}"]) + p[f"b{i+1}"][None, :, None, None]
            mem[i] = BETA * mem[i] + c
            s = (mem[i] - 1.0 > 0).astype(np.float64)
            mem[i] = mem[i] - s
            h = _pool(s, PH[i])
        logits = logits + ro[t] * (h.reshape(N, -1) @ p["wfc"].T + p["bfc"])
    return logits


def float_cnn(p, counts, collect=None):
    h = np.log1p(counts[:, 0].astype(np.float64))
    for i in range(3):
        a = np.maximum(_conv(h, p[f"w{i+1}"]) + p[f"b{i+1}"][None, :, None, None], 0)
        if collect is not None:
            collect.append(a)
        h = _pool(a, PH[i])
    return h.reshape(len(h), -1) @ p["wfc"].T + p["bfc"]


def quantize_snn(p, T):
    layers = []
    for i in range(3):
        w, b = p[f"w{i+1}"], p[f"b{i+1}"]
        s = _absmax_scale(w)
        u = s / XS if i == 0 else s
        layers.append(dict(wq=_qw(w, s), bias=np.rint(b / u).astype(np.int64), scale=np.rint(1.0 / u).astype(np.int64)))
    sf = _absmax_scale(p["wfc"], per_channel=False)
    fc = dict(wq=_qw(p["wfc"], sf), bias=np.rint(p["bfc"] / sf[0]).astype(np.int64))
    ro = (1 - BETA ** (T - np.arange(T))) / (1 - BETA) / T
    return dict(mode="snn", T=T, layers=layers, fc=fc, rw=np.rint(256 * ro).astype(np.int64))


def calibrate_cnn(p, counts):
    acts = []
    float_cnn(p, counts[:, :1], collect=acts)
    return [max(float(np.percentile(a.reshape(-1)[::max(1, a.size // 400000)], 99.9)), 1e-6) / 255.0 for a in acts]


def quantize_cnn(p, s_act):
    layers = []
    s_in = 1.0 / XS
    for i in range(3):
        w, b = p[f"w{i+1}"], p[f"b{i+1}"]
        s = _absmax_scale(w)
        u = s * s_in
        mult = np.rint(u / s_act[i] * (1 << SH)).astype(np.int64)
        assert mult.max() < (1 << 20), f"requant multiplier of layer {i} too large: {mult.max()}"
        layers.append(dict(wq=_qw(w, s), bias=np.rint(b / u).astype(np.int64), scale=mult))
        s_in = s_act[i]
    sf = _absmax_scale(p["wfc"], per_channel=False)
    fc = dict(wq=_qw(p["wfc"], sf), bias=np.rint(p["bfc"] / (sf[0] * s_in)).astype(np.int64))
    return dict(mode="cnn", T=1, layers=layers, fc=fc, rw=np.array([256], dtype=np.int64))


def forward_int(q, counts, stats=None):
    N, T = counts.shape[:2]
    L, fc = q["layers"], q["fc"]
    st = stats if stats is not None else {}
    st.setdefault("maxabs", 0)
    st.setdefault("events", [0.0] * 4)
    st.setdefault("ev_list", [])
    x = lut()[counts]
    if q["mode"] == "snn":
        OUT = (34, 17, 8)
        M = [np.broadcast_to(l["bias"][None, :, None, None], (N, CH[i], OUT[i], OUT[i])).copy() for i, l in enumerate(L)]
        logits = np.zeros((N, 10), np.int64)
        for t in range(T):
            h = x[:, t]
            for i in range(3):
                st["events"][i] += float((h != 0).sum()) / N
                acc = _conv_int(h, L[i]["wq"])
                M[i] += acc
                st["maxabs"] = max(st["maxabs"], int(np.abs(M[i]).max()))
                th = L[i]["scale"][None, :, None, None]
                s = (M[i] > th).astype(np.int64)
                M[i] = M[i] - s * th
                h = _pool(s, PH[i])
                M[i] = ((M[i] * BETA_Q) >> 8) + L[i]["bias"][None, :, None, None]
            st["events"][3] += float((h != 0).sum()) / N
            cur = h.reshape(N, -1) @ fc["wq"].T + fc["bias"][None, :]
            st["maxabs"] = max(st["maxabs"], int(np.abs(cur).max()))
            logits = logits + q["rw"][t] * cur
        st["maxabs_logit"] = int(np.abs(logits).max())
        return logits
    h = x[:, 0]
    for i in range(3):
        st["events"][i] += float((h != 0).sum()) / N
        a = _conv_int(h, L[i]["wq"]) + L[i]["bias"][None, :, None, None]
        st["maxabs"] = max(st["maxabs"], int(np.abs(a).max()))
        a = np.maximum(a, 0)
        o = np.minimum(255, (a * L[i]["scale"][None, :, None, None] + (1 << (SH - 1))) >> SH)
        h = _pool(o, PH[i])
    st["events"][3] += float((h != 0).sum()) / N
    cur = h.reshape(N, -1) @ fc["wq"].T + fc["bias"][None, :]
    st["maxabs"] = max(st["maxabs"], int(np.abs(cur).max()))
    logits = q["rw"][0] * cur
    st["maxabs_logit"] = int(np.abs(logits).max())
    return logits


def _valid_offsets(lim, w):
    nv = np.zeros((w, w), np.int64)
    for y in range(w):
        for x in range(w):
            nv[y, x] = sum(1 for ky in range(3) for kx in range(3) if 0 <= y + 1 - ky < lim and 0 <= x + 1 - kx < lim)
    return nv


def expected_cycles(q, counts, P=64):
    N, T = counts.shape[:2]
    G = (64 // P, 128 // P, 256 // P)
    GI = (1, G[0], G[1], G[2])
    WIN = (34, 17, 8, 4)
    NW = (34 * 34 * G[0], 17 * 17 * G[1], 8 * 8 * G[2])
    nvs = [_valid_offsets(LIM[i], WIN[i]) for i in range(3)]
    L = q["layers"]
    x = lut()[counts]
    out = np.zeros(N, np.int64)
    for n in range(N):
        cyc = 2 * sum(NW) + 10 + 1
        mem = [None] * 3
        for t in range(T):
            h = x[n:n + 1, t]
            for i in range(3):
                cyc += WIN[i] * WIN[i] * GI[i] * 4
                ev = (h[0] != 0).sum(axis=0)
                nv = nvs[i]
                cyc += int(ev.sum()) + int((ev * G[i] * (2 * nv + (9 - nv))).sum())
                cyc += PH[i] * PH[i] * G[i] * 8
                acc = _conv_int(h, L[i]["wq"])
                if q["mode"] == "snn":
                    if t == 0:
                        mem[i] = np.broadcast_to(L[i]["bias"][None, :, None, None], acc.shape).copy()
                    mem[i] = mem[i] + acc
                    th = L[i]["scale"][None, :, None, None]
                    s = (mem[i] > th).astype(np.int64)
                    mem[i] = (((mem[i] - s * th) * BETA_Q) >> 8) + L[i]["bias"][None, :, None, None]
                    h = _pool(s, PH[i])
                else:
                    a = np.maximum(acc + L[i]["bias"][None, :, None, None], 0)
                    o = np.minimum(255, (a * L[i]["scale"][None, :, None, None] + (1 << (SH - 1))) >> SH)
                    h = _pool(o, PH[i])
            cyc += 4 * 4 * G[2] * 4 + 3 * int((h != 0).sum()) + 1
        out[n] = cyc
    return out


def _hex_lanes(vals, bits):
    d = bits // 4
    mask = (1 << bits) - 1
    return ["".join(f"{int(v) & mask:0{d}x}" for v in row[::-1]) for row in vals]


def write_mem(q, counts, logits, outdir, P=64):
    os.makedirs(outdir, exist_ok=True)
    G = (64 // P, 128 // P, 256 // P)
    L = q["layers"]
    rows = []
    for i in range(3):
        wq = L[i]["wq"]
        O, C = wq.shape[:2]
        wt = wq.transpose(1, 2, 3, 0).reshape(C * 9, G[i], P)
        rows += list(wt.reshape(C * 9 * G[i], P))
    open(f"{outdir}/w_conv.mem", "w").write("\n".join(_hex_lanes(np.array(rows), 8)) + "\n")
    for name, key, bits in (("bias", "bias", 24), ("scale", "scale", 24)):
        words = []
