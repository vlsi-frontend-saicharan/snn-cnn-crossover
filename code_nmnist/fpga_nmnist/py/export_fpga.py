# ===== Colab cell: quantise the trained N-MNIST models and write everything the Vivado project needs =====
# Run in the notebook session where `W` (snn_win) is imported and W.init_run(<run tag>) points at the run folder that holds
# the checkpoints (the 20-epoch win_v1_kd_countinput folder has seeds 0-2 for SNN-L T=1..3 and CNN-L).
# Output: <FPGA_DIR>/export/<cfg>/*.mem  (weights, biases, thresholds, test vectors) + export_report.json
import sys, os, json, importlib
import numpy as np, torch

FPGA_DIR = "/content/drive/MyDrive/nmnist_snn_cnn_work/fpga_nmnist"      # <- where you put the unzipped project folder
SEED, P, NV, NACC = 0, 64, 20, 2000      # checkpoint seed, lanes (must equal the RTL parameter P), test vectors, samples for int accuracy
sys.path.insert(0, FPGA_DIR + "/py")
import hwmodel as H; importlib.reload(H)

CONFIGS = [("snn_T1", ("SNN", "L", 1, True)), ("snn_T2", ("SNN", "L", 2, True)),
           ("snn_T3", ("SNN", "L", 3, True)), ("cnn", ("CNN", "L", 1, True))]

def load_params(spec):
    m = W.build(spec)
    m.load_state_dict(torch.load(W._p("ckpt", f"{W.spec_name(spec)}_s{SEED}_final.pt"), map_location=W.dev, weights_only=False))
    f = m.fuse().eval()
    g = lambda t: t.detach().double().cpu().numpy()
    p = {}
    for i in (1, 2, 3):
        c = getattr(f, f"conv{i}"); p[f"w{i}"] = g(c.weight); p[f"b{i}"] = g(c.bias)
    p["wfc"] = g(f.fc.weight); p["bfc"] = g(f.fc.bias)
    return p, f

report = []
for name, spec in CONFIGS:
    T = spec[2]
    p, f = load_params(spec)
    X, Y = W.get_data(T)["test"]
    counts, labels = X[:NACC].cpu().numpy(), Y[:NACC].cpu().numpy()
    if spec[0] == "SNN":
        q = H.quantize_snn(p, T)
    else:
        idx = torch.randperm(len(W.get_data(1)["train"][0]), generator=torch.Generator().manual_seed(0))[:1000]
        q = H.quantize_cnn(p, H.calibrate_cnn(p, W.get_data(1)["train"][0][idx.to(W.dev)].cpu().numpy()))
    st = {}
    logits = np.concatenate([H.forward_int(q, counts[i:i + 100], st) for i in range(0, len(counts), 100)])
    pred_int = logits.argmax(1)
    pred_torch = W.predict(f, X[:NACC]).cpu().numpy()
    cyc = H.write_mem(q, counts[:NV], logits[:NV], f"{FPGA_DIR}/export/{name}", P)
    row = dict(cfg=name, int_acc=round(100 * float((pred_int == labels).mean()), 2),
               torch_acc=round(100 * float((pred_torch == labels).mean()), 2),
               int_vs_torch_agreement=round(100 * float((pred_int == pred_torch).mean()), 2),
               max_acc=int(st["maxabs"]), acc_bits_needed=int(np.ceil(np.log2(st["maxabs"] + 1))) + 1,
               max_logit_bits=int(np.ceil(np.log2(st["maxabs_logit"] + 1))) + 1,
               events_per_layer=[round(e, 1) for e in st["events"]], expected_cycles_avg=int(cyc.mean()))
    report.append(row); print(row)
    assert row["acc_bits_needed"] <= 24, "accumulator needs more than ACCW=24 bits -> raise ACCW in the RTL and hwmodel"
    assert row["max_logit_bits"] <= 40, "logit needs more than LOGW=40 bits"
os.makedirs(f"{FPGA_DIR}/export", exist_ok=True)
json.dump(report, open(f"{FPGA_DIR}/export/export_report.json", "w"), indent=1)
print("\nCheck: int_acc should be within ~0.3 points of torch_acc, and int_vs_torch_agreement should be above ~99%.")
print("Expected cycles/sample at 100 MHz ->", {r["cfg"]: round(r["expected_cycles_avg"] / 100e6 * 1e3, 3) for r in report}, "ms")
