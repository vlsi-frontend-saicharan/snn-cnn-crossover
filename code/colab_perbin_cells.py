# ===== COLAB: per-bin CNN baseline + identical-accuracy check =====
# Runtime > Change runtime type > GPU (T4).  Paste each "CELL" block in its own cell, run in order.

# ---- CELL 1: mount Drive, install, copy the code ----
from google.colab import drive
drive.mount('/content/drive')
!pip -q install pynvml
# Upload these 3 files to /content (left Files panel > upload):
#   dvs128_lowT_kd.py   dvs128_proto.py   dvs128_perbin.py
import os
os.environ['DVS_EXP'] = '/content/drive/MyDrive/dvs128_gesture_snn_cnn/v2_lowT_kd'   # folder that holds results/ ckpt/ ops/ measure/ of the 'proto' runs
# (if your folder has another name, change the line above; packed.npz is read from /content/drive/MyDrive/dvs128_gesture_snn_cnn/)

# ---- CELL 2: identical-accuracy check (no training, ~1 min) ----
import sys; sys.path.insert(0, '/content')
import dvs128_lowT_kd as D, dvs128_proto as P, dvs128_perbin as PB
PB.verify_identical()          # prints: mismatches of recomputed accuracy, and the prediction differences of the two suspicious pairs

# ---- CELL 3: smoke test (~2 min). Must end with SMOKE OK ----
PB.smoke()

# ---- CELL 4: train the per-bin ReLU CNN, width M, T=2,4,8,16, 3 seeds = 12 runs (resumable: rerun if Colab disconnects) ----
PB.train(seeds=(0, 1, 2), widths=('M',))
PB.status()

# ---- CELL 5: measure latency and energy of pcnn + matching snn4/cnn4 in ONE interleaved session (~15 min) ----
PB.measure_all(widths=('M',))

# ---- CELL 6: the result table + McNemar tests. Copy this output back to me ----
PB.table(w=1.0)
