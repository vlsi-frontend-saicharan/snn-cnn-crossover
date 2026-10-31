"""
make_pareto_and_sensitivity.py  --  items 1-3 of the improvement plan. NO training, NO GPU: every number below is copied
from Table III (DVS128, 3-seed means) of the paper. Output files (same folder):
  dvs128_pareto_data.csv        the 30 plotted points (accuracy, estimated energy, measured GPU energy)
  fig_pareto_dvs128.pdf/.png    accuracy versus energy (estimated, measured)
  tstar_sensitivity.csv         T* for each width over energy ratio rho = e_MAC/e_AC and spike-activity scale
  fig_tstar_sensitivity.pdf/png T* versus rho
Method of the sensitivity: from Table III, SNN estimate per step eps = est(T=16)/16 splits into
  eps = e_MAC*N1 + e_AC*A      (N1 = first-layer MACs 0.66/1.33/2.65 M, A = AC-operations at the measured spike rates)
and the best-CNN estimate E_CNN = e_MAC*C.  Then  T*(rho, s) = E_CNN / eps(rho, s)  with  eps = e_AC*(rho*N1 + s*A),
E_CNN = e_AC*rho*C, s = spike-activity scale (1 = measured).  Values derived from rounded table entries (about +-1-2 %).
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# (width, B/T, CNN acc, CNN est uJ, CNN meas mJ, SNN acc, SNN est uJ, SNN meas mJ)  -- Table III
T3 = {
 'S': [(1,89.58,25.6,1.75,85.42, 4.6,1.36),(2,89.35,26.8,1.86,88.43, 9.6,3.77),(4,88.43,29.5,1.95,88.66,19.7,5.74),(8,87.15,33.9,2.23,89.35,39.3,8.88),(16,85.76,40.9,2.31,92.94, 76.2,16.80)],
 'M': [(1,91.09,95.4,2.23,86.69,12.2,4.16),(2,91.44,100.0,2.17,89.00,26.2,5.64),(4,90.86,104.8,1.67,89.47,54.2,9.05),(8,87.50,115.1,1.43,89.81,109.8,17.21),(16,87.50,123.9,1.63,94.21,210.7,31.99)],
 'L': [(1,90.86,376.7,3.61,87.96,37.9,6.61),(2,91.55,386.5,3.66,89.70,82.8,8.77),(4,90.97,391.6,3.93,90.16,176.8,17.80),(8,87.96,405.9,3.95,91.55,355.1,33.16),(16,86.57,422.9,4.18,94.21,677.9,63.72)],
}
rows = []
for w, L in T3.items():
    for (k, ca, ce, cm, sa, se, sm) in L:
        rows.append(dict(model='CNN', width=w, B_or_T=k, acc=ca, est_uJ=ce, meas_mJ=cm))
        rows.append(dict(model='SNN', width=w, B_or_T=k, acc=sa, est_uJ=se, meas_mJ=sm))
with open(os.path.join(HERE, 'dvs128_pareto_data.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

# ---------------------------------------------------------------- item 1: accuracy versus energy
col = {'S': '#1b9e77', 'M': '#d95f02', 'L': '#7570b3'}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.25), sharey=True)
for a, key, lab in ((ax[0], 'est_uJ', 'Estimated energy ($\\mu$J, op-count)'), (ax[1], 'meas_mJ', 'Measured GPU energy (mJ per sample)')):
    for w in 'SML':
        for kind, mk in (('CNN', 's'), ('SNN', 'o')):
            r = [x for x in rows if x['model'] == kind and x['width'] == w]
            a.scatter([x[key] for x in r], [x['acc'] for x in r], marker=mk, s=16, color=col[w],
                      facecolors=col[w] if kind == 'SNN' else 'none', linewidths=1.0, zorder=3)
        for x in rows:
            if x['model'] == 'SNN' and x['width'] == w and x['B_or_T'] == 16:
                a.annotate(f'T=16', (x[key], x['acc']), textcoords='offset points', xytext=(3, 3), fontsize=5.5)
    a.set_xscale('log'); a.set_xlabel(lab, fontsize=7); a.tick_params(labelsize=6.5); a.grid(alpha=.25, lw=.4)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [5, 10, 20, 50, 100, 200, 500] if key == 'est_uJ' else [1, 2, 5, 10, 20, 50]
    a.xaxis.set_major_locator(FixedLocator(tk)); a.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk])); a.xaxis.set_minor_locator(NullLocator())
    lo, hi = a.get_xlim(); a.set_xlim(lo, hi * 1.35)
ax[0].set_ylabel('Accuracy (%)', fontsize=7)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color='k', label='SNN (filled)', markersize=4),
     Line2D([], [], marker='s', ls='', color='k', mfc='none', label='CNN (open)', markersize=4)] + \
    [Line2D([], [], marker='o', ls='', color=col[w], label=w, markersize=4) for w in 'SML']
ax[1].legend(handles=h, fontsize=5.5, loc='lower right', ncol=1, frameon=False)
plt.tight_layout(pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig_pareto_dvs128.{ext}'), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- items 2-3: T* sensitivity
E_MAC, E_AC = 4.6, 0.9          # pJ, fp32 (Horowitz, as in the paper)
N1 = {'S': 0.66, 'M': 1.33, 'L': 2.65}            # M MACs, first layer (paper, Sec. IV-A)
BEST = {'S': 25.6, 'M': 100.0, 'L': 386.5}         # est uJ of best CNN (paper)
EPS = {w: T3[w][-1][5] / 16 for w in 'SML'}        # est uJ per SNN step
A, C = {}, {}
for w in 'SML':
    A[w] = (EPS[w] * 1e6 - E_MAC * N1[w] * 1e6) / E_AC / 1e6      # M AC-ops per step (uJ -> pJ*1e6 ops)
    A[w] = (EPS[w] - E_MAC * N1[w]) / E_AC                           # (uJ - pJ*Mops=uJ) / pJ -> M ops
    C[w] = BEST[w] / E_MAC                                           # M MACs (zero-skipping CNN)
def tstar(w, rho, s=1.0, free_first=False):
    n1 = 0.0 if free_first else N1[w]
    eps = E_AC * (rho * n1 + s * A[w])
    return (E_AC * rho * C[w]) / eps
rho0 = E_MAC / E_AC                       # 5.11 (fp32)
rho8 = (0.2 + 0.03) / 0.03                # 7.67 (8-bit int MAC / 8-bit int add)
out = []
for w in 'SML':
    for rho, nm in ((rho0, 'fp32 (4.6/0.9)'), (rho8, '8-bit int (0.23/0.03)'), (10.0, 'rho=10'), (20.0, 'rho=20')):
        for s in (0.5, 1.0, 2.0):
            out.append(dict(width=w, energy_basis=nm, rho=round(rho, 2), activity_scale=s, Tstar=round(tstar(w, rho, s), 2)))
with open(os.path.join(HERE, 'tstar_sensitivity.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
print('check: T*(fp32, s=1) =', {w: round(tstar(w, rho0), 2) for w in 'SML'}, '  (paper: 5.2-5.4, 7.3-7.6, 8.7-9.3)')
print('first-layer share of eps:', {w: round(E_MAC * N1[w] / EPS[w], 2) for w in 'SML'}, ' (paper: 0.63, 0.46, 0.28)')
print('T*(8-bit, s=1):', {w: round(tstar(w, rho8), 2) for w in 'SML'})
print('T*(8-bit, s=0.5):', {w: round(tstar(w, rho8, .5), 2) for w in 'SML'})
print('T*(fp32, s=0.5):', {w: round(tstar(w, rho0, .5), 2) for w in 'SML'})
print('T*(fp32, s=2):', {w: round(tstar(w, rho0, 2.0), 2) for w in 'SML'})
# smallest rho for which T* >= 16 (SNN energy < CNN at T=16), s=1 and s=0.5
for s in (1.0, 0.5):
    res = {}
    for w in 'SML':
        lo, hi = 0.5, 500
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tstar(w, mid, s) < 16 else (lo, mid)
        res[w] = round(hi, 1)
    print(f'rho needed for T*>=16 at activity x{s}:', res)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
rr = np.linspace(1, 25, 200)
for w in 'SML':
    ax.plot(rr, [tstar(w, r) for r in rr], color=col[w], label=f'{w}, measured rates')
    ax.plot(rr, [tstar(w, r, 0.5) for r in rr], color=col[w], ls='--', lw=.9, label=f'{w}, half rates' if w == 'S' else None)
ax.axhline(16, color='k', lw=.6, ls=':'); ax.text(1.2, 16.6, '$T{=}16$', fontsize=6)
ax.axvline(rho0, color='gray', lw=.5); ax.text(rho0 + .2, 1.0, 'fp32', fontsize=5.5, rotation=90)
ax.axvline(rho8, color='gray', lw=.5); ax.text(rho8 + .2, 1.0, '8-bit', fontsize=5.5, rotation=90)
ax.set_xlabel('energy ratio $e_{MAC}/e_{AC}$', fontsize=7); ax.set_ylabel('$T^{*}$', fontsize=7)
ax.tick_params(labelsize=6.5); ax.grid(alpha=.25, lw=.4); ax.legend(fontsize=5.5, frameon=False, ncol=2)
plt.tight_layout(pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'fig_tstar_sensitivity.{ext}'), dpi=300)
