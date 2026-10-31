# analysis/  (items 1-3, no training; all numbers copied from Table III of the paper, DVS128, 3-seed means)
make_pareto_and_sensitivity.py   regenerates everything below (python3, matplotlib only)
dvs128_pareto_data.csv           the 30 plotted points: model, width, B_or_T, acc, est_uJ, meas_mJ
tstar_sensitivity.csv            T* for each width over energy basis (fp32 4.6/0.9, 8-bit 0.23/0.03, rho=10, rho=20) and spike-activity scale (0.5, 1, 2)
Method: eps = e_MAC*N1 + e_AC*A (N1 first-layer MACs, A = AC ops at measured rates) is recovered from est(T=16)/16 and the paper's N1;
T*(rho, s) = E_CNN / (e_AC*(rho*N1 + s*A)).  Reproduces the paper's T* (5.4, 7.6, 9.1).  Ceiling for rho -> infinity: T* -> C/N1 = 8.4, 16.3, 31.7.
8-bit energies (0.2 pJ multiply, 0.03 pJ add) are from a secondary listing of Horowitz's table: check the original before quoting them in print.
