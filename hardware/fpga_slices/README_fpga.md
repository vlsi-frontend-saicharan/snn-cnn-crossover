Vivado pre-board checks (no board needed)

Files: cnn_slice.sv, snn_slice.sv, tb_slices.sv, run_slice.tcl, run_all.py, tstar_from_csv.py
Not run by the author of these files: no Vivado or Verilog simulator was available when they were written.

Step 1  Simulation. Vivado > Create project > add cnn_slice.sv and snn_slice.sv as design sources and tb_slices.sv as a
        simulation source > Run Simulation > Run Behavioral Simulation > Run All. The Tcl console must end with
        "PASS: 600 cycles, no mismatch".
Step 2  One run to test the flow (Vivado command prompt, in this folder):
        vivado -mode batch -source run_slice.tcl -tclargs xc7z020clg400-1 cnn_slice 18 0 0 default default 4.0 test.csv t1
Step 3  python run_all.py --sets A      (continuity with the paper's Table VII)
Step 4  python run_all.py --sets B C D  (the PYNQ-ZU device xczu5eg-sfvc784-1-e)
Step 5  python tstar_from_csv.py
