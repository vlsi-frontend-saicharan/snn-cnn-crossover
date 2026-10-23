# OPTIONAL, run AFTER 'impl' worked:  vivado -mode batch -source scripts/power_saif.tcl -tclargs <cfg>
# Activity-based power: post-implementation functional simulation records the real switching activity (SAIF),
# then report_power uses it. This is what makes SNN (sparse spikes) and CNN differ in dynamic power.
# It is slow for this design (hours): use NSAMP=1 by editing nsamp below.
set cfg  [lindex $argv 0]
set root [file normalize [file join [file dirname [info script]] ..]]
set proj $root/vivado/$cfg
open_project $proj/snn_cnn_fpga_$cfg.xpr
set mode [expr {[string match cnn* $cfg] ? 1 : 0}]
set T    [expr {$mode ? 1 : [string index $cfg end]}]
set_property generic "MODE=$mode P=64 NSTEPS=$T NSAMP=1" [get_filesets sim_1]
open_run impl_1
launch_simulation -mode post-implementation -type functional
open_saif $proj/activity_$cfg.saif
log_saif [get_objects -r /tb_nmnist_core/dut/*]
run all
close_saif
close_sim
open_run impl_1
read_saif $proj/activity_$cfg.saif -strip_path tb_nmnist_core/dut
report_power -file $proj/power_saif.rpt
puts "wrote $proj/power_saif.rpt"
