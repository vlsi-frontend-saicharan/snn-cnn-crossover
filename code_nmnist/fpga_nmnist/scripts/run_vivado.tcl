# Board-independent flow: simulation -> synthesis -> implementation (out-of-context), one configuration at a time.
#
#   vivado -mode batch -source scripts/run_vivado.tcl -tclargs <cfg> <part> <step>
#     <cfg>  : snn_T1 | snn_T2 | snn_T3 | cnn        (folder export/<cfg>/ made by py/export_fpga.py)
#     <part> : any Vivado part, e.g. xczu7ev-ffvc1156-2-e   (no board needed; change it later when the board is fixed)
#     <step> : sim | synth | impl | all
#
# Run it from the project folder (the one that contains rtl/, scripts/, export/).
set cfg   [lindex $argv 0]
set part  [lindex $argv 1]
set step  [lindex $argv 2]
if {$cfg eq "" || $part eq "" || $step eq ""} { puts "usage: -tclargs <cfg> <part> <sim|synth|impl|all>"; exit 1 }
set root  [file normalize [file join [file dirname [info script]] ..]]
set mode  [expr {[string match cnn* $cfg] ? 1 : 0}]
set T     [expr {$mode ? 1 : [string index $cfg end]}]
set P     64
set nsamp 3
set proj  $root/vivado/$cfg

create_project -force snn_cnn_fpga_$cfg $proj -part $part
add_files -fileset sources_1 $root/rtl/nmnist_core.v
add_files -fileset constrs_1 $root/scripts/constraints.xdc
add_files -fileset sim_1     $root/rtl/tb_nmnist_core.v
foreach f [glob $root/export/$cfg/*.mem] {
    set base [file tail $f]
    if {![string match tv_* $base]} {
        add_files -fileset sources_1 $f
        set_property file_type {Memory Initialization Files} [get_files -of_objects [get_filesets sources_1] $f]
    }
    add_files -fileset sim_1 $f
    set_property file_type {Memory Initialization Files} [get_files -of_objects [get_filesets sim_1] $f]
}
set_property top nmnist_core [get_filesets sources_1]
set_property top tb_nmnist_core [get_filesets sim_1]
set_property generic "MODE=$mode P=$P" [get_filesets sources_1]
set_property generic "MODE=$mode P=$P NSTEPS=$T NSAMP=$nsamp" [get_filesets sim_1]
update_compile_order -fileset sources_1

if {$step eq "sim" || $step eq "all"} {
    launch_simulation
    run all
    close_sim
}

if {$step eq "synth" || $step eq "impl" || $step eq "all"} {
    set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    open_run synth_1
    report_utilization -file $proj/utilization_synth.rpt
    report_timing_summary -file $proj/timing_synth.rpt
    close_design
}

if {$step eq "impl" || $step eq "all"} {
    launch_runs impl_1 -jobs 4
    wait_on_run impl_1
    open_run impl_1
    report_utilization -file $proj/utilization_impl.rpt
    report_timing_summary -file $proj/timing_impl.rpt
    report_power -file $proj/power_vectorless.rpt
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    set fmax [expr {1000.0 / (10.0 - $wns)}]
    puts "SUMMARY cfg=$cfg part=$part WNS_ns=$wns FMAX_MHZ=$fmax"
    set rpt [report_utilization -return_string]
    foreach line [split $rpt "\n"] {
        if {[regexp {CLB LUTs|Slice LUTs|CLB Registers|Slice Registers|Block RAM Tile|DSPs|URAM} $line]} { puts "UTIL $line" }
    }
    puts "Open $proj/power_vectorless.rpt for the vectorless power estimate (see README for the activity-based power flow)."
}
