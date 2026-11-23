# One synthesis + implementation + power run of one slice (out-of-context).
# vivado -mode batch -source run_slice.tcl -tclargs PART TOP N LIF MAXDSP XRATE SRATE PERIOD CSV TAG
#   TOP     cnn_slice | snn_slice
#   MAXDSP  0 = LUT multipliers only, -1 = DSP blocks allowed
#   XRATE   CNN data-bit toggle rate in percent of the clock (50 = random data), or "default" (vectorless)
#   SRATE   SNN spike probability per input per cycle (0.1 = 10 percent), or "default" (vectorless)
lassign $argv part top N lif maxdsp xrate srate period csv tag
set here [file dirname [file normalize [info script]]]
read_verilog -sv [file join $here cnn_slice.sv]
read_verilog -sv [file join $here snn_slice.sv]
read_verilog -sv [file join $here cnn_slice_dsp.sv]

if {$top eq "snn_slice"} { set gen "-generic N=$N -generic LIF=$lif" } else { set gen "-generic N=$N" }
eval synth_design -top $top -part $part -mode out_of_context -max_dsp $maxdsp $gen

create_clock -period $period -name clk [get_ports clk]
opt_design
place_design
route_design

proc grab {re text} { if {[regexp $re $text -> v]} { return $v } else { return NA } }

# switching activity (toggle rates are in percent of the clock frequency)
proc setact {objs sp tr label} {
    if {[llength $objs] == 0} { puts "WARN: no objects found for $label"; return }
    set_switching_activity -static_probability $sp -toggle_rate $tr $objs
    puts "ACT $label n=[llength $objs] sp=$sp tr=$tr"
}
proc nets_for {base} {
    set n [get_nets -quiet "${base}\[*\]"]
    if {[llength $n] == 0} { set n [get_nets -quiet -hier "*${base}*"] }
    return $n
}
set is_cnn [string match cnn_slice* $top]
if {($is_cnn && $xrate ne "default") || (!$is_cnn && $srate ne "default")} {
    setact [get_ports rst]      0   0 rst
    setact [get_ports w_flat*]  0.5 0 w_ports
    setact [nets_for w_r]       0.5 0 w_nets          ;# weights are held
    if {$is_cnn} {
        setact [get_ports x_flat*] 0.5 $xrate x_ports
        setact [nets_for x_r]      0.5 $xrate x_nets
    } else {
        set tr [expr {200.0 * $srate * (1.0 - $srate)}]   ;# independent spikes: toggles with prob 2p(1-p)
        setact [get_ports s_in*]   $srate $tr s_ports
        setact [nets_for s_r]      $srate $tr s_nets
    }
}

# thermal conditions (review R4): state them explicitly instead of Vivado's defaults
if {[catch {set_operating_conditions -ambient_temp 25 -airflow 250 -heatsink medium -board medium -board_layers 12to15} err]} { puts "WARN set_operating_conditions: $err" }
set u [report_utilization -return_string]
set p [report_power -return_string]
set tj   [grab {Junction Temperature \(C\)\s*\|\s*([0-9.]+)} $p]
set marg [grab {Thermal Margin \(C\)\s*\|\s*(-?[0-9.]+)} $p]
set conf [grab {Confidence level\s*\|\s*(\w+)} $p]
set luts [grab {(?:Slice|CLB) LUTs\*?\s*\|\s*(\d+)} $u]
set ffs  [grab {(?:Slice|CLB) Registers\s*\|\s*(\d+)} $u]
set dsps [grab {DSP(?:s| Slices)\s*\|\s*(\d+)} $u]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
if {$wns eq ""} {   ;# DSP runs returned no path: read WNS from the summary report instead
    set ts [report_timing_summary -return_string -quiet]
    set wns [grab {WNS\(ns\)[^\n]*\n[^\n]*\n\s*(-?[0-9.]+)} $ts]
}
set dyn  [grab {Dynamic \(W\)\s*\|\s*([<0-9.]+)} $p]
set stat [grab {Device Static \(W\)\s*\|\s*([<0-9.]+)} $p]
set sig  [grab {\|\s*Signals\s*\|\s*([<0-9.]+)} $p]
set logi [grab {\|\s*Logic\s*\|\s*([<0-9.]+)} $p]
set dspp [grab {\|\s*DSP\s*\|\s*([<0-9.]+)} $p]

set outdir [file join $here results $tag]
file mkdir $outdir
set f [open [file join $outdir util.rpt] w];  puts $f $u; close $f
set f [open [file join $outdir power.rpt] w]; puts $f $p; close $f

set fh [open $csv a]
puts $fh "$tag,$part,$top,$N,$lif,$maxdsp,$xrate,$srate,$period,$luts,$ffs,$dsps,$wns,$dyn,$stat,$sig,$logi,$dspp,$tj,$marg,$conf"
close $fh
puts "DONE $tag : LUT=$luts FF=$ffs DSP=$dsps WNS=$wns dyn=$dyn W Tj=$tj margin=$marg conf=$conf"
