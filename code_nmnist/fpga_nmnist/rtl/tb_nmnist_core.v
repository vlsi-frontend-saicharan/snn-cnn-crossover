`timescale 1ns/1ps
// Self-checking testbench: loads NSAMP test samples, runs the core, compares class + all 10 logits with the
// integer model (py/hwmodel.py) and the cycle count with the model's expected cycle count.
module tb_nmnist_core;
    parameter MODE   = 0;
    parameter P      = 64;
    parameter NSTEPS = 3;
    parameter NSAMP  = 3;                     // increase (up to the number exported) for a longer check
    parameter LOGW   = 40;
    parameter CLK_NS = 10;                       // simulation clock only; real speed comes from the timing report

    reg clk = 0, rst_n = 0;
    always #(CLK_NS/2) clk = ~clk;

    reg         in_we = 0;
    reg  [11:0] in_addr = 0;
    reg  [15:0] in_data = 0;
    reg         start = 0;
    wire        busy, done;
    wire [3:0]  pred;
    wire [10*LOGW-1:0] logits_flat;
    wire [31:0] cycles;

    nmnist_core #(.MODE(MODE), .P(P), .LOGW(LOGW)) dut (
        .clk(clk), .rst_n(rst_n), .in_we(in_we), .in_addr(in_addr), .in_data(in_data),
        .start(start), .nsteps(NSTEPS[1:0]), .busy(busy), .done(done), .pred(pred),
        .logits_flat(logits_flat), .cycles(cycles));

    reg [15:0]       tv_in  [0:NSAMP*NSTEPS*1156-1];
    reg [3:0]        tv_cls [0:NSAMP-1];
    reg [LOGW-1:0]   tv_log [0:NSAMP*10-1];
    reg [31:0]       tv_cyc [0:NSAMP-1];
    initial begin
        $readmemh("tv_in.mem",     tv_in);
        $readmemh("tv_class.mem",  tv_cls);
        $readmemh("tv_logits.mem", tv_log);
        $readmemh("tv_cycles.mem", tv_cyc);
    end

    integer n, a, j, logit_err, cls_ok, cyc_err;
    reg [LOGW-1:0] got;
    reg signed [63:0] total_cycles;

    initial begin
        logit_err = 0; cls_ok = 0; cyc_err = 0; total_cycles = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
        for (n = 0; n < NSAMP; n = n + 1) begin
            for (a = 0; a < NSTEPS*1156; a = a + 1) begin
                @(posedge clk);
                in_we <= 1; in_addr <= a; in_data <= tv_in[n*NSTEPS*1156 + a];
            end
            @(posedge clk);
            in_we <= 0;
            start <= 1;
            @(posedge clk);
            start <= 0;
            @(posedge clk);
            wait (done);
            @(posedge clk);
            if (pred == tv_cls[n]) cls_ok = cls_ok + 1;
            for (j = 0; j < 10; j = j + 1) begin
                got = logits_flat[j*LOGW +: LOGW];
                if (got !== tv_log[n*10 + j]) begin
                    logit_err = logit_err + 1;
                    if (logit_err < 6) $display("  sample %0d logit %0d: rtl %h  model %h", n, j, got, tv_log[n*10 + j]);
                end
            end
            if (cycles !== tv_cyc[n]) begin
                cyc_err = cyc_err + 1;
                if (cyc_err < 6) $display("  sample %0d cycles: rtl %0d  expected %0d", n, cycles, tv_cyc[n]);
            end
            total_cycles = total_cycles + cycles;
        end
        $display("RESULT mode=%0d P=%0d T=%0d : class match %0d/%0d, logit mismatches %0d, cycle-count mismatches %0d, avg cycles/sample %0d",
                 MODE, P, NSTEPS, cls_ok, NSAMP, logit_err, cyc_err, total_cycles / NSAMP);
        if (logit_err == 0 && cls_ok == NSAMP) $display("TEST PASSED"); else $display("TEST FAILED");
        $finish;
    end

    // safety stop
    initial begin
        #(CLK_NS * 40000000);
        $display("TIMEOUT");
        $finish;
    end
endmodule
