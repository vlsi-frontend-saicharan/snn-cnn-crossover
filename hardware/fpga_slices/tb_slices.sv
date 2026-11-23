// tb_slices.sv : self-checking testbench, N=18. Prints PASS or FAIL. Run in Vivado xsim.
`timescale 1ns/1ps
module tb_slices;
    localparam int N   = 18;
    localparam int W   = 8;
    localparam int OWC = 2*W + $clog2(N);
    localparam int SWS = W + $clog2(N);

    logic clk = 0;
    logic rst = 1;
    logic [N*W-1:0] x, w;
    logic [N-1:0]   s;
    logic signed [OWC-1:0] yc;
    logic signed [SWS-1:0] ys;
    logic sp;

    cnn_slice #(.N(N), .W(W))          dut_c (.clk(clk), .rst(rst), .x_flat(x), .w_flat(w), .y(yc));
    snn_slice #(.N(N), .W(W), .LIF(0)) dut_s (.clk(clk), .rst(rst), .s_in(s), .w_flat(w), .y(ys), .spike(sp));

    always #2 clk = ~clk;                       // 250 MHz

    function automatic logic signed [OWC-1:0] fc(input logic [N*W-1:0] xx, input logic [N*W-1:0] ww);
        logic signed [OWC-1:0] a;
        a = '0;
        for (int k = 0; k < N; k++) a = a + $signed(xx[k*W +: W]) * $signed(ww[k*W +: W]);
        return a;
    endfunction

    function automatic logic signed [SWS-1:0] fs(input logic [N-1:0] ss, input logic [N*W-1:0] ww);
        logic signed [SWS-1:0] a;
        a = '0;
        for (int k = 0; k < N; k++) if (ss[k]) a = a + $signed(ww[k*W +: W]);
        return a;
    endfunction

    logic signed [OWC-1:0] ec1, ec2;
    logic signed [SWS-1:0] es1, es2;
    int cyc    = 0;
    int errors = 0;

    always @(posedge clk) begin
        ec1 <= fc(x, w);  ec2 <= ec1;
        es1 <= fs(s, w);  es2 <= es1;
        if (cyc > 8) begin
            if (yc !== ec2) begin errors++; $display("CNN mismatch at %0d: %0d vs %0d", cyc, yc, ec2); end
            if (ys !== es2) begin errors++; $display("SNN mismatch at %0d: %0d vs %0d", cyc, ys, es2); end
        end
        cyc <= cyc + 1;
    end

    always @(negedge clk) begin                 // new random inputs every cycle; spikes about 25 percent
        for (int j = 0; j < N*W; j++) begin
            x[j] = $urandom_range(0, 1);
            w[j] = $urandom_range(0, 1);
        end
        for (int j = 0; j < N; j++) s[j] = ($urandom_range(0, 3) == 0);
    end

    initial begin
        x = '0; w = '0; s = '0;
        repeat (3) @(negedge clk);
        rst = 0;
        repeat (600) @(posedge clk);
        if (errors == 0) $display("PASS: 600 cycles, no mismatch");
        else             $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
