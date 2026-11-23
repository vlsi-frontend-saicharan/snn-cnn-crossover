// cnn_slice_dsp.sv (same as cnn_slice.sv, but the multipliers are forced into DSP blocks) : dense N-input slice. N signed 8-bit multiplies + adder tree, one output per clock.
// Inputs and weights are registered so both slices have the same register stage.
`timescale 1ns/1ps
(* use_dsp = "yes" *)
module cnn_slice_dsp #(
    parameter int N  = 288,
    parameter int W  = 8,
    parameter int OW = 2*W + $clog2(N)
)(
    input  logic                 clk,
    input  logic                 rst,
    input  logic [N*W-1:0]       x_flat,
    input  logic [N*W-1:0]       w_flat,
    output logic signed [OW-1:0] y
);
    logic [N*W-1:0]       x_r, w_r;
    logic signed [OW-1:0] acc;

    always_ff @(posedge clk) begin
        x_r <= x_flat;
        w_r <= w_flat;
    end

    always_comb begin
        acc = '0;
        for (int i = 0; i < N; i++)
            acc = acc + $signed(x_r[i*W +: W]) * $signed(w_r[i*W +: W]);
    end

    always_ff @(posedge clk) begin
        if (rst) y <= '0;
        else     y <= acc;
    end
endmodule
