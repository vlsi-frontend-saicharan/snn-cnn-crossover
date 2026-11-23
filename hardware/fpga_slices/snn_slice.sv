// snn_slice.sv : spike-gated N-input slice. One clock = one timestep: the 8-bit weight of every
// input that spiked is added (no multiplier). LIF=1 adds a leaky membrane (beta=0.5, soft reset).
`timescale 1ns/1ps
module snn_slice #(
    parameter int N     = 288,
    parameter int W     = 8,
    parameter int LIF   = 0,
    parameter int THETA = 64,
    parameter int SW    = W + $clog2(N)
)(
    input  logic                 clk,
    input  logic                 rst,
    input  logic [N-1:0]         s_in,
    input  logic [N*W-1:0]       w_flat,
    output logic signed [SW-1:0] y,
    output logic                 spike
);
    logic [N-1:0]         s_r;
    logic [N*W-1:0]       w_r;
    logic signed [SW-1:0] sum;

    always_ff @(posedge clk) begin
        s_r <= s_in;
        w_r <= w_flat;
    end

    always_comb begin
        sum = '0;
        for (int i = 0; i < N; i++)
            if (s_r[i]) sum = sum + $signed(w_r[i*W +: W]);
    end

    always_ff @(posedge clk) begin
        if (rst) y <= '0;
        else     y <= sum;
    end

    generate
        if (LIF != 0) begin : g_lif
            logic signed [SW+1:0] v, u;
            logic                 fire;
            always_comb begin
                u    = (v >>> 1) + sum;
                fire = (u >= THETA);
            end
            always_ff @(posedge clk) begin
                if (rst) begin
                    v     <= '0;
                    spike <= 1'b0;
                end else begin
                    spike <= fire;
                    v     <= fire ? (u - THETA) : u;
                end
            end
        end else begin : g_nolif
            always_ff @(posedge clk) spike <= 1'b0;
        end
    endgenerate
endmodule
