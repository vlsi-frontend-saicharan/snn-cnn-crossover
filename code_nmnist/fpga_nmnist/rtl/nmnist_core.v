`timescale 1ns/1ps
// ============================================================================
// nmnist_core.v  -  event-driven inference core for the N-MNIST network
//   3 x [ conv3x3(pad 1) -> LIF (MODE=0) / ReLU+int8 (MODE=1) -> maxpool2 ]  +  FC(4096 -> 10)
//   MODE = 0 : SNN. Layers 2..4 are driven by SPIKES (one add per spike per weight, no multiplier),
//              T = nsteps time steps (1..3), leaky membranes kept on chip.
//   MODE = 1 : CNN baseline of the SAME size. Every non-zero activation is a multiply-accumulate
//              (zero inputs are skipped as well, i.e. the strongest fair baseline). One pass.
//   Both modes share the same lanes (P output channels in parallel), memories, schedule and FSM,
//   so cycle counts and resources are directly comparable.
// Bit-exact with py/hwmodel.py (the testbench checks all 10 logits of every sample).
// ============================================================================
module nmnist_core #(
    parameter MODE   = 0,
    parameter P      = 64,      // parallel lanes (output channels per word); 64 / 32 / 16
    parameter ACCW   = 24,      // membrane / accumulator width (multiple of 4)
    parameter LOGW   = 40,
    parameter BETA_Q = 230,     // membrane decay 230/256 ~ 0.9
    parameter SH     = 16       // CNN requantisation shift
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // input event memory: word = {count of polarity 1, count of polarity 0}, address = (t*34 + y)*34 + x
    input  wire                  in_we,
    input  wire [11:0]           in_addr,
    input  wire [15:0]           in_data,
    input  wire                  start,
    input  wire [1:0]            nsteps,       // 1..3 (CNN: 1)
    output reg                   busy,
    output reg                   done,
    output reg  [3:0]            pred,
    output wire [10*LOGW-1:0]    logits_flat,
    output reg  [31:0]           cycles        // clock cycles of the last inference
);
    // ------------------------------------------------------------------ geometry
    localparam G0 = 64/P, G1 = 128/P, G2 = 256/P;
    localparam NW0 = 34*34*G0, NW1 = 17*17*G1, NW2 = 8*8*G2;
    localparam MB0 = 0, MB1 = NW0, MB2 = NW0 + NW1;
    localparam MD  = NW0 + NW1 + NW2;                       // membrane memory depth
    localparam PD0 = 17*17*G0, PD1 = 8*8*G1, PD2 = 4*4*G2;
    localparam PBI0 = 0, PBI1 = PD0, PBI2 = PD0 + PD1;      // pooled outputs of layers 0,1,2
    localparam PBD = PD0 + PD1 + PD2;
    localparam WN0 = 2*9*G0, WN1 = 64*9*G1, WN2 = 128*9*G2;
    localparam WB0 = 0, WB1 = WN0, WB2 = WN0 + WN1;
    localparam WD  = WN0 + WN1 + WN2;
    localparam PRB0 = 0, PRB1 = G0, PRB2 = G0 + G1;
    localparam PRD = G0 + G1 + G2;
    localparam signed [9:0] BQ = BETA_Q;

    localparam AW_M = $clog2(MD), AW_P = $clog2(PBD), AW_W = $clog2(WD), AW_R = $clog2(PRD);

    // ------------------------------------------------------------------ memories (inferred block RAM / ROM)
    reg [P*8-1:0]    wrom   [0:WD-1];        // conv weights, lane l = output channel g*P+l
    reg [P*ACCW-1:0] biasrom[0:PRD-1];
    reg [P*ACCW-1:0] scalerom[0:PRD-1];      // SNN: threshold, CNN: requant multiplier
    reg [79:0]       fcrom  [0:4095];        // fc weights, 10 classes x int8, index = c*16 + y*4 + x
    reg [10*32-1:0]  fcbias_mem [0:0];
    reg [15:0]       rw_rom [0:8];
    reg [7:0]        lut    [0:255];
    reg [P*ACCW-1:0] mem    [0:MD-1];        // membrane potentials / accumulators of all three layers
    reg [P*8-1:0]    pbuf   [0:PBD-1];       // pooled outputs (spikes 0/1 or int8 activations)
    reg [15:0]       inmem  [0:3*34*34-1];

    initial begin
        $readmemh("w_conv.mem",  wrom);
        $readmemh("bias.mem",    biasrom);
        $readmemh("scale.mem",   scalerom);
        $readmemh("w_fc.mem",    fcrom);
        $readmemh("fc_bias.mem", fcbias_mem);
        $readmemh("rw.mem",      rw_rom);
        $readmemh("lut.mem",     lut);
    end
    wire [10*32-1:0] fcbias = fcbias_mem[0];

    reg  [AW_M-1:0] m_ra, m_wa;   reg m_we;  reg [P*ACCW-1:0] m_wd, mrd;
    reg  [AW_P-1:0] p_ra, p_wa;   reg p_we;  reg [P*8-1:0]    p_wd, prd;
    reg  [11:0]     i_ra;         reg [15:0] ird;
    reg  [AW_W-1:0] w_ra;         reg [P*8-1:0] wrd;
    reg  [AW_R-1:0] pr_a;         reg [P*ACCW-1:0] brd, srd;
    reg  [11:0]     f_ra;         reg [79:0] fcrd;

    always @(posedge clk) begin
        mrd  <= mem[m_ra];
        if (m_we) mem[m_wa] <= m_wd;
        prd  <= pbuf[p_ra];
        if (p_we) pbuf[p_wa] <= p_wd;
        ird  <= inmem[i_ra];
        if (in_we) inmem[in_addr] <= in_data;
        wrd  <= wrom[w_ra];
        brd  <= biasrom[pr_a];
        srd  <= scalerom[pr_a];
        fcrd <= fcrom[f_ra];
    end

    // ------------------------------------------------------------------ FSM registers
    localparam [4:0]
        S_IDLE = 0,  S_INIT_A = 1, S_INIT_B = 2,
        S_SC_A = 3,  S_SC_B = 4,   S_SC_C = 5,  S_SC_N = 6,
        S_EV_A = 7,  S_EV_B = 8,
        S_FR_A = 9,  S_FR_B = 10,
        S_FC_A = 11, S_FC_B = 12,
        S_RO = 13,   S_AM = 14,    S_DONE = 15;
    reg [4:0]  state;
    reg [1:0]  t;
    reg [1:0]  layer;                       // 0..2 conv layers, 3 = FC
    reg [5:0]  sy, sx;  reg [3:0] sg;       // scan position
    reg [1:0]  ky, kx;  reg [3:0] kg;       // kernel position / output group
    reg [4:0]  py, px;  reg [3:0] fg;  reg [1:0] q;   // fire sweep position
    reg [15:0] iw;      reg [3:0] ig;       // init sweep
    reg [P-1:0]   mask;
    reg [P*8-1:0] vals;
    reg [7:0]  ev_ci, ev_v;
    reg [P*8-1:0] pool_acc;
    reg signed [31:0] accfc [0:9];
    reg signed [LOGW-1:0] logit [0:9];
    reg signed [LOGW-1:0] bestv;
    reg [3:0]  bidx, jj;

    // ------------------------------------------------------------------ layer configuration
    reg [5:0] WO, LIM;  reg [4:0] PW;  reg [3:0] G, GI;
    reg [15:0] MB, POB, PIB, WB, PRB, NW;
    always @* begin
        case (layer)
        2'd0:    begin WO = 34; LIM = 34; PW = 17; G = G0; GI = 1;  MB = MB0; POB = PBI0; PIB = 0;    WB = WB0; PRB = PRB0; NW = NW0; end
        2'd1:    begin WO = 17; LIM = 16; PW = 8;  G = G1; GI = G0; MB = MB1; POB = PBI1; PIB = PBI0; WB = WB1; PRB = PRB1; NW = NW1; end
        2'd2:    begin WO = 8;  LIM = 8;  PW = 4;  G = G2; GI = G1; MB = MB2; POB = PBI2; PIB = PBI1; WB = WB2; PRB = PRB2; NW = NW2; end
        default: begin WO = 4;  LIM = 4;  PW = 4;  G = 1;  GI = G2; MB = 0;   POB = 0;    PIB = PBI2; WB = 0;   PRB = 0;    NW = 0;   end
        endcase
    end
    wire use_mult = (MODE == 1) || (layer == 2'd0);      // SNN layers 2..4 only add

    // ------------------------------------------------------------------ scan: events from the source word
    wire [7:0] lut0 = lut[ird[7:0]];
    wire [7:0] lut1 = lut[ird[15:8]];
    wire [P*8-1:0] vmap_in = {{(P*8-16){1'b0}}, lut1, lut0};
    wire [P*8-1:0] vmap    = (layer == 2'd0) ? vmap_in : prd;
    wire [P-1:0]   nzmask;
    genvar l;
    generate for (l = 0; l < P; l = l + 1) begin : NZ
        assign nzmask[l] = |vmap[l*8 +: 8];
    end endgenerate
    reg [6:0] low;
    integer ii;
    always @* begin
        low = 0;
        for (ii = P-1; ii >= 0; ii = ii - 1) if (mask[ii]) low = ii;
    end

    // ------------------------------------------------------------------ accumulate datapath (per lane)
    wire signed [8:0] oy = $signed({3'b000, sy}) + 9'sd1 - $signed({7'd0, ky});
    wire signed [8:0] ox = $signed({3'b000, sx}) + 9'sd1 - $signed({7'd0, kx});
    wire ev_valid = (oy >= 0) && (oy < $signed({3'b000, LIM})) && (ox >= 0) && (ox < $signed({3'b000, LIM}));
    wire [11:0] pix  = oy[5:0] * WO + ox[5:0];
    wire [P*ACCW-1:0] rmw_new;
    generate for (l = 0; l < P; l = l + 1) begin : RMW
        wire signed [ACCW-1:0] ml = mrd[l*ACCW +: ACCW];
        wire signed [7:0]      wl = wrd[l*8 +: 8];
        wire signed [16:0]     mulres = wl * $signed({1'b0, ev_v});
        wire signed [16:0]     pl = use_mult ? mulres : {{9{wl[7]}}, wl};
        assign rmw_new[l*ACCW +: ACCW] = ml + pl;
    end endgenerate

    // ------------------------------------------------------------------ fire datapath (per lane)
    wire [P*8-1:0]    fr_vals;
    wire [P*ACCW-1:0] fr_newm;
    generate for (l = 0; l < P; l = l + 1) begin : FIRE
        wire signed [ACCW-1:0] m = mrd[l*ACCW +: ACCW];
        wire signed [ACCW-1:0] b = brd[l*ACCW +: ACCW];
        if (MODE == 0) begin : SNN
            wire signed [ACCW-1:0] th = srd[l*ACCW +: ACCW];
            wire s = (m > th);
            wire signed [ACCW-1:0] m2 = s ? (m - th) : m;
            wire signed [ACCW+9:0] mm = m2 * BQ;
            wire signed [ACCW+9:0] shf = mm >>> 8;
            assign fr_newm[l*ACCW +: ACCW] = shf[ACCW-1:0] + b;
            assign fr_vals[l*8 +: 8] = {7'b0, s};
        end else begin : CNN
            wire [19:0]       mult = srd[l*ACCW +: 20];
            wire [ACCW-1:0]   ap   = m[ACCW-1] ? {ACCW{1'b0}} : m;
            wire [ACCW+19:0]  pr   = ap * mult + (1 << (SH-1));
            wire [ACCW+19:0]  qv   = pr >> SH;
            assign fr_vals[l*8 +: 8] = (qv > 255) ? 8'd255 : qv[7:0];
            assign fr_newm[l*ACCW +: ACCW] = m;
        end
    end endgenerate
    wire [P*8-1:0] pool_next;
    generate for (l = 0; l < P; l = l + 1) begin : POOL
        wire [7:0] a = fr_vals[l*8 +: 8];
        wire [7:0] c = pool_acc[l*8 +: 8];
        assign pool_next[l*8 +: 8] = (q == 2'd0) ? a : ((a > c) ? a : c);
    end endgenerate
    wire [5:0]  foy  = {py, q[1]};
    wire [5:0]  fox  = {px, q[0]};
    wire [11:0] fpix = foy * WO + fox;

    // ------------------------------------------------------------------ FC datapath
    wire [10*17-1:0] fcprod;
    generate for (l = 0; l < 10; l = l + 1) begin : FCP
        wire signed [7:0]  w = fcrd[l*8 +: 8];
        wire signed [16:0] mulr = w * $signed({1'b0, ev_v});
        assign fcprod[l*17 +: 17] = (MODE == 1) ? mulr : {{9{w[7]}}, w};
    end endgenerate
    wire [3:0] rw_idx = ({2'b00, nsteps} - 4'd1) * 4'd3 + {2'b00, t};
    wire signed [17:0] rwv = {2'b00, rw_rom[rw_idx]};

    // ------------------------------------------------------------------ combinational memory control
    always @* begin
        m_ra = 0; m_wa = 0; m_we = 0; m_wd = 0;
        p_ra = 0; p_wa = 0; p_we = 0; p_wd = 0;
        i_ra = 0; w_ra = 0; pr_a = 0; f_ra = 0;
        case (state)
        S_INIT_A, S_INIT_B: begin
            pr_a = PRB + ig;
            m_wa = MB + iw;
            m_wd = brd;
            m_we = (state == S_INIT_B);
        end
        S_SC_A, S_SC_B: begin
            i_ra = t * 1156 + sy * 34 + sx;
            p_ra = PIB + (sy * WO + sx) * GI + sg;
        end
        S_EV_A, S_EV_B: begin
            m_ra = MB + pix * G + kg;
            m_wa = MB + pix * G + kg;
            m_wd = rmw_new;
            m_we = (state == S_EV_B);
            w_ra = WB + ((ev_ci * 9 + ky * 3 + kx) * G + kg);
        end
        S_FR_A, S_FR_B: begin
            m_ra = MB + fpix * G + fg;
            m_wa = MB + fpix * G + fg;
            m_wd = fr_newm;
            m_we = (state == S_FR_B) && (MODE == 0);
            pr_a = PRB + fg;
            p_wa = POB + (py * PW + px) * G + fg;
            p_wd = pool_next;
            p_we = (state == S_FR_B) && (q == 2'd3);
        end
        S_FC_A, S_FC_B: begin
            f_ra = {ev_ci, sy[1:0], sx[1:0]};
        end
        default: ;
        endcase
    end

    // ------------------------------------------------------------------ sequencing
    wire kg_last = (kg == G - 1);
    wire kx_last = (kx == 2);
    wire ky_last = (ky == 2);
    wire ev_last = kg_last && kx_last && ky_last;
    wire sg_last = (sg == GI - 1);
    wire sx_last = (sx == WO - 1);
    wire sy_last = (sy == WO - 1);
    wire scan_last = sg_last && sx_last && sy_last;
    integer j;

    assign logits_flat[0*LOGW +: LOGW] = logit[0];
    assign logits_flat[1*LOGW +: LOGW] = logit[1];
    assign logits_flat[2*LOGW +: LOGW] = logit[2];
    assign logits_flat[3*LOGW +: LOGW] = logit[3];
    assign logits_flat[4*LOGW +: LOGW] = logit[4];
    assign logits_flat[5*LOGW +: LOGW] = logit[5];
    assign logits_flat[6*LOGW +: LOGW] = logit[6];
    assign logits_flat[7*LOGW +: LOGW] = logit[7];
    assign logits_flat[8*LOGW +: LOGW] = logit[8];
    assign logits_flat[9*LOGW +: LOGW] = logit[9];

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 0; done <= 0; pred <= 0; cycles <= 0;
            t <= 0; layer <= 0; sy <= 0; sx <= 0; sg <= 0; ky <= 0; kx <= 0; kg <= 0;
            py <= 0; px <= 0; fg <= 0; q <= 0; iw <= 0; ig <= 0; mask <= 0; vals <= 0;
            ev_ci <= 0; ev_v <= 0; pool_acc <= 0; bestv <= 0; bidx <= 0; jj <= 0;
            for (j = 0; j < 10; j = j + 1) begin accfc[j] <= 0; logit[j] <= 0; end
        end else begin
            if (busy) cycles <= cycles + 1;
            case (state)
            // ---------------------------------------------------------------- idle / init
            S_IDLE: if (start) begin
                busy <= 1; done <= 0; cycles <= 0;
                t <= 0; layer <= 0; iw <= 0; ig <= 0;
                sy <= 0; sx <= 0; sg <= 0;
                for (j = 0; j < 10; j = j + 1) begin accfc[j] <= 0; logit[j] <= 0; end
                state <= S_INIT_A;
            end
            S_INIT_A: state <= S_INIT_B;
            S_INIT_B: begin
                ig <= (ig == G - 1) ? 4'd0 : ig + 4'd1;
                if (iw == NW - 1) begin
                    iw <= 0; ig <= 0;
                    if (layer == 2'd2) begin layer <= 0; state <= S_SC_A; end
                    else begin layer <= layer + 2'd1; state <= S_INIT_A; end
                end else begin
                    iw <= iw + 16'd1; state <= S_INIT_A;
                end
            end
            // ---------------------------------------------------------------- scan source words, pick events
            S_SC_A: state <= S_SC_B;
            S_SC_B: begin mask <= nzmask; vals <= vmap; state <= S_SC_C; end
            S_SC_C: begin
                if (mask == 0) state <= S_SC_N;
                else begin
                    ev_ci <= sg * P + low;
                    ev_v  <= vals[low*8 +: 8];
                    mask  <= mask & (mask - 1'b1);
                    ky <= 0; kx <= 0; kg <= 0;
                    state <= (layer == 2'd3) ? S_FC_A : S_EV_A;
                end
            end
            S_SC_N: begin
                if (!sg_last) sg <= sg + 4'd1;
                else begin
                    sg <= 0;
                    if (!sx_last) sx <= sx + 6'd1;
                    else begin sx <= 0; sy <= sy_last ? 6'd0 : sy + 6'd1; end
                end
                if (scan_last) begin
                    sy <= 0; sx <= 0; sg <= 0;
                    if (layer == 2'd3) state <= S_RO;
                    else begin py <= 0; px <= 0; fg <= 0; q <= 0; state <= S_FR_A; end
                end else state <= S_SC_A;
            end
            // ---------------------------------------------------------------- accumulate one event into the output map
            S_EV_A: begin
                if (ev_valid) state <= S_EV_B;
                else begin
                    if (!kg_last) kg <= kg + 4'd1;
                    else begin kg <= 0; if (!kx_last) kx <= kx + 2'd1; else begin kx <= 0; ky <= ky_last ? 2'd0 : ky + 2'd1; end end
                    if (ev_last) state <= S_SC_C;
                end
            end
            S_EV_B: begin
                if (!kg_last) kg <= kg + 4'd1;
                else begin kg <= 0; if (!kx_last) kx <= kx + 2'd1; else begin kx <= 0; ky <= ky_last ? 2'd0 : ky + 2'd1; end end
                state <= ev_last ? S_SC_C : S_EV_A;
            end
            // ---------------------------------------------------------------- fire + pool sweep
            S_FR_A: state <= S_FR_B;
            S_FR_B: begin
                pool_acc <= pool_next;
                if (q != 2'd3) begin q <= q + 2'd1; state <= S_FR_A; end
                else begin
                    q <= 0;
                    if (fg != G - 1) begin fg <= fg + 4'd1; state <= S_FR_A; end
                    else begin
                        fg <= 0;
                        if (px != PW - 1) begin px <= px + 5'd1; state <= S_FR_A; end
                        else begin
                            px <= 0;
                            if (py != PW - 1) begin py <= py + 5'd1; state <= S_FR_A; end
                            else begin
                                py <= 0; layer <= layer + 2'd1; state <= S_SC_A;
                            end
                        end
                    end
                end
            end
            // ---------------------------------------------------------------- fully connected layer
            S_FC_A: state <= S_FC_B;
            S_FC_B: begin
                for (j = 0; j < 10; j = j + 1) accfc[j] <= accfc[j] + $signed(fcprod[j*17 +: 17]);
                state <= S_SC_C;
            end
            S_RO: begin
                for (j = 0; j < 10; j = j + 1) begin
                    logit[j] <= logit[j] + rwv * (accfc[j] + $signed(fcbias[j*32 +: 32]));
                    accfc[j] <= 0;
                end
                if ({1'b0, t} + 3'd1 < {1'b0, nsteps}) begin
                    t <= t + 2'd1; layer <= 0; state <= S_SC_A;
                end else begin
                    jj <= 0; state <= S_AM;
                end
            end
            S_AM: begin
                if (jj == 0 || logit[jj] > bestv) begin bestv <= logit[jj]; bidx <= jj; end
                if (jj == 9) state <= S_DONE; else jj <= jj + 4'd1;
            end
            S_DONE: begin
                pred <= bidx; busy <= 0; done <= 1; state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
