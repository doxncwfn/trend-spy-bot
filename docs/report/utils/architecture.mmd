flowchart TD

%% =======================
%% INPUT
%% =======================
subgraph INPUT["Input Processing"]
    A0["Input Tensor<br/>Shape: [Batch, 53 Stocks, 60 Days, 6 Features]"]
    Ids["Stock IDs Tensor<br/>Shape: [Batch, 53 Stocks]"]
end

%% =======================
%% TEMPORAL PATH
%% =======================
subgraph TEMP["Temporal Path (A)"]
    A2["LSTM Encoder<br/>2 Layers, 128 Hidden<br/>(Dropout 0.44)"]
    A3["Final Hidden State h₆₀<br/>ℝ^{128}"]
end

%% =======================
%% STATIC PATH
%% =======================
subgraph STATIC["Static Path (B)"]
    B2["Embedding Layer<br/>Stock Identity Vector<br/>ℝ^{16}"]
end

%% =======================
%% FUSION
%% =======================
subgraph FUSION["Fusion Layer"]
    F1["Concatenate<br/>h₆₀ (128) + Embedding (16)<br/>→ Fused Vector (144)"]
end

%% =======================
%% CONTEXT / ALPHA LAYER
%% =======================
subgraph CONTEXT["Context Layer — Alpha Extraction"]
    C0["LayerNorm"]
    C1["Transformer Encoder<br/>1 Layer · 4 Heads"]
    Cnote["Cross-Sectional Attention<br/>(Across Batch/Stocks)"]
end

%% =======================
%% PROBABILISTIC HEAD
%% =======================
subgraph HEAD["Probabilistic Output Head"]
    H1["MLP<br/>Linear → ReLU → Dropout"]
    H2_mu["μ Head<br/>(Mean Return)"]
    H2_sigma["log(σ²) Head<br/>(Exp → Uncertainty)"]
    H3["Normal Distribution<br/>𝒩(μ, σ)"]
    H4["Derived Confidence:<br/>P(R > 0)"]
end

%% =======================
%% LOSS
%% =======================
subgraph LOSS["Training Objective"]
    L1["Hybrid Loss<br/>Ranking Loss (α=0.7) + GNLL"]
end

%% ARROWS / FLOW
A0 --> A2 --> A3
Ids --> B2

A3 --> F1
B2 --> F1

F1 --> C0 --> C1 --> H1
C1 -.- Cnote

H1 --> H2_mu
H1 --> H2_sigma

H2_mu --> H3
H2_sigma --> H3

H3 --> H4
H3 --> L1