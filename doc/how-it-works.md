# How SAM-Audio Works

## End-to-End Inference — Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Processor as SAMAudioProcessor<br/>processor.py
    participant Codec as DACVAE<br/>codec.py
    participant T5 as T5TextEncoder<br/>text_encoder.py
    participant PE as PerceptionEncoder<br/>vision_encoder.py
    participant Align as AlignModalities<br/>align.py
    participant DiT as Diffusion Transformer<br/>transformer.py
    participant ODE as ODE Solver<br/>torchdiffeq
    participant Ranker as EnsembleRanker<br/>ranking/ranker.py

    User->>Processor: audios, descriptions, [videos], [anchors]
    Processor->>Processor: load & resample audio to 48kHz
    Processor->>Processor: tokenize text descriptions
    Processor->>Processor: extract video frames (if visual)
    Processor-->>User: Batch object

    User->>Codec: encode(audio waveform)
    Codec-->>User: audio features [B, T, 256]

    User->>T5: encode(description tokens)
    T5-->>User: text embeddings [B, L, 768]

    alt visual prompting
        User->>PE: encode(video frames)
        PE-->>User: video embeddings [B, F, 1024]
    end

    User->>Align: fuse(audio_feats, text_emb, [video_emb], [anchors])
    Align-->>User: conditioned context tensor

    loop K candidates (reranking_candidates)
        User->>ODE: solve(y0=noise, t_span=[0,1])
        loop ODE steps (midpoint)
            ODE->>DiT: forward(xt, t, context)
            DiT-->>ODE: velocity prediction
        end
        ODE-->>User: denoised features [B*2, T, 128]
        User->>Codec: decode(features)
        Codec-->>User: target wav + residual wav
    end

    alt reranking_candidates > 1
        User->>Ranker: score(K candidates, description)
        Ranker-->>User: best candidate index
    end

    User-->>User: target.wav, residual.wav
```

---

## Step-by-Step Data Flow

### Step 1 — Audio Encoding

```mermaid
flowchart LR
    WAV["Input waveform\n48 kHz, mono\nany length"] -->|batch_audio| RESAMP[Resample if needed]
    RESAMP --> PAD[Pad/trim to max length]
    PAD --> ENC[DACVAE Encoder\nConv stack ×4\nstrides: 2·8·10·12]
    ENC --> FEAT["Audio features\n[B, T, 256]\n~25 frames/sec"]
```

### Step 2 — Text Encoding

```mermaid
flowchart LR
    DESC["Description string\ne.g. 'man speaking'"] --> TOK[T5 Tokenizer]
    TOK --> EMB[T5 Encoder\nt5-base]
    EMB --> OUT["Text embeddings\n[B, L, 768]"]
    OUT --> MASK["+ attention mask\n[B, L]"]
```

### Step 3 — Visual Encoding (optional)

```mermaid
flowchart LR
    FRAMES["Video frames\n[N, C, H, W]"] --> PREP[Normalize & resize\nto PE input size]
    MASK_VID["Binary mask\n[N, H, W]"] --> BLEND[Blend mask with frames]
    PREP & BLEND --> PE[Perception Encoder\nPE-AV large\nViT backbone]
    PE --> VE["Video embeddings\n[B, F, 1024]"]
```

### Step 4 — Modality Alignment

```mermaid
flowchart TB
    AF["Audio features\n[B, T, 256]"] --> PROJ1[Linear projection\n256 → d_model]
    TE["Text embeddings\n[B, L, 768]"] --> PROJ2[Linear projection\n768 → d_model]
    VE["Video embeddings\n[B, F, 1024]\noptional"] --> PROJ3[1D Conv projection\n1024 → d_model]
    ANC["Temporal anchors\n(+/- spans)"] --> EMBAGG[Span embedding aggregation]

    PROJ1 & PROJ2 & PROJ3 & EMBAGG --> CONCAT[Concatenate context tokens]
    CONCAT --> CTX["Context tensor\n[B, T+L+F+A, d_model]"]
```

### Step 5 — Diffusion Generation (ODE)

```mermaid
flowchart TB
    NOISE["Random noise\n[B*2, T, 256]"] --> Y0[y₀ = noise at t=0]
    Y0 --> ODE_LOOP

    subgraph ODE_LOOP["ODE Integration  t: 0 → 1"]
        T1[t=0.0] --> MID[midpoint t=0.5]
        MID --> T2[t=1.0]

        subgraph DIT_STEP["DiT forward pass (each step)"]
            XT["xₜ + timestep embedding"] --> SELF_ATT[Self-attention\nwith RoPE]
            SELF_ATT --> CROSS_ATT[Cross-attention\n← context]
            CROSS_ATT --> FFN2[FFN / SwiGLU]
            FFN2 --> VEL["velocity v_theta(x_t, t)"]
        end
    end

    ODE_LOOP --> FEAT_OUT["Denoised features\n[B*2, T, 128]"]
    FEAT_OUT --> SPLIT[Split into target + residual halves]
```

### Step 6 — Audio Decoding

```mermaid
flowchart LR
    FEAT_TGT["Target features\n[B, T, 128]"] --> DEC[DACVAE Decoder\nTransposed Conv stack]
    FEAT_RES["Residual features\n[B, T, 128]"] --> DEC2["DACVAE Decoder\nshared weights"]
    DEC --> TGT["target.wav\n48kHz"]
    DEC2 --> RES["residual.wav\n48kHz"]
```

---

## Span Prediction Flow (predict_spans=True)

When enabled, SAM-Audio first predicts the *when* before the *what*:

```mermaid
flowchart TD
    TXT2["Text description"] --> SP[Span Predictor\nencoder inside model]
    AUD2["Audio features"] --> SP
    SP --> SPANS["Predicted time spans\ne.g. [['+', 2.1, 4.7]]"]
    SPANS --> DIFF["Feed spans as anchors\ninto AlignModalities"]
    DIFF --> GEN2["Generate separation\nwith temporal guidance"]
```

---

## Judge Model — Quality Scoring

The Judge model is a separate evaluator, not used during generation:

```mermaid
sequenceDiagram
    participant Caller
    participant JP as SAMAudioJudgeProcessor
    participant JM as SAMAudioJudgeModel

    Caller->>JP: process(mixture, separated, description)
    JP->>JP: tokenize description
    JP->>JP: encode both audios via codec
    JP-->>Caller: JudgeBatch

    Caller->>JM: forward(JudgeBatch)
    JM->>JM: T5 encode description
    JM->>JM: cross-attend over mixture + separated features
    JM-->>Caller: scores {overall, precision, recall, faithfulness}
```
