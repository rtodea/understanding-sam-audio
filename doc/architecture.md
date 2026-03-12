# SAM-Audio Architecture

## High-Level Component Map

```mermaid
graph TB
    subgraph Inputs
        AUD[Audio Waveform<br/>48 kHz]
        TXT[Text Description<br/>noun/verb phrase]
        VID[Video Frames<br/>optional]
        SPN[Time Spans<br/>optional anchors]
    end

    subgraph Encoders
        CODEC[DACVAE Encoder<br/>sam_audio/model/codec.py]
        T5[T5 Text Encoder<br/>sam_audio/model/text_encoder.py]
        PE[Perception Encoder<br/>sam_audio/model/vision_encoder.py]
    end

    subgraph Fusion
        ALIGN[AlignModalities<br/>sam_audio/model/align.py]
    end

    subgraph Generation
        DIT[Diffusion Transformer DiT<br/>sam_audio/model/transformer.py]
        ODE[ODE Solver<br/>torchdiffeq midpoint]
    end

    subgraph Decoding
        DEC[DACVAE Decoder<br/>sam_audio/model/codec.py]
    end

    subgraph Reranking["Re-Ranking (optional)"]
        JUDGE[Judge Ranker<br/>sam_audio/ranking/judge.py]
        CLAP[CLAP Ranker<br/>sam_audio/ranking/clap.py]
        IB[ImageBind Ranker<br/>sam_audio/ranking/imagebind.py]
        SAR[Sound Activity Ranker<br/>sam_audio/ranking/sound_activity.py]
        ENS[Ensemble Ranker<br/>sam_audio/ranking/ranker.py]
    end

    subgraph Outputs
        TGT[target.wav<br/>isolated sound]
        RES[residual.wav<br/>everything else]
    end

    AUD --> CODEC
    TXT --> T5
    VID --> PE
    SPN --> ALIGN

    CODEC --> ALIGN
    T5 --> ALIGN
    PE --> ALIGN

    ALIGN --> DIT
    DIT <-->|iterative steps| ODE
    ODE --> DEC

    DEC --> TGT
    DEC --> RES

    TGT --> JUDGE
    TGT --> CLAP
    TGT --> IB
    TGT --> SAR
    JUDGE & CLAP & IB & SAR --> ENS
    ENS -->|best candidate| TGT
```

---

## Package Structure

```mermaid
graph LR
    subgraph sam_audio
        PROC[processor.py<br/>SAMAudioProcessor]
        subgraph model
            BASE[base.py<br/>BaseModel + HF integration]
            MODEL[model.py<br/>SAMAudio]
            JUDGE_M[judge.py<br/>SAMAudioJudgeModel]
            CFG[config.py<br/>all config dataclasses]
            CODEC_M[codec.py<br/>DACVAE]
            T5_M[text_encoder.py<br/>T5TextEncoder]
            VIS_M[vision_encoder.py<br/>PerceptionEncoder]
            TRANS[transformer.py<br/>DiT]
            ALN[align.py<br/>AlignModalities]
            PTH[patcher.py<br/>Conv1d utils]
            ROPE[rope.py<br/>RotaryEmbedding]
        end
        subgraph ranking
            RNK[ranker.py<br/>Ranker / EnsembleRanker]
            JR[judge.py<br/>JudgeRanker]
            CR[clap.py<br/>ClapRanker]
            IBR[imagebind.py<br/>ImageBindRanker]
            SAR2[sound_activity.py<br/>SoundActivityRanker]
        end
    end

    PROC --> MODEL
    MODEL --> BASE
    MODEL --> CODEC_M
    MODEL --> T5_M
    MODEL --> VIS_M
    MODEL --> TRANS
    MODEL --> ALN
    TRANS --> ROPE
    TRANS --> PTH
    MODEL --> RNK
    RNK --> JR & CR & IBR & SAR2
```

---

## Model Size Variants

| Variant | HuggingFace ID | Training focus |
|---------|---------------|----------------|
| Small | `facebook/sam-audio-small` | General, fastest |
| Base | `facebook/sam-audio-base` | Balanced |
| Large | `facebook/sam-audio-large` | Best overall quality |
| Small TV | `facebook/sam-audio-small-tv` | Better correctness + visual prompting |
| Base TV | `facebook/sam-audio-base-tv` | Better correctness + visual prompting |
| Large TV | `facebook/sam-audio-large-tv` | Best correctness + visual prompting |

### Performance Benchmark (subjective scores, higher = better)

```mermaid
xychart-beta
    title "SAM-Audio Large — Subjective Scores by Category"
    x-axis ["General SFX", "Speech", "Speaker", "Music", "Instr (wild)", "Instr (pro)"]
    y-axis "Score" 0 --> 5
    bar [3.50, 4.03, 3.60, 4.22, 3.66, 4.49]
```

---

## Codec Details — DACVAE

```mermaid
graph LR
    WAV[Waveform<br/>48 kHz, mono] -->|encode| ENC[Conv stack<br/>rates: 2×8×10×12]
    ENC -->|features| FEAT[Latent Features<br/>B × T × 256]
    FEAT -->|decode| DEC[Transposed conv stack]
    DEC --> WAV2[Waveform<br/>48 kHz, mono]

    note1["Hop length = 1920 samples<br/>≈ 25 feature frames per second"]
```

---

## Diffusion Transformer (DiT) Internals

```mermaid
graph TB
    IN[Input: noisy audio features<br/>B × T × d_model]

    subgraph Layer["DiT Layer (×N)"]
        RMS1[RMSNorm]
        SELF[Self-Attention<br/>with QK-norm + RoPE]
        CROSS[Cross-Attention<br/>conditioned on text/video]
        RMS2[RMSNorm]
        FFN[FFN / SwiGLU gated projection]
        TS[Timestep Embedding<br/>modulates scale + shift]
    end

    OUT[Output: denoised features<br/>B × T × d_model]

    IN --> RMS1 --> SELF --> CROSS --> RMS2 --> FFN --> OUT
    TS -.->|AdaLN conditioning| RMS1
    TS -.->|AdaLN conditioning| RMS2
```

---

## Re-Ranking Pipeline

When `reranking_candidates = K > 1`, SAM-Audio generates K independent candidates and selects the best:

```mermaid
flowchart LR
    NOISE["K random noise tensors"] --> GEN["Run DiT K times<br/>(different seeds)"]
    GEN --> CANDS["K candidate<br/>separated audios"]

    CANDS --> J[Judge Ranker<br/>precision · recall · faithfulness]
    CANDS --> C[CLAP Ranker<br/>audio↔text similarity]
    CANDS --> I[ImageBind Ranker<br/>audio↔video similarity<br/>visual mode only]
    CANDS --> S[Sound Activity Ranker<br/>silence detection]

    J & C & I & S --> E[Ensemble Ranker<br/>weighted combination]
    E --> BEST[Best candidate]
```
