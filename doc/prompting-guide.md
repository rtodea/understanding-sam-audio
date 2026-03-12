# Prompting Guide

SAM-Audio supports three independent prompting strategies. They can be combined (e.g., text + span) for better results.

---

## Decision Tree — Choosing a Prompt Mode

```mermaid
flowchart TD
    START([I want to isolate a sound]) --> Q1{Do I know what\nit sounds like in words?}
    Q1 -->|Yes| Q2{Do I also know\napprox. when it occurs?}
    Q1 -->|No| Q3{Do I have a video\nwith a visual cue?}

    Q2 -->|Yes| COMBO[Text + Span Prompting\n+ predict_spans=False]
    Q2 -->|No, let the model decide| SP[Text + predict_spans=True\nor text-only]
    Q2 -->|No need| TEXT[Text Prompting\npredict_spans=False]

    Q3 -->|Yes| VIS[Visual Prompting\ndraw mask over object]
    Q3 -->|No| SPAN[Span Prompting only\nmark time range]

    TEXT --> DONE([Separate audio])
    SP --> DONE
    COMBO --> DONE
    VIS --> DONE
    SPAN --> DONE
```

---

## Mode 1 — Text Prompting

Describe the target sound using a lowercase noun phrase or verb phrase.

```mermaid
flowchart LR
    subgraph Input
        A[audio mixture] & D[description string]
    end
    subgraph Processing
        A --> P[SAMAudioProcessor]
        D --> P
        P --> B[Batch]
        B --> M[SAMAudio.separate\npredict_spans=False\nreranking_candidates=1]
    end
    subgraph Output
        M --> T[target.wav]
        M --> R[residual.wav]
    end
```

### Tips for good text descriptions

| Instead of... | Use... | Why |
|---------------|--------|-----|
| "There is a man talking" | `"man speaking"` | Match training NP/VP format |
| "Dog" | `"dog barking"` | Include action for specificity |
| "Music in background" | `"background music"` | Noun phrase preferred |
| "The loud thunder sound" | `"thunder"` | Keep it concise |

### When to use `predict_spans`

```mermaid
flowchart LR
    SOUND{Sound type?} -->|Short, non-ambient events\ne.g. a door slam, a gunshot| PS_ON["predict_spans=True\n+ reranking_candidates=8\nbest quality"]
    SOUND -->|Continuous ambience\ne.g. rain, crowd noise| PS_OFF["predict_spans=False\nfaster"]
```

---

## Mode 2 — Visual Prompting

Isolate sounds associated with a visible object in a video by drawing a mask.

```mermaid
flowchart TD
    VID[Video file\n.mp4 / .mov] --> FRAMES[Extract frames]
    FRAMES --> MASK[Draw binary mask\nover target object]
    MASK --> BLEND[processor.mask_videos\nframes × mask]
    BLEND --> PROC[SAMAudioProcessor\nmasked_videos=...]
    VID --> PROC
    PROC --> SEP[SAMAudio.separate]
    SEP --> TGT[Isolated sound\nof masked object]
```

```python
frames, mask = extract_frames_and_mask(video_path)
masked = processor.mask_videos([frames], [mask])

batch = processor(
    audios=[video_path],
    descriptions=[""],          # empty or hint text
    masked_videos=masked,
).to("cuda")

result = model.separate(batch)
```

**Re-ranking for visual mode** uses ImageBind (audio ↔ video similarity) automatically when `masked_videos` is provided.

---

## Mode 3 — Span Prompting

Specify the time ranges (in seconds) where the target sound occurs.

```mermaid
flowchart LR
    subgraph Anchor format
        POS["'+' = target IS present\ne.g. ['+', 2.0, 5.5]"]
        NEG["'-' = target is NOT present\ne.g. ['-', 0.0, 1.0]"]
    end

    subgraph Usage
        ANC[anchors list] --> PROC2[SAMAudioProcessor\nanchors=...]
        PROC2 --> SEP2[SAMAudio.separate\npredict_spans=False]
    end
```

```python
# "dog barking" appears at 6.3–7.0 s and NOT at 0–2 s
anchors = [[
    ["+", 6.3, 7.0],
    ["-", 0.0, 2.0],   # optional negative span
]]

batch = processor(
    audios=[audio_path],
    descriptions=["dog barking"],
    anchors=anchors,
).to("cuda")
```

---

## Combining Modes

All three modes compose naturally:

```mermaid
flowchart LR
    TXT3[Text description] --> P3[SAMAudioProcessor]
    AUD3[Audio / Video] --> P3
    SPN3["Anchor spans\n(manual or predicted)"] --> P3
    MASK3[Video mask\noptional] --> P3
    P3 --> SEP3[SAMAudio.separate]
    SEP3 -->|reranking selects best| OUT3[Separated audio]
```

---

## Re-Ranking Quality vs. Latency Trade-off

```mermaid
xychart-beta
    title "Quality vs. Latency (relative)"
    x-axis ["1 cand.", "2 cand.", "4 cand.", "8 cand."]
    y-axis "Relative value" 0 --> 10
    line "Quality" [5, 6.5, 8, 9.5]
    line "Latency" [2, 4, 6, 9]
```

| Setting | Use case |
|---------|----------|
| `reranking_candidates=1` | Real-time / batch with tight latency |
| `reranking_candidates=4` | Good balance for most use cases |
| `reranking_candidates=8` | Highest quality, offline processing |
