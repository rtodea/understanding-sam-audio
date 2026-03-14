# WebRTC + SAM-Audio in the Cloud (CTO Brief)

This document outlines practical cloud deployment options for the WebRTC pipeline, with tradeoffs in cost, developer experience, iteration speed, and shutdown behavior.

It is based on current project constraints from `doc/setup-windows.md` and `doc/realtime-webrtc.md`:
- `sam-audio-base` fits comfortably on ~8 GB VRAM.
- `sam-audio-large` typically needs ~9-11 GB VRAM (fp16 strongly recommended).
- Near-real-time behavior is streaming-style (roughly 2-5s), not sub-100ms telephony latency.

---

## What Changes in Cloud

```mermaid
flowchart LR
    Browser[Browser WebRTC client] --> Ingress[Global HTTPS LB + WebSocket/WebRTC signaling]
    Ingress --> API[Control plane API\nsmall CPU pods]
    API --> Queue[Job/stream session queue]
    Queue --> GPU[GPU worker pool\nSAM-Audio inferencing]
    GPU --> Storage[Optional object storage\nsession outputs]
    GPU --> Browser
```

Design principle: keep signaling/control on cheap CPU, and isolate GPU inference into independently scalable workers.

---

## Option Matrix

> Cost ranges are directional monthly estimates for **one continuously active GPU equivalent**, excluding egress/storage/tax. Validate with provider calculators before commitment.

| Option | Typical GPU profile | Cost (rough) | Developer experience | Iteration speed | Shutdown behavior |
|---|---|---:|---|---|---|
| A. GKE dedicated GPU node pool (on-demand) | L4 / A10 / T4 / A100 class | $$-$$$$ | Best fit with existing GKE stack | Good once CI/CD is in place | Good with cluster autoscaler; can scale pool to zero with no pending GPU pods |
| B. GKE mixed on-demand + Spot GPU pool | On-demand baseline + Spot burst | $-$$$ | Good, slightly more ops complexity | Very good for load tests | Excellent cost control; Spot interruptions must be handled |
| C. Vertex AI online endpoints for inference + GKE control plane | Managed endpoint GPU backing | $$-$$$$ | Great managed ML ops, less k8s tuning | Good, but release flow differs from app flow | Good autoscaling; scale-to-zero depends on min replicas/config |
| D. Hybrid burst provider (RunPod/Lambda/etc.) behind GKE API | External GPU pods/servers | $-$$$ | Fastest to try; more vendor integration | Fast for prototyping | Strong manual/TTL shutdown patterns; more moving parts |

Legend: more `$` means higher expected spend.

---

## Option A: Stay Native in GKE (Recommended Default)

Best when you already run managed k8s on Google and want one operational model.

### Architecture

```mermaid
flowchart TB
    subgraph GKE["GKE Cluster"]
        Ingress[Ingress / Gateway]
        API[FastAPI signaling + session API\nCPU node pool]
        Redis[Redis / queue]
        subgraph GPU_POOL["GPU Node Pool"]
            Worker1[SAM worker pod]
            Worker2[SAM worker pod]
        end
        KEDA[KEDA/HPA scaler]
    end

    Browser[Web client] --> Ingress --> API
    API --> Redis
    Redis --> Worker1
    Redis --> Worker2
    KEDA --> GPU_POOL
    Worker1 --> Browser
    Worker2 --> Browser
```

### Pros
- Single platform with existing GKE operations, IAM, observability, and deploy flow.
- Clear separation between CPU control services and GPU workers.
- Works for both real-time-ish sessions and async/offline jobs.

### Cons
- GPU nodes are expensive when idle if autoscaling is not tuned.
- Need careful pod disruption handling for long-running sessions.
- More infra tuning than fully managed inference products.

### Practical Cost/Iteration Notes
- Start with `sam-audio-base` to reduce GPU class requirements and cost.
- Use image + weight caching to reduce cold start and speed iteration.
- Keep a tiny on-demand baseline (or none in non-prod), burst with autoscaling.

---

## Option B: GKE with Spot GPUs for Bursty Workloads

Use on-demand for production floor, Spot for burst and non-prod throughput.

```mermaid
flowchart LR
    Queue[Session queue depth] --> Scaler[KEDA decision]
    Scaler --> OD[On-demand GPU pool\nstable floor]
    Scaler --> SP[Spot GPU pool\nburst capacity]
    SP --> Evict[Spot eviction events]
    Evict --> Retry[Session retry / requeue]
```

### Pros
- Highest cost efficiency for load spikes and experimentation.
- Keeps architecture consistent with GKE-native approach.
- Strong for nightly regression runs and batch backfills.

### Cons
- Session interruption risk during Spot reclaim events.
- Requires retry/resume logic and queue idempotency.
- Slightly harder SLO management for interactive sessions.

### Shutdown
- Very strong: let Spot pools collapse fully when queue is empty.
- Add TTL/idle scale-down on non-prod namespaces.

---

## Option C: Vertex AI Endpoints + GKE Control Plane

Offload model-serving lifecycle while keeping your app/web stack in GKE.

```mermaid
flowchart LR
    Browser --> GKEAPI[GKE API/WebRTC signaling]
    GKEAPI --> Vertex[Vertex AI endpoint\nGPU-managed serving]
    Vertex --> GKEAPI
    GKEAPI --> Browser
```

### Pros
- Managed model serving, scaling, and endpoint operations.
- Reduced custom k8s GPU ops burden.
- Better fit if roadmap includes broader ML platform features.

### Cons
- Two-platform workflow (app on GKE + model on Vertex).
- Potentially higher unit economics at steady high utilization.
- Streaming/WebRTC path can require extra adaptation versus plain request/response.

### Shutdown
- Usually good autoscaling behavior; verify minimum replica and cold-start policies.

---

## Option D: Hybrid External GPU Burst (Fastest to Prove)

Keep GKE control plane; call an external GPU runtime for inferencing.

```mermaid
flowchart TB
    Browser --> GKEAPI[GKE API]
    GKEAPI --> Broker[Routing/broker layer]
    Broker --> GKEGPU[GKE GPU workers]
    Broker --> EXTGPU[External GPU provider]
    GKEGPU --> Browser
    EXTGPU --> Browser
```

### Pros
- Fast proof-of-capacity beyond laptop without waiting for full cluster work.
- Useful if GCP quota or procurement lead times block immediate GPU scaling.
- Good for comparative benchmarking across GPU classes.

### Cons
- Extra vendor/security/networking surface area.
- Operational complexity (credentials, routing, observability split).
- Harder long-term governance and cost attribution.

### Shutdown
- Strong if provider supports stop-on-idle or job TTL termination.

---

## Recommendation for Your Current Context

1. **Primary path:** Option A (GKE-native), with Option B enhancements.
2. **Model strategy:** Start production pilot on `sam-audio-base`; reserve `sam-audio-large` for premium quality tiers or offline processing.
3. **Capacity policy:** Keep control plane always on (CPU), GPU pool autoscaled from zero in non-prod and low baseline in prod.
4. **Risk control:** Add queue-based retries and session fallback when Spot/preemptions occur.

---

## 90-Day Move-Forward Plan

```mermaid
gantt
    title WebRTC Cloud Rollout
    dateFormat  YYYY-MM-DD
    section Foundation
    GPU quota + node pool setup            :a1, 2026-03-16, 10d
    Container hardening + HF cache strategy:a2, after a1, 7d
    section Pilot
    Internal pilot (sam-audio-base)        :b1, after a2, 14d
    SLO + cost dashboard                   :b2, after b1, 7d
    section Scale
    Spot burst pool + retry logic          :c1, after b2, 10d
    External benchmark (optional)          :c2, after c1, 7d
    section Decision
    CTO go/no-go on large-tier rollout     :d1, after c2, 3d
```

---

## KPI Targets for CTO Review

- **Latency:** p95 end-to-end streaming delay by model tier.
- **Quality:** MOS/task success or internal evaluator acceptance.
- **Cost:** cost per processed audio minute and per active session-hour.
- **Reliability:** session success rate and interruption/retry rate.
- **Velocity:** median deploy-to-validation cycle time.

These KPIs make the go-forward decision explicit: keep scaling GKE-native, add Spot aggressively, or offload serving to Vertex/external providers.
