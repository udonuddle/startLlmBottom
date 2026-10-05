# 8주 로드맵 — LLM from Zero

목표: **8주 뒤 LLM을 API 사용자 수준이 아니라, 학습·추론·메모리·CUDA까지 설명할 수 있는 상태.**

주당 **8~12시간** 기준. 핵심 원칙은 하나:

> **내가 먼저 구현 → 테스트 실패 → 원인 가설 → AI 도움 → reference 확인**
>
> reference 코드를 먼저 베끼지 않는다.

`microGPT`는 dataset→tokenizer→autograd→GPT→Adam→training/inference를 약 200줄의 dependency-free Python으로 압축했고, `bigram → MLP → autograd → attention → full GPT → Adam`의 단계별 progression도 공개되어 있다. 첫 2주의 교재로 특히 좋다. ([microGPT](https://github.com/karpathy/karpathy.github.io/blob/master/_posts/2026-02-12-microgpt.markdown))

---

## 전체 지도

```text
Week 1  Language Model + Autograd
   │
Week 2  Attention + GPT
   │
Week 3  PyTorch + GPU Training
   │
Week 4  Llama Architecture
   │
Week 5  C Inference + KV Cache
   │
Week 6  C Training + Backprop
   │
Week 7  CUDA + Profiling
   │
Week 8  LLM Systems Capstone
```

최종적으로 연결할 것:

```text
Text → Tokenizer → Embedding → Transformer → Logits → Sampling

Training:   forward → loss → backward → Adam

Inference:  Q → K/V cache → Attention → token

Hardware:   PyTorch → C → CUDA → kernel → L2/HBM
```

---

## 매주 고정 루틴

```text
① 개념 공부        1~2h
② 직접 구현        3~4h
③ 테스트/실험      2h
④ reference 비교   1~2h
⑤ AI 구두시험      30m
⑥ README 기록      30m
```

특히 **④를 구현보다 먼저 하지 않는 것**이 중요하다.

---

## Week 1 — Language Model을 가장 밑바닥에서

Transformer는 아직 하지 않는다. 먼저

```text
"emma"
e → m
m → m
m → a
```

라는 **next-token prediction 자체**를 이해한다. microGPT progression 앞부분(bigram → MLP/manual gradient → autograd)을 따라간다.

**직접 구현:** `bigram.py`, `mlp.py`, `value.py`, `nn.py`, `train.py`, `tests/`

1. **Bigram LM** — `count(x_t, x_t+1) → probability → sample`, cross entropy 개념까지 연결
2. **Scalar autograd** — `Value(data, grad)`, `__add__`, `__mul__`, `tanh`, `exp`, `backward`

```text
a ──┐
    × → c ─┐
b ──┘      + → L
        d ─┘
```

에서 gradient가 역전파되는 것을 직접 확인.

**반드시 할 실험:** finite difference vs backprop — 오차가 충분히 작아야 한다.

**완료 조건 (코드 없이 답하기):**

- language model은 무엇을 학습하는가?
- logits와 probability의 차이는?
- cross entropy가 감소한다는 건 무엇인가?
- computational graph란?
- backward pass가 실제로 하는 일은?
- parameter는 어떻게 업데이트되는가?

---

## Week 2 — Attention을 직접 만들어 GPT 완성

이 주가 핵심. microGPT 후반부: position embedding, single-head attention, RMSNorm, residual, multi-head attention, 여러 layer, Adam → 완전한 GPT.

**직접 구현:** `embedding.py`, `attention.py`, `mlp.py`, `norm.py`, `block.py`, `gpt.py`, `generate.py`

```text
Token embedding + Position embedding
      ↓
      X
      ↓
 Q    K    V
  \   |   /
  Attention
      ↓
  residual
      ↓
     MLP
      ↓
  residual
      ↓
   logits
```

**손으로 한번 계산:** token 3개, hidden dim 4로 `Q=XWq, K=XWk, V=XWv` → `QKᵀ` → causal mask → softmax → V weighted sum 까지 NumPy로 확인. PyTorch부터 쓰지 않는다.

**실험:**

- A. causal mask 제거 → training loss ↓↓↓. 그런데 왜 문제가 되는지 설명
- B. position embedding 제거
- C. 1 head → 4 head
- D. temperature 0.2 / 0.7 / 1.0 / 1.5 생성 결과 비교

**완료 조건:** 화이트보드에 설명

```text
Q: "내가 지금 무엇을 찾고 싶은가?"
K: "나는 어떤 정보인가?"
V: "내 실제 내용은 무엇인가?"
```

- causal attention이 왜 필요한가?
- attention matrix는 왜 `T×T`인가?
- context가 2배면 어떤 비용이 커지는가?
- residual connection은 왜 있는가?
- multi-head는 왜 쓰는가?

---

## Week 3 — PyTorch + GPU에서 진짜 학습

직접 만든 알고리즘을 PyTorch tensor로 다시 만든다. 처음으로 CPU → GPU.

[build-nanogpt](https://github.com/karpathy/build-nanogpt)는 빈 파일에서 시작해 GPT-2 124M 재현까지 commit을 단계적으로 남겨놓은 repo라 이 시점부터 reference로 쓰기 좋다.

**구현:** `model.py`, `dataset.py`, `train.py`, `generate.py`, `benchmark.py`

- 금지: `nn.Transformer`, `nn.MultiheadAttention`, HuggingFace GPT model
- 허용: `nn.Linear`, `nn.Embedding`, tensor matmul, autograd, AdamW

**GPU 학습** — 처음엔 작게: `d_model=256, layers=4, heads=4, context=256`. TinyStories나 작은 corpus면 충분.

**반드시 기록 (CSV):** parameters, batch size, context, training loss, validation loss, GPU memory, tokens/sec

```text
run,d_model,layers,context,tok_s,vram,val_loss
001,128,4,256,...
002,256,4,256,...
003,256,6,256,...
```

**처음으로 profiler** — 아직 `ncu`는 아니다. `torch.profiler`로 matmul / attention / MLP / LayerNorm / optimizer 중 시간이 어디에 쓰이는지만 본다.

---

## Week 4 — GPT에서 Llama로 진화

[llama2.c](https://github.com/karpathy/llama2.c)를 시작한다. PyTorch에서 작은 Llama를 학습하고 C에서 inference하는 학습용 full-stack 구조이며, 핵심 C inference engine이 단일 파일 중심이다.

```text
GPT:    LayerNorm, GELU, learned positional embedding, MHA
  ↓
Llama:  RMSNorm, SwiGLU, RoPE, GQA/MQA 계열, bias 없는 Linear
```

**하나씩 변경:**

```text
Baseline GPT
+ RMSNorm  → loss / speed
+ RoPE     → loss
+ SwiGLU   → parameters / loss
+ GQA      → KV memory
```

그러면 최신 LLM architecture가 단순 암기가 아니라 "왜 GPT 구조에서 이것으로 바뀌었지?"로 이해된다.

**완료 조건:** LayerNorm vs RMSNorm / absolute position vs RoPE / GELU vs SwiGLU / MHA vs GQA vs MQA.
특히 **GQA가 왜 KV-cache 크기를 줄이는가?** 는 꼭 설명.

---

## Week 5 — `llama2.c`: Python을 버리고 C로 inference

```text
Python model → checkpoint → binary file → C → malloc() → matmul() → attention() → token
```

**`run.c`를 처음부터 읽지 말 것.** 먼저 직접 C skeleton:

```c
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
} Config;
```

`load_weights()`, `rmsnorm()`, `matmul()`, `rope()`, `softmax()`, `attention()`, `forward()` 를 만들고, 막힌 뒤 `run.c`를 본다.

**최중요: KV cache** — 직접 array를 만든다.

```text
key_cache  [layer][token][kv_head][head_dim]
value_cache[layer][token][kv_head][head_dim]

new token → Q_new, K_new / V_new → cache에 append → 기존 K/V 읽기 → attention
```

**직접 계산:** `layers=32, kv_heads=8, head_dim=128, seq=8192, FP16` 이면 KV cache가 몇 GB인지. 공식만 외우지 말고 **array layout에서 byte 수를 유도**.

---

## Week 6 — `llm.c`: backward를 C에서 이해

[llm.c](https://github.com/karpathy/llm.c)는 GPT-2/GPT-3 계열 pretraining에 초점을 두며, PyTorch reference와 약 1,000줄 규모의 FP32 CPU C reference를 함께 제공한다.

첫 목표는 CUDA가 아니다. `train_gpt2.py` vs `train_gpt2.c` 비교.

Linear `Y = XW` 에서 forward만이 아니라 `dX`, `dW`가 실제 C array에 어떻게 계산되는지 추적. Embedding / LayerNorm / Attention / MLP / Cross entropy의 forward/backward pair 정리.

**산출물:** `notes/backprop.md`

```text
# Linear
Forward:   Y = XW
Backward:  dX = dY Wᵀ
           dW = Xᵀ dY

# Attention
QKᵀ → softmax → PV
Backward: ...
```

**완료 조건:** `loss.backward()` 없이 "Transformer parameter의 gradient는 어디서 오는가?"를 큰 흐름으로 설명.

---

## Week 7 — CUDA + profiler

이제야 `ncu`가 등장. 순서는 반드시 **PyTorch profiler → Nsight Systems → Nsight Compute**.

**1. 전체 timeline (`nsys`)** — GPU가 언제 놀고 있나? CPU launch overhead? Memcpy? kernel 사이 gap?

**2. Kernel 내부 (`ncu`)** — 처음엔 GEMM 전체 최적화 금지. 한 kernel만(LayerNorm / Softmax / RoPE).
측정: kernel latency, SM utilization, occupancy, DRAM throughput, L2 hit rate, register usage.

**컴퓨터구조와 연결:**

```text
"이 코드가 느리다"
  → compute-bound인가 memory-bound인가?
  → arithmetic intensity가 낮은가?
  → L2가 잡아주고 있는가, HBM까지 내려가는가?
```

이 사고방식으로 바뀌면 성공.

---

## Week 8 — 최종 프로젝트: KV Cache Memory Profiler

**주제:** KV-cache scaling behavior of autoregressive LLM inference
**질문:** Context가 길어질수록 실제 GPU inference는 왜 느려지는가?

context length `128 / 256 / 512 / 1024 / 2048 / 4096 / 8192` 각각:
time-to-first-token, decode latency/token, tokens/sec, GPU memory, KV cache size, GPU memory traffic, L2 hit rate.

모델이 커서 안 돌아가면 아주 작은 모델이어도 된다. **연구 질문이 중요하지 모델 크기가 중요한 게 아니다.**

**비교:**

```text
A. MHA   Q heads = 8, KV heads = 8
B. GQA   Q heads = 8, KV heads = 2
C. MQA   Q heads = 8, KV heads = 1

KV capacity → HBM traffic → decode latency ?
```

KV-cache/storage 및 memory-system 연구로 자연스럽게 이어진다.

---

## AI 사용 규칙

**금지 1 — 구현 전에 코드 생성시키기**

- 나쁜 예: "Multi-head attention 구현해줘."
- 좋은 예: "Multi-head attention을 직접 구현하려 한다. 필요한 tensor와 각 shape만 알려줘. 코드는 작성하지 마."

**금지 2 — 에러를 바로 붙이고 수정 요청.** 먼저 직접 쓴다:

```text
Observed:   loss가 NaN이 됨
Hypothesis: softmax 이전 score가 너무 큰 것 같다.
Checked:    learning rate 낮춤 → 그대로
            gradient norm → layer 3부터 폭증
```

그 다음: "내 가설을 평가하고 다음으로 확인할 실험 하나만 제안해줘."

**역할 ① 교수** (매주 종료 때)

> 이번 주에 내가 구현한 내용은 causal self-attention이다. 대학원 구두시험 수준으로 한 문제씩 질문해줘. 내가 답하기 전에는 정답을 알려주지 마. 내 답에서 정확한 부분, 잘못된 부분, 빠진 부분을 구분해서 평가해줘.

**역할 ② Reviewer** (git diff를 보여주고)

> 이 코드를 다시 작성하지 마. 1. conceptual correctness 2. tensor shape 3. numerical stability 4. performance — 네 종류의 문제만 찾아줘. 위치와 이유만. 수정 코드는 내가 요청할 때만.

**역할 ③ Test Engineer** — 적극 맡겨도 됨

> 현재 구현을 깨뜨릴 수 있는 테스트를 만들어줘. 구현 코드는 수정하지 말고 pytest만 작성해줘.

**역할 ④ Research Assistant** (Week 7~8부터 비중 ↑)

- 사람: hypothesis, experiment design, metric selection, 결과 해석
- Agent: 반복 실행, parameter sweep, CSV 기록, plot 생성, regression test, benchmark automation

> **판단은 내가 하고 반복 노동은 agent에게 준다.**

---

## 반드시 지킬 규칙 하나

각 주 마지막에 **reference 없이 빈 종이에 구조를 그린다.** 예: Week 5

```text
token → embedding → RMSNorm → Q K V → RoPE → KV cache → attention → residual
      → RMSNorm → SwiGLU → residual → ... → logits
```

이걸 못 그리면 **그 주를 끝낸 게 아니다.**

---

## 8주 뒤 기대 수준

```text
LLM  = tokenizer + matrix multiplication + normalization + attention
     + MLP + loss/backprop + optimizer + sampling

성능 = compute + memory traffic + kernel implementation + parallelism
```

그때부터 `KV cache`, `GQA`, `FlashAttention`, `PagedAttention`, `KV offloading` 같은 말을 봐도 별개의 신기술이 아니라 **"이 계산 graph/메모리 구조 중 어디를 바꾸는 기술인지"** 바로 위치를 잡을 수 있다.

**레퍼런스:** `microGPT → build-nanogpt → llama2.c → llm.c`
microGPT로 알고리즘을 벗겨보고, build-nanogpt로 실제 GPU 학습을 경험하고, llama2.c로 inference와 메모리를 C 수준까지 내린 뒤, llm.c로 training/CUDA까지 내려간다.
