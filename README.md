# startLlmBottom — LLM from Zero

> 8주 뒤, LLM을 **API 사용자 수준이 아니라** 학습·추론·메모리·CUDA까지 설명할 수 있는 상태.

**원칙:** 내가 먼저 구현 → 테스트 실패 → 원인 가설 → AI 도움 → reference 확인
reference 코드를 먼저 베끼지 않는다. 전체 계획은 [ROADMAP.md](ROADMAP.md).

---

## 진행 현황

| Week | 주제 | 폴더 | 상태 |
|---|---|---|---|
| 1 | Language Model + Autograd | [01_autograd](01_autograd/) | ⬜ |
| 2 | Attention + GPT | [02_microgpt](02_microgpt/) | ⬜ |
| 3 | PyTorch + GPU Training | [03_torch_gpt](03_torch_gpt/) | ⬜ |
| 4 | Llama Architecture | [04_llama](04_llama/) | ⬜ |
| 5 | C Inference + KV Cache | [05_c_inference](05_c_inference/) | ⬜ |
| 6 | C Training + Backprop | [06_c_training](06_c_training/) | ⬜ |
| 7 | CUDA + Profiling | [07_cuda](07_cuda/) | ⬜ |
| 8 | KV-cache Profiler (Capstone) | [08_kv_profiler](08_kv_profiler/) | ⬜ |

### I implemented:

- [ ] scalar autograd
- [ ] autoregressive LM
- [ ] multi-head attention
- [ ] GPT
- [ ] GPU training
- [ ] RoPE / RMSNorm / SwiGLU / GQA
- [ ] Llama inference in C
- [ ] KV cache
- [ ] GPT backward in C
- [ ] CUDA profiling
- [ ] KV-cache memory benchmark

---

## 구조

```text
01_autograd/ ~ 08_kv_profiler/   주차별 구현 (각 폴더 README = 체크리스트)
notes/                           attention.md, backprop.md, llama_vs_gpt.md, gpu_memory.md
experiments/results.csv          모든 실험 기록 (machine 열에 어느 컴퓨터인지 기록)
experiments/plots/               그래프
data/                            큰 데이터셋 (gitignore — 컴퓨터마다 따로 다운로드)
references/                      레퍼런스 repo clone (gitignore)
```

---

## 두 컴퓨터(집 ↔ 학교)에서 작업하기

### 처음 한 번 (새 컴퓨터에서)

```bash
git clone https://github.com/udonuddle/startLlmBottom.git
cd startLlmBottom
git config pull.rebase true          # pull 시 merge commit 대신 rebase

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Week 3부터: 그 컴퓨터의 GPU에 맞는 PyTorch 설치
```

### 매번

```bash
# 작업 시작할 때
git pull

# 작업 끝낼 때 (반드시 push하고 자리 뜨기)
git add -A
git commit -m "week1: Value.backward 구현"
git push
```

- 다른 컴퓨터에서 push를 잊었다면, 이쪽에서 작업 전에 반드시 확인하기. 같은 파일을 양쪽에서 고치면 충돌 난다.
- 체크포인트(`*.pt`, `*.bin`), 데이터셋(`data/`), 프로파일 결과(`*.nsys-rep`, `*.ncu-rep`)는 커밋되지 않는다. 결과는 **숫자를 `results.csv`에, 그림을 `plots/`에** 남긴다.
- 작은 데이터 파일(예: `names.txt`)은 해당 주차 폴더에 두고 커밋해도 된다.

### 컴퓨터별 환경

| 컴퓨터 | GPU | CUDA | 할 수 있는 주차 |
|---|---|---|---|
| (이 컴퓨터) | AMD Radeon RX 470/580급 | ❌ | 1, 2, 4, 5, 6 (CPU), 3 (CPU로 작게) |
| (다른 컴퓨터) | ? | ? | Week 3, 7, 8은 NVIDIA GPU 필요 (`nsys`, `ncu`) |

---

## 레퍼런스

`microGPT → build-nanogpt → llama2.c → llm.c` 순서. **구현해 보고 막힌 뒤에만** 열어 본다.

| Week | Reference |
|---|---|
| 1–2 | [microGPT](https://github.com/karpathy/karpathy.github.io/blob/master/_posts/2026-02-12-microgpt.markdown) |
| 3 | [build-nanogpt](https://github.com/karpathy/build-nanogpt) |
| 4–5 | [llama2.c](https://github.com/karpathy/llama2.c) |
| 6–7 | [llm.c](https://github.com/karpathy/llm.c) |

```bash
# 로컬에서만 볼 용도 (references/는 gitignore)
mkdir -p references && cd references
git clone https://github.com/karpathy/build-nanogpt.git
git clone https://github.com/karpathy/llama2.c.git
git clone https://github.com/karpathy/llm.c.git
```

---

## 매주 고정 루틴

```text
① 개념 공부        1~2h
② 직접 구현        3~4h
③ 테스트/실험      2h
④ reference 비교   1~2h   ← ②보다 먼저 하지 않기
⑤ AI 구두시험      30m
⑥ README 기록      30m
```

각 주 마지막: **reference 없이 빈 종이에 구조 그리기.** 못 그리면 그 주를 끝낸 게 아니다.
