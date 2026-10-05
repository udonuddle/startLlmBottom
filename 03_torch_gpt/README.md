# Week 3 — PyTorch + GPU에서 진짜 학습

> 직접 만든 알고리즘을 PyTorch tensor로 다시 만든다. CPU → GPU. (NVIDIA GPU 컴퓨터에서)

**금지:** `nn.Transformer`, `nn.MultiheadAttention`, HuggingFace GPT model
**허용:** `nn.Linear`, `nn.Embedding`, tensor matmul, autograd, AdamW

## 직접 만들 파일

- [ ] `model.py`
- [ ] `dataset.py`
- [ ] `train.py`
- [ ] `generate.py`
- [ ] `benchmark.py`

## 학습 설정 (처음엔 작게)

`d_model=256, layers=4, heads=4, context=256` — TinyStories 또는 작은 corpus

## 매 실험 기록 → `experiments/results.csv`

parameters, batch size, context, train loss, val loss, GPU memory, tokens/sec

## Profiler

- [ ] `torch.profiler`로 matmul / attention / MLP / LayerNorm / optimizer 중 어디에 시간이 쓰이는지

## 완료 조건

- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
