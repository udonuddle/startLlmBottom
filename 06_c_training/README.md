# Week 6 — `llm.c`: backward를 C에서 이해

> 첫 목표는 CUDA가 아니다. `train_gpt2.py` vs `train_gpt2.c` 비교.

## 추적할 것

Linear `Y = XW`에서 `dX`, `dW`가 실제 C array에 어떻게 계산되는지.

forward / backward pair 정리:

- [ ] Embedding
- [ ] LayerNorm
- [ ] Attention
- [ ] MLP
- [ ] Cross entropy

## 산출물

- [ ] `notes/backprop.md`

## 완료 조건

- [ ] `loss.backward()` 없이 "Transformer parameter의 gradient는 어디서 오는가?"를 큰 흐름으로 설명
- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
