# Week 1 — Language Model + Autograd

> Transformer는 아직 안 한다. `"emma"` → `e→m, m→m, m→a` — **next-token prediction 자체**를 이해한다.

## 직접 만들 파일

- [ ] `bigram.py` — `count(x_t, x_t+1)` → probability → sample, cross entropy까지
- [ ] `mlp.py`
- [ ] `value.py` — scalar autograd: `Value(data, grad)`, `__add__`, `__mul__`, `tanh`, `exp`, `backward`
- [ ] `nn.py`
- [ ] `train.py`
- [ ] `tests/`

## 필수 실험

- [ ] numerical gradient(finite difference) vs backprop — 오차가 충분히 작은가?

## 완료 조건 (코드 없이 답하기)

- [ ] language model은 무엇을 학습하는가?
- [ ] logits와 probability의 차이는?
- [ ] cross entropy가 감소한다는 건 무엇인가?
- [ ] computational graph란?
- [ ] backward pass가 실제로 하는 일은?
- [ ] parameter는 어떻게 업데이트되는가?
- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
