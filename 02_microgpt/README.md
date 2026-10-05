# Week 2 — Attention을 직접 만들어 GPT 완성

```text
Token embedding + Position embedding → X → Q K V → Attention → residual → MLP → residual → logits
```

## 직접 만들 파일

- [ ] `embedding.py`
- [ ] `attention.py`
- [ ] `mlp.py`
- [ ] `norm.py`
- [ ] `block.py`
- [ ] `gpt.py`
- [ ] `generate.py`

## 손계산 (NumPy, PyTorch 금지)

- [ ] token 3개, hidden 4: `Q=XWq, K=XWk, V=XWv` → `QKᵀ` → causal mask → softmax → V weighted sum

## 실험

- [ ] A. causal mask 제거 → training loss가 왜 급감하고, 왜 문제인가?
- [ ] B. position embedding 제거
- [ ] C. 1 head → 4 head
- [ ] D. temperature 0.2 / 0.7 / 1.0 / 1.5 생성 결과 비교

## 완료 조건

- [ ] Q / K / V를 화이트보드에 설명 ("무엇을 찾나 / 나는 어떤 정보인가 / 내 실제 내용은")
- [ ] causal attention이 왜 필요한가?
- [ ] attention matrix는 왜 `T×T`인가?
- [ ] context가 2배면 어떤 비용이 커지는가?
- [ ] residual connection은 왜 있는가?
- [ ] multi-head는 왜 쓰는가?
- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
