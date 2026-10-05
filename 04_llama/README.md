# Week 4 — GPT에서 Llama로 진화

| GPT | Llama |
|---|---|
| LayerNorm | RMSNorm |
| GELU | SwiGLU |
| learned positional embedding | RoPE |
| MHA | GQA / MQA |
| Linear with bias | bias 없는 Linear |

## 하나씩 바꾸며 측정

- [ ] Baseline GPT
- [ ] + RMSNorm → loss / speed
- [ ] + RoPE → loss
- [ ] + SwiGLU → parameters / loss
- [ ] + GQA → KV memory

## 완료 조건

- [ ] LayerNorm vs RMSNorm
- [ ] absolute position vs RoPE
- [ ] GELU vs SwiGLU
- [ ] MHA vs GQA vs MQA
- [ ] **GQA가 왜 KV-cache 크기를 줄이는가?**
- [ ] `notes/llama_vs_gpt.md` 작성
- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
