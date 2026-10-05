# Week 5 — Python을 버리고 C로 inference

```text
Python model → checkpoint → binary file → C → malloc() → matmul() → attention() → token
```

> `run.c`를 처음부터 읽지 말 것. 직접 skeleton을 만들고, 막힌 뒤에 본다.

## 직접 만들 것

- [ ] `Config` struct (`dim, hidden_dim, n_layers, n_heads, ...`)
- [ ] `load_weights()`
- [ ] `rmsnorm()`
- [ ] `matmul()`
- [ ] `rope()`
- [ ] `softmax()`
- [ ] `attention()`
- [ ] `forward()`

## KV cache (최중요)

```text
key_cache[layer][token][kv_head][head_dim]
value_cache[layer][token][kv_head][head_dim]

new token → Q_new, K_new / V_new → cache에 append → 기존 K/V 읽기 → attention
```

- [ ] 직접 계산: `layers=32, kv_heads=8, head_dim=128, seq=8192, FP16` → KV cache 몇 GB?
      (공식 암기 말고 **array layout에서 byte 수 유도**)

## 완료 조건

- [ ] reference 없이 빈 종이에 구조 그리기
      (token → embedding → RMSNorm → QKV → RoPE → KV cache → attention → residual → RMSNorm → SwiGLU → residual → … → logits)

## 기록
