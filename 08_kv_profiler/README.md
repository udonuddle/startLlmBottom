# Week 8 — Capstone: KV-cache scaling behavior of autoregressive LLM inference

> **질문:** Context가 길어질수록 실제 GPU inference는 왜 느려지는가?
> 모델 크기보다 연구 질문이 중요하다. 작은 모델이어도 된다.

## 실험

context length: `128, 256, 512, 1024, 2048, 4096, 8192`

각각 측정:

- [ ] time-to-first-token
- [ ] decode latency / token
- [ ] tokens/sec
- [ ] GPU memory
- [ ] KV cache size
- [ ] GPU memory traffic
- [ ] L2 hit rate

## 비교

| | Q heads | KV heads |
|---|---|---|
| A. MHA | 8 | 8 |
| B. GQA | 8 | 2 |
| C. MQA | 8 | 1 |

```text
KV capacity → HBM traffic → decode latency ?
```

## 역할 분담

- 내가: hypothesis, experiment design, metric selection, 결과 해석
- Agent: 반복 실행, parameter sweep, CSV 기록, plot 생성, regression test, benchmark 자동화

## 기록
