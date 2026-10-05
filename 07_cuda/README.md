# Week 7 — CUDA + Profiler

> 순서: PyTorch profiler → Nsight Systems → Nsight Compute. (NVIDIA GPU 컴퓨터에서)

## 1. 전체 timeline (`nsys`)

- [ ] GPU가 언제 놀고 있나?
- [ ] CPU launch overhead가 있나?
- [ ] Memcpy가 있나?
- [ ] kernel 사이 gap은?

## 2. Kernel 내부 (`ncu`) — 한 kernel만 고른다 (LayerNorm / Softmax / RoPE)

- [ ] kernel latency
- [ ] SM utilization
- [ ] occupancy
- [ ] DRAM throughput
- [ ] L2 hit rate
- [ ] register usage

## 사고방식 전환

"느리다" → compute-bound인가 memory-bound인가? → arithmetic intensity가 낮은가? → L2가 잡아주나, HBM까지 내려가나?

## 완료 조건

- [ ] `notes/gpu_memory.md` 작성
- [ ] reference 없이 빈 종이에 구조 그리기

## 기록
