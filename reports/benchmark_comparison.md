# VoiceOps 2026 - Benchmark Evaluation Summary
Generated at: 2026-10-25 15:14:03 UTC

| Evaluation Phase | Total Samples | Must-Include Acc | Must-Not-Include Acc | Decision Acc | Overall Pass Rate | Avg Latency |
|---|---|---|---|---|---|---|
| **Step 1: Baseline** | 5 | 20.0% | 100.0% | 100.0% | 20.0% | 919.4 ms |
| **Step 2: Grounded RAG** | 5 | 80.0% | 100.0% | 100.0% | 80.0% | 2098.3 ms |
| **Step 2: RAG (ASR Robustness)** | 5 | 80.0% | 100.0% | 100.0% | 80.0% | 3470.2 ms |
| **Step 3: Multi-turn Convs** | 22 turns | 100.0% | 100.0% | 95.45% | 95.45% | 3162.6 ms |