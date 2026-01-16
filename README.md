
Separation of Concerns:
src/env/          → "What is the problem?"
src/agent/        → "Who solves it?"
src/training/     → "How do we learn?"
src/evaluation/   → "How well did we do?"
src/utils/        → "What tools help us?"
scripts/          → "How do we run this?"
assets/           → "What did we produce?"

Challenge: Environment Performance on Windows
During development, we observed significantly lower simulation throughput (≈12–15 FPS) compared to commonly reported benchmarks. Profiling revealed that collision detection and local coordinate transformations dominate runtime due to O(n²) neighbor computations in highway-env. While reducing vehicle count partially mitigates the issue, the limitation is intrinsic to the environment’s Python implementation on Windows. Training was therefore conducted with longer wall-clock time, which does not affect policy correctness.