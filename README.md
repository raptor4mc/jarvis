# jarvis
A actuall jarvis

## Training improvements
2. **Batch your tokens**

You are training with:
- Kod
- micro-batch size: 32
- grad_accum_steps: 1

But your model likely still processes each token sequentially.

You want:
- full sequence matmul
- full batch matmul
- fused attention operations
