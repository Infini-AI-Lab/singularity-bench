# Fused bias + ReLU

Implement `kernel.py:bias_relu(x, bias)` with a Triton GPU kernel. `x` is a
contiguous CUDA tensor with shape `(rows, cols)` and `bias` is a contiguous CUDA
tensor with shape `(cols,)`, on the same device and with the same dtype (FP16 or
FP32). Return `max(x + bias, 0)` with the same shape, dtype, and device as `x`.
Do not modify the inputs. Support positive dimensions, including odd widths
and lengths that are not multiples of your launch block size.

Use PyTorch for allocation and the reference, and Triton for the fused operation.
Run the public checker and inspect its correctness and timing output:

```sh
python3 /app/examples/run.py check triton-bias-relu --workspace "$PWD"
```

Timing has no pass threshold. You may benchmark and optimize further if useful.
