# Repair the CUDA row reduction

The package's `reduction.py:row_sum(x)` should return one sum per row of a
contiguous FP32 CUDA matrix, without modifying its input. It uses a locally
compiled C++/CUDA extension. It currently has a build error, followed by a
correctness bug affecting widths that are not multiples of its block size.

Fix both defects, preserving the Python API and using the CUDA extension for
the reduction. Keep the extension sources in this repository; do not replace
the operation with a PyTorch reduction. All build dependencies are installed.

```sh
python3 /app/examples/run.py check cuda-reduction --workspace "$PWD"
```

The checker builds the extension, compares GPU results with a PyTorch reference,
and prints diagnostic timings. Its first build can take longer than later runs.
