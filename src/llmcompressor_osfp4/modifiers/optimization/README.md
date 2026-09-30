# OSFP4 optimization internals

This private package implements `optimize_rtn()` and `optimize_sic()`.
`OSFP4Modifier` owns calibration, smoothing, deployment, and checkpoint state.

For weight shape `[R, C]`, with `G = C / 16`:

| Tensor | Shape |
| --- | --- |
| Weight groups | `[G, R, 16]` |
| Activation groups | `[G, 16, S]` |
| Hessian/Cholesky | `[C, C]` |
| Final weight scales | `[R, G]` |

All 16-column groups in a mapping are optimized together with one Adam
optimizer. Their parameters and losses remain independent. The first target
layer's attached OSFP4 observer performs continuous optimization and E4M3 scale
selection.

RTN uses `diag(X.T @ X) / sample_count` for continuous optimization and final
scale selection. It installs the selected qparams while retaining the balanced
floating-point weights.

SIC dampens `X.T @ X / sample_count`, computes upper Cholesky factor `U`, and
uses `diag(U)²` as its metric. After continuous optimization, it initializes
`Y = U @ W.T` and processes 16-column blocks from right to left. At each block,
it forms the effective target
`T_eff = ((Y_local - unwanted_cross_terms) / U_diag_block[:, None]).T`, where
`unwanted_cross_terms = U_upper_strict @ W_local.T`. Selection uses metric
`U_diag_block²` and the actual FP4 quantizer, and includes
feedback from completed right-hand blocks; within-block feedback is approximated
by retaining the original unprocessed weights. Candidate scales and output rows
are searched together within the block. The block's scales are then held fixed
while its columns are quantized from right to left using live `Y`. Each rounded
reconstruction is propagated through `U` before selecting the next block's scales.

For NVFP4, activation subsampling limits only the explicit activation loss.
The full cache still supplies statistics and input-scale replay. Activation
assembly allocates one FP32 destination and converts one batch at a time.
E4M3 selection gathers alpha and metric rows only for its current tile. There
is no configurable quantization-group batching. NVFP4A16 does not assemble
activation groups.
