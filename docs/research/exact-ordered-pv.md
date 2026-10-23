# Exact ordered attention-value sums

This experiment targets the existing paged PV operation: round each probability/value product to FP32, add the products in key order with separate FP32 rounding, then round the result to BF16. It changes neither QK nor softmax. A guarded integer formulation reproduces the original sum exactly in selected blocks, but its current GPU implementation loses to the production kernels and is not retained. [Results and validation boundaries](../qwen3-8b-w8a8/results/pro6000-exact-pv.md) are recorded separately. All executable work runs on `ssh wx-workstation`.

## Why a final error interval was insufficient

An initial CPU screen bounds the original ordered sum around an FP64 center using weighted absolute products, absolute intermediate prefixes, or bounds on prefixes within blocks. All accepted outputs agree, and all original sums lie inside their computed intervals. Nevertheless, even the bound using every intermediate prefix leaves approximately 51–59% of synthetic 32K outputs unresolved. The block bounds additionally need prefix and absolute-product work. This does not rule out every interval formulation, but it provides no clear route to a cheap, broadly accepted replacement for long-context PV.

## Constant-spacing blocks

Let the original rounded products be `a_j`, and let a block start with the exact FP32 carry `s`. For a finite nonzero carry with `2^e <= abs(s) < 2^(e+1)`, FP32 spacing within that signed interval is `h = 2^(e-23)`. Write `s = n_0 h`, where `n_0` is an integer. If every rounded intermediate sum remains strictly inside this interval, the original FP32 additions equal rounding on the integer grid:

```text
n_j = round_to_nearest_even(n_(j-1) + a_j/h)
s_j = n_j h
```

Strict interior matters at powers of two, where the spacing on the other side changes. The candidate checks every intermediate integer prefix, not only the final sum. If a prefix touches or crosses either interval endpoint, the complete block uses the original serial addition loop. This also handles blocks that cross a boundary and later return.

For each product, compute `b_j = round_to_nearest_even(a_j/h)`. Away from halfway cases, the increment is exactly `b_j`, independent of the carry. At a halfway case, `b_j` is even. An even incoming integer needs no correction; an odd incoming integer needs `sign(a_j/h - b_j)`. The result of every halfway addition is even. Therefore the incoming parity at any halfway position is determined by the parity of ordinary increments since the most recent earlier halfway position, or by the initial carry parity if no earlier halfway position exists.

The GPU prototype assigns a warp to one output coordinate. Two ballots record halfway positions and odd ordinary increments. Each lane finds the most recent earlier halfway position and uses a population count to compute its incoming parity. After the correction, a five-step integer prefix scan produces all thirty-two candidate intermediate sums. A warp-wide predicate accepts only when every prefix lies strictly inside the original signed spacing interval. Negative sums obey the same integer parity rule.

## Guards and exactness boundary

The prototype admits carry exponents from −103 through 126, so both `h` and its reciprocal are representable normal FP32 powers of two. A normal, finite scaled product is exact under this power-of-two multiplication. If scaling underflows, its magnitude is far below one half and its nearest integer is still zero. Nonfinite scaled products or magnitudes above `2^24` reject the block. Guarded increments and thirty-two-term prefix sums fit signed 32-bit integers. Accepted prefixes have at most twenty-four significant bits, so conversion back to FP32 and multiplication by `h` are exact.

If the integer-grid rounded value is strictly interior, its unrounded argument lies within one half grid step of an interior FP32 value. It therefore cannot encounter the different spacing outside the interval. Induction from the exact input carry establishes equality with every original intermediate addition. The final accepted carry is bit-identical; no numerical tolerance is involved. Rejected blocks reuse their already rounded products in the original serial loop. Zero, subnormal carries outside the guard, unsupported exponents and interval crossings take that fallback.

This argument concerns the arithmetic transformation. Memory layout, launch ownership, synchronization, compiler realization and performance require separate validation. The standalone CUDA probe uses a padded shared tile, 1,024 threads per block and 4,352 shared bytes, with no stack or spills. The raw-output build uses 26 registers per thread; the BF16-output production-comparison build uses 24. It passes the recorded exact comparisons and scoped sanitizers, but its extra per-coordinate work overwhelms the saved serial dependency on the production comparison. No Rust kernel, host dispatch or production arithmetic change follows from this prototype.
