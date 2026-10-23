# Exact ordered PV: rejected GPU prototype

The constant-spacing PV prototype preserves the original arithmetic in the tested cases, but loses every comparison against the frozen production Rust PV kernels. Do not retain or integrate it. All CPU screens, CUDA builds, sanitizer checks and event measurements run on `ssh wx-workstation`; production code and the original independent numerical gate remain unchanged. The [arithmetic argument](../../research/exact-ordered-pv.md) explains the guards, rounding ties and fallback.

## CPU feasibility

Two screens use the same twenty-nine probability/value inputs: seven sampled causal rows from a captured 513-token prefill, one captured six-token decode, and uniform/diffuse/concentrated synthetic probabilities at seven lengths from 129 through 32,768 keys. Values remain BF16 and products are rounded separately to FP32. Captured replays reproduce every stored raw FP32 and BF16 output. Synthetic probabilities use fixed seeds and a twenty-unit score-span cap; they are not long-context model captures.

The interval screen checks five outward-computed bounds per case. Every original result is enclosed and no certified output differs, but long-context acceptance is weak: at 32K, the bound using every exact-prefix approximation leaves 58.79%, 57.86% and 51.20% of uniform/diffuse/concentrated outputs requiring fallback. The ordinary weighted-absolute bound leaves 100%, 100% and 89.43% unresolved. Even this feasibility screen excludes the cost of constructing a fast center and certification metadata.

The constant-spacing method instead checks exact thirty-two-term blocks. Every accepted intermediate prefix and every completed sum matches the ordered FP32 replay bit for bit. Its 189 additional adversarial output cases cover positive/negative sums, halfway ties, spacing boundaries and zero/subnormal fallback. Accepted block fractions are 42.87% for the sampled final captured prefill row, 61.51% for diffuse 8K inputs, and 71.63%/75.22%/96.74% for uniform/diffuse/concentrated 32K inputs. These fractions measure block eligibility, not speed.

## CUDA checks and measurement

A standalone diagnostic implements the grid method and two simple CUDA ordered-sum controls. Four inputs are tested at one and thirty-two query rows; the larger cases repeat the same probability matrix and share one V sequence. All 540,672 output entries match the CPU raw-FP32 replay, and accepted-block totals match the CPU formulation. The initial operator screen shows a modest solo gain against the simpler controls but severe batched regressions. Those controls are not the production kernels, so their apparent solo gain does not qualify the candidate.

A second adapter loads the actual paged and grouped PV kernels from the frozen production CLI's embedded cubin. It pads score stride and physical KV pages, builds the corresponding row/page metadata and compares BF16-rounded outputs on identical inputs. The cubin's bytes are verified against the original CLI, not rebuilt or translated. All output comparisons pass.

Both adapters complete memcheck at captured 513-key/thirty-two-row and synthetic 32K/one-row shapes, plus racecheck at captured 513-key/one-row shape. All sanitizer processes exit zero with complete zero-error/zero-warning summaries. This does not claim racecheck at every length or row count, model sanitizer coverage, adversarial GPU coverage equivalent to the CPU screen, or a production Rust candidate.

Each timing process validates outputs before measuring, evicts at least twice the reported L2 capacity and at least 256 MiB outside the event interval, and runs variants in paged/grouped/grid/grid/grouped/paged order. Each arm excludes three warmups and records five event samples, yielding ten samples per variant. No build or other GPU job overlaps. The following are operator medians, not model/API latency or stock-engine rankings.

| Input | Query rows | Production paged | Production grouped | Grid prototype |
|---|---:|---:|---:|---:|
| Captured 513-key prefill row | 1 | 0.014336 ms | 0.016400 ms | 0.032768 ms |
| Captured 513-key prefill row | 32 | 0.040960 ms | 0.022496 ms | 0.562176 ms |
| Synthetic diffuse, 8K | 1 | 0.143360 ms | 0.196608 ms | 0.385024 ms |
| Synthetic diffuse, 8K | 32 | 0.525312 ms | 0.239616 ms | 7.540224 ms |
| Synthetic uniform, 32K | 1 | 0.536576 ms | 0.760592 ms | 1.464320 ms |
| Synthetic uniform, 32K | 32 | 2.129232 ms | 0.912224 ms | 28.901376 ms |
| Synthetic concentrated, 32K | 1 | 0.536576 ms | 0.757760 ms | 1.226752 ms |
| Synthetic concentrated, 32K | 32 | 2.156384 ms | 0.954368 ms | 23.760752 ms |

The prototype is about 2.3–2.7 times slower than the faster production kernel for solo cases and 25–32 times slower for thirty-two rows. High block eligibility does not compensate for this implementation's thread, scan, staging and fallback costs. It remains an isolated mathematical/diagnostic result, with no model integration or retained dispatch.

## Artifacts and audit

Raw sources, executables, inputs and results remain under `clean-work/pro6000-baseline/` on the workstation. The final audit rechecks all five campaigns' source identities, completed validation receipts, output hashes, sanitizer summaries, medians, the earlier captured-input audits and production-cubin provenance. All runners are terminal and no GPU compute process remains at completion.

| Artifact | SHA-256 |
|---|---|
| `pv-interval-screen-v1/summary.json` | `6c98075d14e8c4d03145146292cf6f9405945609dffd03e84b68cbe00b9b7b5d` |
| `pv-grid-screen-v1/summary.json` | `38ee79e4368532f660e46b7b90f1aac3dcc0c15acf501b96f2ae7f453bd15b67` |
| `pv-grid-gpu-v1/summary.json` | `ce7945c925a2a8310a7cde55b3ef1ba8f3db057cfe69f9eddb2185f1015d8d31` |
| `pv-grid-gpu-v1/validation-v1/summary.json` | `d3a96725db602f2af79ad1a8ecd9e578e7e0abeccbeaec5565a58b8973647994` |
| `pv-grid-production-v1/summary.json` | `6c632701a45945ce49032bef64e1484df7c9cfa829e5db55bb3646a3c9ddd77a` |
| Production-comparison source | `dd1d9597b68d5c4f2ff3e0759bca9dd4138a6951bfacddb88a06e24b4e193554` |
| Production-comparison executable | `1cbf94046130e9b9e2c482349035b36e96903051dad70bc3fe792f091b475de4` |
| Frozen production PV cubin | `dd2990ccdad45c1ce7c9adc97e558700a6f421523d0d8ac577e8c647ac1cff65` |
| `pv-grid-production-v1/completion-audit.json` | `2e35baa3642977195ff1a7562012af020ea8a34479c592b31604adfc2093609b` |
