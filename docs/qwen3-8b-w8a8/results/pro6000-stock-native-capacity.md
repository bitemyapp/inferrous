# RTX PRO 6000 stock native-capacity checks

All eight stock configurations planned for the final comparison complete the five selected native-capacity workloads at a 64 GiB KV allocation on `ssh wx-workstation`. The audited campaign contains forty complete workload records, 712 requests and all 2,368 requested output tokens. This establishes completion for these workloads and settings. It does not establish numerical equivalence, fastest-engine rankings, full-pool saturation or completion of the final matrix.

## Configuration and workloads

The campaign preserves the frozen final plan's pinned vLLM/B12X images, original W8A8 weights with FP32 scales, BF16 model/KV configuration, explicit Triton INT8 linear backend, 32,768-token context, thirty-two sequence slots and 68,719,476,736-byte KV budget. FlashInfer, FlashAttention, Triton attention and B12X each run with graph-only and compiled execution. B12X retains its planned 128-token blocks and supported `--disable-cutedsl-warmup` option. Prefix caching is disabled, and each process uses within-request KV reuse.

Each candidate runs one complete resident suite with one invocation per workload and no excluded warmups. GPU jobs are serial, with no concurrent external builds. Requests ignore EOS during generation and return exactly their specified output lengths. The five workloads are copied unchanged from the ninety-nine-scenario final matrix.

| Workload | Requests | Prompt lengths | Requested outputs | Total output tokens per candidate |
|---|---:|---|---|---:|
| Short batch | 32 | 129 each | 2 each | 64 |
| Native solo | 1 | 32,766 | 2 | 2 |
| Native batch sixteen | 16 | 32,766 each | 2 each | 32 |
| Native batch thirty-two | 32 | 32,766 each | 2 each | 64 |
| Native ragged/sustained | 8 | 5, 129, 512, 2,048, 4,096, 8,192, 16,384, 32,736 | 32, 24, 16, 12, 8, 6, 4, 32 | 134 |

The native uniform requests reserve two output tokens inside the 32,768-token total limit. The sixteen/thirty-two-request workloads exceed simultaneous residency at this KV budget. Successful completion demonstrates scheduling of the oversized request set, not that every request resides concurrently or that the allocator reaches saturation. Logs show that the stock scheduler can keep only a few long requests active while others wait. This campaign does not claim forced eviction/preemption coverage.

## Completed configurations

Every row below passes all five workloads and returns 296 output tokens across eighty-nine requests. All eight engine logs confirm a 64.0 GiB KV reservation and report maximum native-context concurrency rounded to 14.22. B12X's different block size yields a slightly smaller cached-token capacity under the same byte budget.

| Attention and execution | Logged cached-token capacity | Observed device memory peak, MiB | Telemetry samples | Result |
|---|---:|---:|---:|---|
| FlashInfer graph | 466,032 | 79,101 | 310 | Passed |
| FlashInfer compiled | 466,032 | 79,383 | 321 | Passed |
| FlashAttention graph | 466,032 | 77,853 | 337 | Passed |
| FlashAttention compiled | 466,032 | 78,013 | 348 | Passed |
| Triton attention graph | 466,032 | 77,855 | 575 | Passed |
| Triton attention compiled | 466,032 | 78,085 | 586 | Passed |
| B12X graph | 465,920 | 78,167 | 356 | Passed |
| B12X compiled | 465,920 | 78,347 | 368 | Passed |

Memory is sampled device-wide every 500 ms and includes desktop/context allocations; it is not a per-engine allocation measurement or a continuous peak guarantee. Single-invocation durations include workload-dependent first-use effects and are not reported as benchmark rankings. These records cannot be substituted for the final matrix's excluded warmups and twenty measured samples per candidate/scenario.

## Audit and remaining scope

The unchanged maintained executor validates input/image identities, command bindings, configuration, complete ordered records, successful process exit, telemetry and artifact hashes. The completion audit revalidates all eight suites and every recorded artifact, checks each request's output count and token range, confirms native-context bounds, and verifies the actual KV reservation/capacity log entries. All eight processes and the campaign runner exit zero. No failed or partial suite is counted as a pass.

Cold within-request reuse is the only cache mode exercised here. Full recomputation, warm-prefix behavior, every remaining scenario and paired performance require the complete matrix. The production engine's independent numerical gate remains unresolved. This stock-only campaign changes neither production arithmetic nor the frozen final-matrix plan, and does not waive their acceptance requirements.

Raw artifacts remain on the workstation under `clean-work/pro6000-baseline/stock-native-capacity-v1/`. Summary SHA-256 is `35146c6af75b63ccfc43044f6df08bf0c946e383bd837598e07885a65b8c4d10`, completion audit source `e7d2775769b709987ce217b90796deed0d9fbe2fa9018e8809b7a43b1f70f86c`, execution receipt `2eef825e633d0705a05f098f3c18a8c2f07398c663c9b763b2f678136852cc52`, plan `b11503a310c63eaa132e84d97469eef9db30bf5a64c5877531d95e2c82fe5052`, workloads `0fd3db256303c33f5d112a3a536ba5b44c64cea1bdaa31ffce0a48efce00d578`, and executor `17f7587951d66bfe7d9d81ba9a06177a512432feb9b04cf312e6bef0c2e4d768`. The summary pins all eight result receipts and output files, whose receipts pin the remaining raw artifacts. The parent final plan remains `d3c872005269aa11dceb80c0a8bcd4f2f7d02e35cfb2e5b5dc30d731e1f13c49`.
