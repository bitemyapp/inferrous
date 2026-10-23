//! Ordinary floating-point Qwen operators and symmetric INT8 projections.
#![allow(dead_code, unused_imports, clippy::too_many_arguments)]
use cuda_device::dotprod::dp4a_s32;
use cuda_device::float::{
    add_rn_f32 as add, div_rn_f32 as div, ex2_approx_f32, mul_rn_f32 as mul,
    rsqrt_approx_f32 as rsqrt,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::cuda_module;

#[inline(always)]
fn bf(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff0000)
}
#[inline(always)]
fn widen(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
#[inline(always)]
fn exp(x: f32) -> f32 {
    ex2_approx_f32(mul(x, core::f32::consts::LOG2_E))
}
#[inline(always)]
fn round(x: f32) -> i8 {
    let x = x.clamp(-127.0, 127.0);
    let result: i32;
    // Same nearest-even conversion, now one native instruction. A converted
    // NaN has low byte zero, matching the previous saturating Rust cast.
    unsafe {
        cuda_device::ptx_asm!("cvt.rni.s32.f32 %0, %1;", out("=r") result, in("f") x);
    }
    result as i8
}

// Explicit integer PTX avoids this backend revision incorrectly tagging the
// integer redux intrinsics as SM100-only floating-point ReduxF32.
// Every caller supplies a complete converged warp; SM120 supports integer redux.
#[inline(always)]
fn warp_sum(value: i32) -> i32 {
    let result: u32;
    unsafe {
        cuda_device::ptx_asm!("redux.sync.add.u32 %0, %1, 0xffffffff;", out("=r") result, in("r") value);
    }
    result as i32
}
#[inline(always)]
fn warp_max(value: u32) -> u32 {
    let result: u32;
    unsafe {
        cuda_device::ptx_asm!("redux.sync.max.u32 %0, %1, 0xffffffff;", out("=r") result, in("r") value);
    }
    result
}

/// Positive maximum over a full 1024-thread block. Shared storage needs 32 slots.
#[inline(always)]
fn max_1024(mut m: f32, p: *mut f32, t: usize) -> f32 {
    m = f32::from_bits(warp_max(m.to_bits()));
    if t % 32 == 0 {
        unsafe {
            *p.add(t / 32) = m;
        }
    }
    thread::sync_threads();
    if t < 32 {
        let mut m = unsafe { *p.add(t) };
        m = f32::from_bits(warp_max(m.to_bits()));
        if t == 0 {
            unsafe {
                *p = m;
            }
        }
    }
    thread::sync_threads();
    unsafe { *p }
}

#[inline(always)]
fn stage_values(
    tile: *mut u32,
    probability: *mut f32,
    scores: &[f32],
    v: &[u16],
    query: usize,
    kv: usize,
    column: usize,
    t: usize,
    base: usize,
    end: usize,
) {
    let slot = base / 64 % 2;
    // SAFETY: a full 32-thread CTA owns this distinct staging slot. All bases
    // are 64-aligned within the 4096-token cache. The consumer waits and fences.
    unsafe {
        if base + t < end {
            *probability.add(slot * 64 + t) = scores[query * 4096 + base + t];
        }
        if base + t + 32 < end {
            *probability.add(slot * 64 + t + 32) = scores[query * 4096 + base + t + 32];
        }
        let mut piece = 0;
        while piece < 8 {
            let i = t + piece * 32;
            cuda_device::async_copy::cp_async_ca_16(
                tile.add(slot * 1024 + i * 4),
                v.as_ptr()
                    .add(((base + i / 4) * 8 + kv) * 128 + column + i % 4 * 8)
                    .cast::<u32>(),
            );
            piece += 1;
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

#[inline(always)]
fn stage_values_paged(
    tile: *mut u32,
    probability: *mut f32,
    scores: &[f32],
    v: &[u16],
    query: usize,
    stride: usize,
    page: usize,
    kv: usize,
    column: usize,
    t: usize,
    base: usize,
    end: usize,
) {
    let slot = base / 64 % 2;
    // A physical page contains a complete 64-token tile even at a causal tail.
    // Two shared slots let the next page load overlap ordered accumulation.
    unsafe {
        if base + t < end {
            *probability.add(slot * 64 + t) = scores[query * stride + base + t];
        }
        if base + t + 32 < end {
            *probability.add(slot * 64 + t + 32) = scores[query * stride + base + t + 32];
        }
        let mut piece = 0;
        while piece < 8 {
            let i = t + piece * 32;
            cuda_device::async_copy::cp_async_ca_16(
                tile.add(slot * 1024 + i * 4),
                v.as_ptr()
                    .add((page * 64 + i / 4) * 1024 + kv * 128 + column + i % 4 * 8)
                    .cast::<u32>(),
            );
            piece += 1;
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

#[inline(always)]
fn stage_values_grouped(
    tile: *mut u32,
    probability: *mut f32,
    scores: &[f32],
    v: &[u16],
    query: usize,
    stride: usize,
    page: usize,
    kv: usize,
    column: usize,
    t: usize,
    base: usize,
    end: usize,
) {
    let slot = base / 64 % 2;
    // One complete warp owns four probability rows and a shared 64x32 V tile.
    // Every head is in the same GQA group, hence has the same V coordinates.
    unsafe {
        let mut head = 0;
        while head < 4 {
            if base + t < end {
                *probability.add(slot * 256 + head * 64 + t) =
                    scores[(query + head) * stride + base + t];
            }
            if base + t + 32 < end {
                *probability.add(slot * 256 + head * 64 + t + 32) =
                    scores[(query + head) * stride + base + t + 32];
            }
            head += 1;
        }
        let mut piece = 0;
        while piece < 8 {
            let i = t + piece * 32;
            cuda_device::async_copy::cp_async_ca_16(
                tile.add(slot * 1024 + i * 4),
                v.as_ptr()
                    .add((page * 64 + i / 4) * 1024 + kv * 128 + column + i % 4 * 8)
                    .cast::<u32>(),
            );
            piece += 1;
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

#[inline(always)]
fn head_add_four(x: &[f32], starts: [usize; 4], sums: [f32; 4], j: usize, weight: f32) -> [f32; 4] {
    [
        add(sums[0], mul(x[starts[0] + j], weight)),
        add(sums[1], mul(x[starts[1] + j], weight)),
        add(sums[2], mul(x[starts[2] + j], weight)),
        add(sums[3], mul(x[starts[3] + j], weight)),
    ]
}

#[inline(always)]
fn head_reduce_four(mut sums: [f32; 4]) -> [f32; 4] {
    let mut d = 16;
    while d > 0 {
        sums = [
            add(
                sums[0],
                f32::from_bits(warp::shuffle_down(sums[0].to_bits(), d)),
            ),
            add(
                sums[1],
                f32::from_bits(warp::shuffle_down(sums[1].to_bits(), d)),
            ),
            add(
                sums[2],
                f32::from_bits(warp::shuffle_down(sums[2].to_bits(), d)),
            ),
            add(
                sums[3],
                f32::from_bits(warp::shuffle_down(sums[3].to_bits(), d)),
            ),
        ];
        d /= 2;
    }
    sums
}

/// Const specialization keeps the hot accumulators in registers for both widths.
#[inline(always)]
fn head_packed_group<const EIGHT: bool>(
    x: &[f32],
    output_rows: &[u32],
    words: &[u32],
    offsets: &[u32],
    exceptions: &[u16],
    mut out: DisjointSlice<f32>,
) {
    dependency_wait();
    let tid = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let index = tid / 32;
    let width = if EIGHT { 8 } else { 4 };
    let groups = output_rows.len().div_ceil(width);
    let batch = index % groups * width;
    let row = index / groups;
    if row >= 151936 {
        return;
    }
    let last = output_rows.len() - 1;
    // Padded members evaluate a valid row but never publish its result.
    let starts0 = [
        output_rows[batch] as usize * 4096,
        output_rows[(batch + 1).min(last)] as usize * 4096,
        output_rows[(batch + 2).min(last)] as usize * 4096,
        output_rows[(batch + 3).min(last)] as usize * 4096,
    ];
    let starts1 = if EIGHT {
        [
            output_rows[(batch + 4).min(last)] as usize * 4096,
            output_rows[(batch + 5).min(last)] as usize * 4096,
            output_rows[(batch + 6).min(last)] as usize * 4096,
            output_rows[(batch + 7).min(last)] as usize * 4096,
        ]
    } else {
        [0; 4]
    };
    let mut sums0 = [0.0; 4];
    let mut sums1 = [0.0; 4];
    let mut group = 0;
    while group < 16 {
        let chunk = row * 16 + group;
        let word = chunk * 96 + lane;
        let w0 = words[word];
        let w1 = words[word + 32];
        let w2 = words[word + 64];
        let j = group * 256 + lane;
        // Spell out the eight packed fragments, preserving the lane's original
        // term order. Each reconstruction is shared by four or eight dots.
        macro_rules! term {
            ($code:expr, $offset:expr) => {{
                let weight = widen(unpack_code($code, offsets, exceptions, chunk));
                sums0 = head_add_four(x, starts0, sums0, j + $offset, weight);
                if EIGHT {
                    sums1 = head_add_four(x, starts1, sums1, j + $offset, weight);
                }
            }};
        }
        term!(w0, 0);
        term!(w0 >> 12, 32);
        term!((w0 >> 24) | (w1 << 8), 64);
        term!(w1 >> 4, 96);
        term!(w1 >> 16, 128);
        term!((w1 >> 28) | (w2 << 4), 160);
        term!(w2 >> 8, 192);
        term!(w2 >> 20, 224);
        group += 1;
    }
    sums0 = head_reduce_four(sums0);
    if EIGHT {
        sums1 = head_reduce_four(sums1);
    }
    if lane == 0 {
        let mut b = 0;
        while b < 4 {
            if batch + b < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b) * 151936 + row) = bf(sums0[b]);
                }
            }
            if EIGHT && batch + b + 4 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 4) * 151936 + row) = bf(sums1[b]);
                }
            }
            b += 1;
        }
    }
}

/// Stage one aligned 256-feature stripe for eight or sixteen admitted rows.
/// Padded rows reuse the last valid input; all 256 threads issue two/four copies.
#[inline(always)]
fn stage_head_input<const SIXTEEN: bool>(
    staged: *mut u32,
    x: &[f32],
    selected: &[u32],
    batch: usize,
    chunk: usize,
    t: usize,
) {
    unsafe {
        let mut piece = 0;
        while piece < (if SIXTEEN { 4 } else { 2 }) {
            let i = t * 4 + piece * 1024;
            let row = (batch + i / 256).min(selected.len() - 1);
            let source = selected[row] as usize * 4096 + chunk * 256 + i % 256;
            cuda_device::async_copy::cp_async_ca_16(
                staged.add(i),
                x.as_ptr().add(source).cast::<u32>(),
            );
            piece += 1;
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

/// Each CTA owns eight vocabulary rows and one output group. Adjacent CTAs
/// interleave output groups, retaining weight locality while all eight warps
/// reuse the same staged inputs. The vocabulary is divisible by eight, so the
/// exact grid has no partial CTA. Both slots are fenced before read or reuse.
#[inline(always)]
fn head_packed_staged<const SIXTEEN: bool>(
    staged: *mut u32,
    x: &[f32],
    output_rows: &[u32],
    words: &[u32],
    offsets: &[u32],
    exceptions: &[u16],
    mut out: DisjointSlice<f32>,
) {
    dependency_wait();
    let t = thread::threadIdx_x() as usize;
    let lane = t % 32;
    let width = if SIXTEEN { 16 } else { 8 };
    let groups = output_rows.len().div_ceil(width);
    let block = thread::blockIdx_x() as usize;
    let batch = block % groups * width;
    let row = block / groups * 8 + t / 32;
    let starts0 = [0, 256, 512, 768];
    let starts1 = [1024, 1280, 1536, 1792];
    stage_head_input::<SIXTEEN>(staged, x, output_rows, batch, 0, t);
    let mut sums0 = [0.0; 4];
    let mut sums1 = [0.0; 4];
    let mut sums2 = [0.0; 4];
    let mut sums3 = [0.0; 4];
    let mut group = 0;
    while group < 16 {
        unsafe {
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        let slot = group % 2;
        if group + 1 < 16 {
            stage_head_input::<SIXTEEN>(
                unsafe { staged.add((1 - slot) * width * 256) },
                x,
                output_rows,
                batch,
                group + 1,
                t,
            );
        }
        let input = unsafe {
            core::slice::from_raw_parts(staged.add(slot * width * 256).cast::<f32>(), width * 256)
        };
        let chunk = row * 16 + group;
        let word = chunk * 96 + lane;
        let w0 = words[word];
        let w1 = words[word + 32];
        let w2 = words[word + 64];
        let j = lane;
        // Spell out the eight packed fragments, preserving the lane's original
        // term order. Each reconstruction is shared by eight or sixteen dots.
        macro_rules! term {
            ($code:expr, $offset:expr) => {{
                let weight = widen(unpack_code($code, offsets, exceptions, chunk));
                sums0 = head_add_four(input, starts0, sums0, j + $offset, weight);
                sums1 = head_add_four(input, starts1, sums1, j + $offset, weight);
                if SIXTEEN {
                    sums2 =
                        head_add_four(input, [2048, 2304, 2560, 2816], sums2, j + $offset, weight);
                    sums3 =
                        head_add_four(input, [3072, 3328, 3584, 3840], sums3, j + $offset, weight);
                }
            }};
        }
        term!(w0, 0);
        term!(w0 >> 12, 32);
        term!((w0 >> 24) | (w1 << 8), 64);
        term!(w1 >> 4, 96);
        term!(w1 >> 16, 128);
        term!((w1 >> 28) | (w2 << 4), 160);
        term!(w2 >> 8, 192);
        term!(w2 >> 20, 224);
        thread::sync_threads();
        group += 1;
    }
    sums0 = head_reduce_four(sums0);
    sums1 = head_reduce_four(sums1);
    if SIXTEEN {
        sums2 = head_reduce_four(sums2);
        sums3 = head_reduce_four(sums3);
    }
    if lane == 0 {
        let mut b = 0;
        while b < 4 {
            if batch + b < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b) * 151936 + row) = bf(sums0[b]);
                }
            }
            if batch + b + 4 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 4) * 151936 + row) = bf(sums1[b]);
                }
            }
            if SIXTEEN && batch + b + 8 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 8) * 151936 + row) = bf(sums2[b]);
                }
            }
            if SIXTEEN && batch + b + 12 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 12) * 151936 + row) = bf(sums3[b]);
                }
            }
            b += 1;
        }
    }
}

/// Operator-only 32-output head with one staging slot. Each CTA owns eight
/// vocabulary columns. Ascending terms and descending warp reduction are unchanged.
#[inline(always)]
fn head_packed32_single(
    staged: *mut u32,
    x: &[f32],
    output_rows: &[u32],
    words: &[u32],
    offsets: &[u32],
    exceptions: &[u16],
    mut out: DisjointSlice<f32>,
) {
    dependency_wait();
    let t = thread::threadIdx_x() as usize;
    let lane = t % 32;
    let groups = output_rows.len().div_ceil(32);
    let block = thread::blockIdx_x() as usize;
    let batch = block % groups * 32;
    let row = block / groups * 8 + t / 32;
    let mut sums0 = [0.0; 4];
    let mut sums1 = [0.0; 4];
    let mut sums2 = [0.0; 4];
    let mut sums3 = [0.0; 4];
    let mut sums4 = [0.0; 4];
    let mut sums5 = [0.0; 4];
    let mut sums6 = [0.0; 4];
    let mut sums7 = [0.0; 4];
    let mut group = 0;
    while group < 16 {
        // One complete 32x256 stripe, with eight disjoint aligned copies per thread.
        // Padded rows read a valid selected row and never publish an output.
        unsafe {
            let mut piece = 0;
            while piece < 8 {
                let i = t * 4 + piece * 1024;
                let selected = (batch + i / 256).min(output_rows.len() - 1);
                let from = output_rows[selected] as usize * 4096 + group * 256 + i % 256;
                cuda_device::async_copy::cp_async_ca_16(
                    staged.add(i),
                    x.as_ptr().add(from).cast::<u32>(),
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        let input = unsafe { core::slice::from_raw_parts(staged.cast::<f32>(), 8192) };
        let chunk = row * 16 + group;
        let word = chunk * 96 + lane;
        let w0 = words[word];
        let w1 = words[word + 32];
        let w2 = words[word + 64];
        macro_rules! term32 {
            ($code:expr, $offset:expr) => {{
                let weight = widen(unpack_code($code, offsets, exceptions, chunk));
                sums0 = head_add_four(input, [0, 256, 512, 768], sums0, lane + $offset, weight);
                sums1 = head_add_four(
                    input,
                    [1024, 1280, 1536, 1792],
                    sums1,
                    lane + $offset,
                    weight,
                );
                sums2 = head_add_four(
                    input,
                    [2048, 2304, 2560, 2816],
                    sums2,
                    lane + $offset,
                    weight,
                );
                sums3 = head_add_four(
                    input,
                    [3072, 3328, 3584, 3840],
                    sums3,
                    lane + $offset,
                    weight,
                );
                sums4 = head_add_four(
                    input,
                    [4096, 4352, 4608, 4864],
                    sums4,
                    lane + $offset,
                    weight,
                );
                sums5 = head_add_four(
                    input,
                    [5120, 5376, 5632, 5888],
                    sums5,
                    lane + $offset,
                    weight,
                );
                sums6 = head_add_four(
                    input,
                    [6144, 6400, 6656, 6912],
                    sums6,
                    lane + $offset,
                    weight,
                );
                sums7 = head_add_four(
                    input,
                    [7168, 7424, 7680, 7936],
                    sums7,
                    lane + $offset,
                    weight,
                );
            }};
        }
        term32!(w0, 0);
        term32!(w0 >> 12, 32);
        term32!((w0 >> 24) | (w1 << 8), 64);
        term32!(w1 >> 4, 96);
        term32!(w1 >> 16, 128);
        term32!((w1 >> 28) | (w2 << 4), 160);
        term32!(w2 >> 8, 192);
        term32!(w2 >> 20, 224);
        // Every warp finishes consuming the sole slot before any thread overwrites it.
        thread::sync_threads();
        group += 1;
    }
    sums0 = head_reduce_four(sums0);
    sums1 = head_reduce_four(sums1);
    sums2 = head_reduce_four(sums2);
    sums3 = head_reduce_four(sums3);
    sums4 = head_reduce_four(sums4);
    sums5 = head_reduce_four(sums5);
    sums6 = head_reduce_four(sums6);
    sums7 = head_reduce_four(sums7);
    if lane == 0 {
        let mut b = 0;
        while b < 4 {
            if batch + b + 0 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 0) * 151936 + row) = bf(sums0[b]);
                }
            }
            if batch + b + 4 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 4) * 151936 + row) = bf(sums1[b]);
                }
            }
            if batch + b + 8 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 8) * 151936 + row) = bf(sums2[b]);
                }
            }
            if batch + b + 12 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 12) * 151936 + row) = bf(sums3[b]);
                }
            }
            if batch + b + 16 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 16) * 151936 + row) = bf(sums4[b]);
                }
            }
            if batch + b + 20 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 20) * 151936 + row) = bf(sums5[b]);
                }
            }
            if batch + b + 24 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 24) * 151936 + row) = bf(sums6[b]);
                }
            }
            if batch + b + 28 < output_rows.len() {
                unsafe {
                    *out.get_unchecked_mut((batch + b + 28) * 151936 + row) = bf(sums7[b]);
                }
            }
            b += 1;
        }
    }
}

#[inline(always)]
fn best_warp(mut value: f32, mut index: u32, mut bad: u32) -> (f32, u32, u32) {
    let mut d = 16;
    while d > 0 {
        let other = f32::from_bits(warp::shuffle_down(value.to_bits(), d));
        let other_index = warp::shuffle_down(index, d);
        bad |= warp::shuffle_down(bad, d);
        if other > value || (other == value && other_index < index) {
            value = other;
            index = other_index;
        }
        d /= 2;
    }
    (value, index, bad)
}

#[inline(always)]
fn publish_absmax(value: f32, p: *mut u32) {
    let bits = value.to_bits() & 0x7fffffff;
    // f32::max ignores NaNs; positive float bits otherwise have unsigned order.
    let mut maximum = if bits > 0x7f800000 { 0 } else { bits };
    maximum = warp_max(maximum);
    if thread::threadIdx_x() % 32 == 0 {
        // SAFETY: reset by an earlier stream-ordered kernel. Concurrent accesses
        // in this producer are exclusively device-scope atomic RMWs.
        unsafe {
            cuda_device::atomic::DeviceAtomicU32::from_ptr(p)
                .fetch_max(maximum, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
    }
}

// Inline PTX avoids CUDA 12.8's offline-linker crash on the tensor-copy intrinsic.
// SAFETY: immutable encoded map/source allocations outlive the copy; dst is a
// 1024-byte-aligned CTA-local tile and barrier is initialized, fenced, and expecting
// exactly this tile's bytes. Consumers wait before reading or reusing the tile.
#[inline(always)]
unsafe fn tma_copy(
    dst: *mut u8,
    map: *const cuda_device::tma::TmaDescriptor,
    x: i32,
    y: i32,
    barrier: *mut cuda_device::barrier::Barrier,
) {
    unsafe {
        cuda_device::ptx_asm!(
            "{ .reg .u64 d64, b64; .reg .u32 d32, b32; cvta.to.shared.u64 d64, %0; cvt.u32.u64 d32, d64; cvta.to.shared.u64 b64, %4; cvt.u32.u64 b32, b64; cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [d32], [%1, {%2, %3}], [b32]; }",
            in("l") dst,
            in("l") map,
            in("r") x,
            in("r") y,
            in("l") barrier,
        );
    }
}

/// Decode graphs use programmatic edges: a kernel may launch while its
/// predecessor runs. Every real CTA waits for the predecessor's completion and
/// memory visibility before touching any produced, consumed or reused buffer,
/// then admits its own successor, keeping exactly one kernel of lookahead.
/// Both instructions are no-ops for ordinary stream launches.
#[inline(always)]
fn dependency_wait() {
    unsafe {
        cuda_device::ptx_asm!("griddepcontrol.wait;", clobber("memory"));
        cuda_device::ptx_asm!("griddepcontrol.launch_dependents;", clobber("memory"));
    }
}

/// Attention leaves DRAM idle. Blocks at or beyond `real` warm L2 with bytes
/// [from, to) of the next immutable weight stream, one contiguous share per
/// block, then exit before any barrier. They read and write no buffer, so they
/// need not wait on the predecessor; under programmatic launch they overlap it.
/// A bulk L2 prefetch returns no data and never affects results. Returns
/// whether the caller is a prefetch block.
#[inline(always)]
fn prefetch_share(base: *const u8, len: usize, from: u32, to: u32, real: usize) -> bool {
    let block = thread::blockIdx_x() as usize;
    if block < real {
        return false;
    }
    let to = (to as usize).min(len);
    let from = from as usize;
    if thread::threadIdx_x() == 0 && from < to {
        let parts = thread::gridDim_x() as usize - real;
        // 128-byte shares keep every address and 16-byte size multiple aligned.
        let share = ((to - from).div_ceil(parts) + 127) / 128 * 128;
        let mut offset = from + (block - real) * share;
        let end = (offset + share).min(to);
        while offset + 16 <= end {
            let bytes = (end - offset).min(16384);
            // SAFETY: [offset, offset+bytes) lies within the live allocation;
            // base is 256-byte aligned and offset/bytes are multiples of 16.
            unsafe {
                cuda_device::ptx_asm!(
                    "cp.async.bulk.prefetch.L2.global [%0], %1;",
                    in("l") base.add(offset),
                    in("r") (bytes & !15) as u32,
                );
            }
            offset += 16384;
        }
    }
    true
}

/// Before waiting on its predecessor, a projection warp warms L2 with the leading
/// bytes of its own immutable weight row. Under programmatic launch this runs
/// while the preceding latency-bound kernel executes. A pure cache hint.
#[inline(always)]
fn prefetch_row(row: *const u8, row_bytes: usize, bytes: u32, lane: usize) {
    let end = (bytes as usize).min(row_bytes);
    let mut offset = lane * 128;
    while offset < end {
        // SAFETY: the line start lies within this live weight row.
        unsafe {
            cuda_device::ptx_asm!("prefetch.global.L2 [%0];", in("l") row.add(offset));
        }
        offset += 4096;
    }
}

/// Warm a layer's visible KV prefix while the QKV projection streams, so the
/// following attention's cache tiles hit L2 instead of queueing behind weight
/// prefetches. One 128-byte line per thread: K is coordinate-major (64 lines per
/// 4096-token row), V is token-major and contiguous. A pure cache hint.
#[inline(always)]
fn prefetch_kv(ck: &[u16], cv: &[u16], position: &[u32]) {
    let end = position[1] as usize + 1;
    let g = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    let k_lines = ck.len() / 4096 * 64;
    let address = if g < k_lines {
        if g % 64 * 64 < end {
            Some(unsafe { ck.as_ptr().add(g / 64 * 4096 + g % 64 * 64) })
        } else {
            None
        }
    } else if (g - k_lines) * 64 < (end * 1024).min(cv.len()) {
        Some(unsafe { cv.as_ptr().add((g - k_lines) * 64) })
    } else {
        None
    };
    if let Some(address) = address {
        // SAFETY: the line start lies within the live cache allocation.
        unsafe {
            cuda_device::ptx_asm!("prefetch.global.L2 [%0];", in("l") address);
        }
    }
}

// Same bounded integer dot and ordered FP32 scaling as the single-projection oracle.
#[inline(always)]
fn scaled_dot(
    q: &[i8],
    w: &[i8],
    asc: &[f32],
    ws: &[f32],
    r: usize,
    c: usize,
    k: usize,
    lane: usize,
) -> f32 {
    let mut acc = 0_i32;
    let mut j = lane * 4;
    while j < k {
        // SAFETY: all admitted widths divisible by four; CUDA bases and row offsets aligned.
        let a = unsafe { q.as_ptr().add(r * k + j).cast::<u32>().read() };
        let b = unsafe { w.as_ptr().add(c * k + j).cast::<u32>().read() };
        acc = dp4a_s32(a, b, acc);
        j += 128;
    }
    acc = warp_sum(acc);
    bf(mul(mul(acc as f32, asc[r]), ws[c]))
}

// Decode a complete original BF16 bit pattern, including uncommon exponents.
#[inline(always)]
fn unpack_code(code: u32, offsets: &[u32], exceptions: &[u16], chunk: usize) -> u16 {
    let code = code & 4095;
    if code >> 8 == 15 {
        exceptions[offsets[chunk] as usize + (code & 255) as usize]
    } else {
        (((code & 128) << 8) | (((code >> 8) + 109) << 7) | (code & 127)) as u16
    }
}

#[inline(always)]
fn unpack_bf16(words: &[u32], offsets: &[u32], exceptions: &[u16], i: usize) -> u16 {
    let step = i % 256 / 32;
    let word = i / 256 * 96 + i % 32 + (step * 12 / 32) * 32;
    let shift = step * 12 % 32;
    let mut code = words[word] >> shift;
    if shift > 20 {
        code |= words[word + 32] << (32 - shift);
    }
    unpack_code(code, offsets, exceptions, i / 256)
}

// Shared arrays are 1024-byte aligned, making the TMA swizzle base offset zero.
#[inline(always)]
fn tma_word(row: usize, word: usize) -> usize {
    row * 32 + (word ^ ((row % 8) * 4))
}

#[inline(always)]
unsafe fn stage_projection_tile<const NARROW: bool>(
    a_shared: *mut u32,
    b_shared: *mut u32,
    q: &[i8],
    w: &[i8],
    nr: usize,
    n: usize,
    k: usize,
    row: usize,
    col: usize,
    t: usize,
    base: usize,
) {
    // SAFETY: each thread owns one aligned 16-byte A fragment and two
    // B fragments. Tail rows copy zero bytes from a valid dummy address.
    // The issuing thread waits for its copies, then the CTA publishes them.
    unsafe {
        let r = row + t / 8;
        let offset = if r < nr { r * k + base + t % 8 * 16 } else { 0 };
        cuda_device::async_copy::cp_async_ca_zfill_16(
            a_shared.add(t / 8 * 36 + t % 8 * 4),
            q.as_ptr().add(offset).cast::<u8>(),
            if r < nr { 16 } else { 0 },
        );
        let mut piece = 0;
        while piece < 2 {
            let i = t + piece * (if NARROW { 128 } else { 256 });
            let c = col + i / 8;
            let offset = if c < n { c * k + base + i % 8 * 16 } else { 0 };
            cuda_device::async_copy::cp_async_ca_zfill_16(
                b_shared.add(i / 8 * 36 + i % 8 * 4),
                w.as_ptr().add(offset).cast::<u8>(),
                if c < n { 16 } else { 0 },
            );
            piece += 1;
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

#[cuda_module]
pub mod device {
    use super::*;
    #[kernel]
    pub fn rounding_probe(input: &[f32], mut out: DisjointSlice<i8>) {
        let i = thread::index_1d().get();
        if i < input.len() {
            unsafe {
                *out.get_unchecked_mut(i) = round(input[i]);
            }
        }
    }
    #[kernel]
    pub fn embedding(ids: &[u32], w: &[u16], mut out: DisjointSlice<f32>) {
        dependency_wait();
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        // SAFETY: admitted IDs and distinct output coordinates.
        unsafe {
            *out.get_unchecked_mut(i) = widen(w[ids[i / 4096] as usize * 4096 + i % 4096]);
        }
    }
    #[kernel]
    pub fn norm(x: &[f32], gain: &[f32], mut out: DisjointSlice<f32>) {
        static mut SUM: SharedArray<f32, 256> = SharedArray::UNINIT;
        dependency_wait();
        let row = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let n = gain.len();
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUM) };
        // Issue each 4096-coordinate chunk's loads before its dependent sums; the
        // per-thread addition order is unchanged. A model row is one chunk, whose
        // values and gains stay in registers for the output.
        let mut values = [0.0_f32; 16];
        let mut gains = [0.0_f32; 16];
        let mut s = 0.0;
        let mut base = 0;
        while base < n {
            let mut i = 0;
            while i < 16 {
                let j = base + t + i * 256;
                if j < n {
                    values[i] = x[row * n + j];
                    gains[i] = gain[j];
                }
                i += 1;
            }
            let mut i = 0;
            while i < 16 && base + t + i * 256 < n {
                s = add(s, mul(values[i], values[i]));
                i += 1;
            }
            base += 4096;
        }
        // SAFETY: 256 threads initialize distinct slots; barriers separate tree levels.
        unsafe {
            *p.add(t) = s;
        }
        thread::sync_threads();
        let mut d = 128;
        while d > 0 {
            if t < d {
                unsafe {
                    *p.add(t) = add(*p.add(t), *p.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let scale = rsqrt(add(div(unsafe { *p }, n as f32), 1e-6));
        if n <= 4096 {
            let mut i = 0;
            while i < 16 && t + i * 256 < n {
                unsafe {
                    *out.get_unchecked_mut(row * n + t + i * 256) =
                        bf(mul(mul(values[i], scale), gains[i]));
                }
                i += 1;
            }
        } else {
            let mut j = t;
            while j < n {
                unsafe {
                    *out.get_unchecked_mut(row * n + j) =
                        bf(mul(mul(x[row * n + j], scale), gain[j]));
                }
                j += 256;
            }
        }
    }
    #[kernel]
    pub fn quantize(
        x: &[f32],
        width: u32,
        mut q: DisjointSlice<i8>,
        mut scales: DisjointSlice<f32>,
        pf: &[i8],
        pf_from: u32,
        pf_to: u32,
    ) {
        static mut MAX: SharedArray<f32, 32> = SharedArray::UNINIT;
        if prefetch_share(pf.as_ptr().cast(), pf.len(), pf_from, pf_to, scales.len()) {
            return;
        }
        dependency_wait();
        let r = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let n = width as usize;
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX) };
        let mut m = 1e-8_f32;
        let mut j = t;
        while j < n {
            m = m.max(x[r * n + j].abs());
            j += 1024;
        }
        let s = div(max_1024(m, p, t), 127.0);
        if t == 0 {
            unsafe {
                *scales.get_unchecked_mut(r) = s;
            }
        }
        let mut j = t;
        while j < n {
            unsafe {
                *q.get_unchecked_mut(r * n + j) = round(div(x[r * n + j], s));
            }
            j += 1024;
        }
    }
    /// Quantize independent coordinates using a completed producer's token max.
    #[kernel]
    pub fn quantize_with_max(
        x: &[f32],
        width: u32,
        maxima: &[u32],
        mut q: DisjointSlice<i8>,
        mut scales: DisjointSlice<f32>,
    ) {
        dependency_wait();
        let i = thread::index_1d().get();
        if i < x.len() {
            let n = width as usize;
            let r = i / n;
            let scale = div(f32::from_bits(maxima[r]), 127.0);
            unsafe {
                *q.get_unchecked_mut(i) = round(div(x[i], scale));
                if i % n == 0 {
                    *scales.get_unchecked_mut(r) = scale;
                }
            }
        }
    }
    /// Fixed hidden-width RMSNorm followed by dynamic token quantization.
    /// Keep the BF16 norm boundary and the original 256-thread sum tree.
    #[kernel]
    pub fn norm_quantize(
        x: &[f32],
        gain: &[f32],
        mut out: DisjointSlice<f32>,
        mut q: DisjointSlice<i8>,
        mut scales: DisjointSlice<f32>,
        mut maxima: DisjointSlice<u32>,
    ) {
        static mut REDUCE: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut VALUES: SharedArray<f32, 4096> = SharedArray::UNINIT;
        dependency_wait();
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE) };
        let shared_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUES) };
        let r = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let mut values = [0.0_f32; 4];
        // Gains are cold in L2 each step; load them before the reduction.
        let mut gains = [0.0_f32; 4];
        let mut i = 0;
        while i < 4 {
            let v = x[r * 4096 + t + i * 1024];
            values[i] = v;
            gains[i] = gain[t + i * 1024];
            unsafe {
                *shared_values.add(t + i * 1024) = v;
            }
            i += 1;
        }
        thread::sync_threads();
        if t < 256 {
            let mut sum = 0.0;
            let mut i = 0;
            while i < 16 {
                let v = unsafe { *shared_values.add(t + i * 256) };
                sum = add(sum, mul(v, v));
                i += 1;
            }
            unsafe {
                *p.add(t) = sum;
            }
        }
        thread::sync_threads();
        if t < 32 {
            // Precisely the old distances 128, 64, 32, then warp distances 16..1.
            let mut s = unsafe {
                add(
                    add(
                        add(*p.add(t), *p.add(t + 128)),
                        add(*p.add(t + 64), *p.add(t + 192)),
                    ),
                    add(
                        add(*p.add(t + 32), *p.add(t + 160)),
                        add(*p.add(t + 96), *p.add(t + 224)),
                    ),
                )
            };
            let mut d = 16;
            while d > 0 {
                s = add(s, f32::from_bits(warp::shuffle_down(s.to_bits(), d)));
                d /= 2;
            }
            if t == 0 {
                unsafe {
                    *p = s;
                }
            }
        }
        thread::sync_threads();
        let inv = rsqrt(add(div(unsafe { *p }, 4096.0), 1e-6));
        let mut maximum = 1e-8_f32;
        let mut i = 0;
        while i < 4 {
            let j = t + i * 1024;
            let v = bf(mul(mul(values[i], inv), gains[i]));
            values[i] = v;
            maximum = maximum.max(v.abs());
            unsafe {
                *out.get_unchecked_mut(r * 4096 + j) = v;
            }
            i += 1;
        }
        // Every thread must consume the norm scale before REDUCE is reused.
        thread::sync_threads();
        let s = div(max_1024(maximum, p, t), 127.0);
        if t == 0 {
            unsafe {
                *scales.get_unchecked_mut(r) = s;
                *maxima.get_unchecked_mut(r) = 1e-8_f32.to_bits();
            }
        }
        let mut i = 0;
        while i < 4 {
            unsafe {
                *q.get_unchecked_mut(r * 4096 + t + i * 1024) = round(div(values[i], s));
            }
            i += 1;
        }
    }
    /// Preserve the rounded projection while emitting its rounded residual sum.
    #[kernel]
    pub fn linear_residual(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        residual: &[f32],
        mut out: DisjointSlice<f32>,
        mut combined: DisjointSlice<f32>,
        prefetch: u32,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let index = tid / 32;
        let k = width as usize;
        if index < out.len() {
            let row = unsafe { w.as_ptr().add(index % ws.len() * k) };
            prefetch_row(row.cast(), k, prefetch, lane);
        }
        dependency_wait();
        if index >= out.len() {
            return;
        }
        let value = scaled_dot(
            q,
            w,
            asc,
            ws,
            index / ws.len(),
            index % ws.len(),
            width as usize,
            lane,
        );
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(index) = value;
                *combined.get_unchecked_mut(index) = bf(add(residual[index], value));
            }
        }
    }
    /// Each warp computes matching gate/up coordinates from shared activation loads.
    #[kernel]
    pub fn linear_swiglu(
        q: &[i8],
        asc: &[f32],
        width: u32,
        w0: &[i8],
        s0: &[f32],
        mut gate: DisjointSlice<f32>,
        w1: &[i8],
        s1: &[f32],
        mut up: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut maxima: DisjointSlice<u32>,
        prefetch: u32,
    ) {
        let lane = thread::threadIdx_x() as usize % 32;
        let c = thread::index_1d().get() / 32;
        if c < s0.len() {
            let k = width as usize;
            prefetch_row(unsafe { w0.as_ptr().add(c * k) }.cast(), k, prefetch, lane);
            prefetch_row(unsafe { w1.as_ptr().add(c * k) }.cast(), k, prefetch, lane);
        }
        dependency_wait();
        static mut MAX: SharedArray<u32, 8> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let c = thread::index_1d().get() / 32;
        let k = width as usize;
        let mut maximum = 0;
        if c < s0.len() {
            let mut g = 0_i32;
            let mut u = 0_i32;
            let mut j = lane * 4;
            while j < k {
                // SAFETY: complete four-byte-aligned rows, K divisible by 128.
                unsafe {
                    let a = q.as_ptr().add(j).cast::<u32>().read();
                    g = dp4a_s32(a, w0.as_ptr().add(c * k + j).cast::<u32>().read(), g);
                    u = dp4a_s32(a, w1.as_ptr().add(c * k + j).cast::<u32>().read(), u);
                }
                j += 128;
            }
            g = warp_sum(g);
            u = warp_sum(u);
            if lane == 0 {
                let g = bf(mul(mul(g as f32, asc[0]), s0[c]));
                let u = bf(mul(mul(u as f32, asc[0]), s1[c]));
                let value = bf(mul(div(g, add(1.0, exp(-g))), u));
                unsafe {
                    *gate.get_unchecked_mut(c) = g;
                    *up.get_unchecked_mut(c) = u;
                    *out.get_unchecked_mut(c) = value;
                }
                maximum = if value.is_nan() {
                    0
                } else {
                    value.abs().to_bits()
                };
            }
        }
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MAX) };
        if lane == 0 {
            unsafe {
                *p.add(t / 32) = maximum;
            }
        }
        thread::sync_threads();
        if t < 32 {
            let value = if t < 8 {
                f32::from_bits(unsafe { *p.add(t) })
            } else {
                0.0
            };
            publish_absmax(value, maxima.as_mut_ptr());
        }
    }
    /// Independent sibling projections share one grid; each warp owns one dot.
    #[kernel]
    pub fn linear_pair(
        q: &[i8],
        asc: &[f32],
        width: u32,
        w0: &[i8],
        s0: &[f32],
        mut out0: DisjointSlice<f32>,
        w1: &[i8],
        s1: &[f32],
        mut out1: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let index = tid / 32;
        let channels = s0.len() + s1.len();
        if index >= asc.len() * channels {
            return;
        }
        let r = index / channels;
        let channel = index % channels;
        if channel < s0.len() {
            let c = channel - (0);
            let value = scaled_dot(q, w0, asc, s0, r, c, width as usize, lane);
            if lane == 0 {
                unsafe {
                    *out0.get_unchecked_mut(r * s0.len() + c) = value;
                }
            }
        } else if channel < s0.len() + s1.len() {
            let c = channel - (s0.len());
            let value = scaled_dot(q, w1, asc, s1, r, c, width as usize, lane);
            if lane == 0 {
                unsafe {
                    *out1.get_unchecked_mut(r * s1.len() + c) = value;
                }
            }
        }
    }
    /// Independent sibling projections share one grid; each warp owns one dot.
    #[kernel]
    pub fn linear_triple(
        q: &[i8],
        asc: &[f32],
        width: u32,
        w0: &[i8],
        s0: &[f32],
        mut out0: DisjointSlice<f32>,
        w1: &[i8],
        s1: &[f32],
        mut out1: DisjointSlice<f32>,
        w2: &[i8],
        s2: &[f32],
        mut out2: DisjointSlice<f32>,
        ck: &[u16],
        cv: &[u16],
        position: &[u32],
        prefetch: u32,
    ) {
        if prefetch != 0 {
            prefetch_kv(ck, cv, position);
        }
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let index = tid / 32;
        let channels = s0.len() + s1.len() + s2.len();
        if index < asc.len() * channels {
            let channel = index % channels;
            let k = width as usize;
            let row = if channel < s0.len() {
                unsafe { w0.as_ptr().add(channel * k) }
            } else if channel < s0.len() + s1.len() {
                unsafe { w1.as_ptr().add((channel - s0.len()) * k) }
            } else {
                unsafe { w2.as_ptr().add((channel - s0.len() - s1.len()) * k) }
            };
            prefetch_row(row.cast(), k, prefetch, lane);
        }
        dependency_wait();
        if index >= asc.len() * channels {
            return;
        }
        let r = index / channels;
        let channel = index % channels;
        if channel < s0.len() {
            let c = channel - (0);
            let value = scaled_dot(q, w0, asc, s0, r, c, width as usize, lane);
            if lane == 0 {
                unsafe {
                    *out0.get_unchecked_mut(r * s0.len() + c) = value;
                }
            }
        } else if channel < s0.len() + s1.len() {
            let c = channel - (s0.len());
            let value = scaled_dot(q, w1, asc, s1, r, c, width as usize, lane);
            if lane == 0 {
                unsafe {
                    *out1.get_unchecked_mut(r * s1.len() + c) = value;
                }
            }
        } else if channel < s0.len() + s1.len() + s2.len() {
            let c = channel - (s0.len() + s1.len());
            let value = scaled_dot(q, w2, asc, s2, r, c, width as usize, lane);
            if lane == 0 {
                unsafe {
                    *out2.get_unchecked_mut(r * s2.len() + c) = value;
                }
            }
        }
    }
    #[kernel]
    pub fn linear(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let index = tid / 32;
        if index >= out.len() {
            return;
        }
        let k = width as usize;
        let r = index / ws.len();
        let c = index % ws.len();
        let mut acc = 0_i32;
        let mut j = lane * 4;
        while j < k {
            // SAFETY: all admitted widths divisible by four; CUDA bases and row offsets aligned.
            let a = unsafe { q.as_ptr().add(r * k + j).cast::<u32>().read() };
            let b = unsafe { w.as_ptr().add(c * k + j).cast::<u32>().read() };
            acc = dp4a_s32(a, b, acc);
            j += 128;
        }
        let mut d = 16;
        while d > 0 {
            acc += warp::shuffle_down(acc as u32, d) as i32;
            d /= 2;
        }
        // SAFETY: one writer per warp/output; sum bounded by 12288*127^2 < 2^31.
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(index) = bf(mul(mul(acc as f32, asc[r]), ws[c]));
            }
        }
    }
    /// One warp computes a 16-row by 8-channel tile using exact INT8 tensor cores.
    /// Fragment layout follows PTX mma.m16n8k32.row.col.s32.s8.s8.s32.
    #[kernel]
    pub fn linear_prefill(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let tile = tid / 32;
        let columns = ws.len().div_ceil(8);
        let row = (tile / columns) * 16;
        let col = (tile % columns) * 8;
        if row >= asc.len() {
            return;
        }
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let mut acc = [0_i32; 4];
        let mut j = 0;
        while j < k {
            let mut a = [0_u32; 4];
            let mut b = [0_u32; 2];
            // SAFETY: complete four-byte packed fragments; K is a multiple of 32.
            // Missing tile rows/columns are zero padded without reading outside buffers.
            unsafe {
                if row + group < asc.len() {
                    a[0] = q
                        .as_ptr()
                        .add((row + group) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    a[2] = q
                        .as_ptr()
                        .add((row + group) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                if row + group + 8 < asc.len() {
                    a[1] = q
                        .as_ptr()
                        .add((row + group + 8) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    a[3] = q
                        .as_ptr()
                        .add((row + group + 8) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                if col + group < ws.len() {
                    b[0] = w
                        .as_ptr()
                        .add((col + group) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    b[1] = w
                        .as_ptr()
                        .add((col + group) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                // Every live warp lane executes this operation; INT32 cannot overflow.
                acc = cuda_device::wmma::mma_m16n8k32_s32_s8(acc, a, b);
            }
            j += 32;
        }
        let mut i = 0;
        while i < 4 {
            let r = row + group + (i / 2) * 8;
            let c = col + part * 2 + i % 2;
            if r < asc.len() && c < ws.len() {
                // SAFETY: each warp/lane/fragment owns a distinct admitted output.
                unsafe {
                    *out.get_unchecked_mut(r * ws.len() + c) =
                        bf(mul(mul(acc[i] as f32, asc[r]), ws[c]));
                }
            }
            i += 1;
        }
    }
    /// Eight warps divide K for an exact 16-row by 8-channel INT8 tile.
    /// Fragment layout follows PTX mma.m16n8k32.row.col.s32.s8.s8.s32.
    #[kernel]
    pub fn linear_splitk(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<i32, 1024> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tile = thread::blockIdx_x() as usize;
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let columns = ws.len().div_ceil(8);
        let row = (tile / columns) * 16;
        let col = (tile % columns) * 8;
        if row >= asc.len() {
            return;
        }
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let mut acc = [0_i32; 4];
        let mut j = warp_id * (k / 8);
        let end = j + k / 8;
        while j < end {
            let mut a = [0_u32; 4];
            let mut b = [0_u32; 2];
            // SAFETY: complete four-byte packed fragments; K is a multiple of 32.
            // Missing tile rows/columns are zero padded without reading outside buffers.
            unsafe {
                if row + group < asc.len() {
                    a[0] = q
                        .as_ptr()
                        .add((row + group) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    a[2] = q
                        .as_ptr()
                        .add((row + group) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                if row + group + 8 < asc.len() {
                    a[1] = q
                        .as_ptr()
                        .add((row + group + 8) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    a[3] = q
                        .as_ptr()
                        .add((row + group + 8) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                if col + group < ws.len() {
                    b[0] = w
                        .as_ptr()
                        .add((col + group) * k + j + part * 4)
                        .cast::<u32>()
                        .read();
                    b[1] = w
                        .as_ptr()
                        .add((col + group) * k + j + part * 4 + 16)
                        .cast::<u32>()
                        .read();
                }
                // Every live warp lane executes this operation; INT32 cannot overflow.
                acc = cuda_device::wmma::mma_m16n8k32_s32_s8(acc, a, b);
            }
            j += 32;
        }
        let mut i = 0;
        while i < 4 {
            // Each lane owns four distinct cells within its warp's 16x8 partial tile.
            unsafe {
                *partial.add(warp_id * 128 + (group + i / 2 * 8) * 8 + part * 2 + i % 2) = acc[i];
            }
            i += 1;
        }
        thread::sync_threads();
        if t < 128 {
            let r = row + t / 8;
            let c = col + t % 8;
            if r < asc.len() && c < ws.len() {
                let mut sum = 0_i32;
                let mut w = 0;
                while w < 8 {
                    sum += unsafe { *partial.add(w * 128 + t) };
                    w += 1;
                }
                // Same scaling/rounding as DP4A. Exact integer sum cannot overflow.
                unsafe {
                    *out.get_unchecked_mut(r * ws.len() + c) =
                        bf(mul(mul(sum as f32, asc[r]), ws[c]));
                }
            }
        }
    }
    /// A 32x64 output tile cooperatively stages 128 K elements at a time.
    /// Eight warps each own a 16x16 tile (two adjacent m16n8k32 MMAs).
    #[kernel]
    pub fn linear_tiled(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut A: SharedArray<u32, 1152, 16> = SharedArray::UNINIT;
        static mut B: SharedArray<u32, 2304, 16> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tiles_n = ws.len().div_ceil(64);
        let tile = thread::blockIdx_x() as usize;
        let row = tile / tiles_n * 32;
        let col = tile % tiles_n * 64;
        let wr = warp_id / 4 * 16;
        let wc = warp_id % 4 * 16;
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let a_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A) };
        let b_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B) };
        let mut c0 = [0_i32; 4];
        let mut c1 = [0_i32; 4];
        let mut base = 0;
        while base < k {
            // SAFETY: each thread owns one aligned 16-byte A fragment and two
            // B fragments. Tail rows copy zero bytes from a valid dummy address.
            // The issuing thread waits for its copies, then the CTA publishes them.
            unsafe {
                let r = row + t / 8;
                let offset = if r < asc.len() {
                    r * k + base + t % 8 * 16
                } else {
                    0
                };
                cuda_device::async_copy::cp_async_ca_zfill_16(
                    a_shared.add(t / 8 * 36 + t % 8 * 4),
                    q.as_ptr().add(offset).cast::<u8>(),
                    if r < asc.len() { 16 } else { 0 },
                );
                let mut piece = 0;
                while piece < 2 {
                    let i = t + piece * 256;
                    let c = col + i / 8;
                    let offset = if c < ws.len() {
                        c * k + base + i % 8 * 16
                    } else {
                        0
                    };
                    cuda_device::async_copy::cp_async_ca_zfill_16(
                        b_shared.add(i / 8 * 36 + i % 8 * 4),
                        w.as_ptr().add(offset).cast::<u8>(),
                        if c < ws.len() { 16 } else { 0 },
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let mut j = 0;
            while j < 32 {
                unsafe {
                    // Each lane supplies an aligned row of one of four 8x8
                    // b16 matrices. Their bytes are the same INT8 MMA fragments.
                    let a = cuda_device::wmma::ldmatrix_x4(
                        a_shared.add((wr + lane % 16) * 36 + j + lane / 16 * 4),
                    );
                    let b = cuda_device::wmma::ldmatrix_x4(
                        b_shared.add((wc + lane / 16 * 8 + lane % 8) * 36 + j + lane / 8 % 2 * 4),
                    );
                    let b0 = [b[0], b[1]];
                    let b1 = [b[2], b[3]];
                    c0 = cuda_device::wmma::mma_m16n8k32_s32_s8(c0, a, b0);
                    c1 = cuda_device::wmma::mma_m16n8k32_s32_s8(c1, a, b1);
                }
                j += 8;
            }
            // No thread may overwrite staging before the last warp consumes it.
            thread::sync_threads();
            base += 128;
        }
        let mut i = 0;
        while i < 4 {
            let r = row + wr + group + i / 2 * 8;
            let c = col + wc + part * 2 + i % 2;
            if r < asc.len() {
                unsafe {
                    if c < ws.len() {
                        *out.get_unchecked_mut(r * ws.len() + c) =
                            bf(mul(mul(c0[i] as f32, asc[r]), ws[c]));
                    }
                    if c + 8 < ws.len() {
                        *out.get_unchecked_mut(r * ws.len() + c + 8) =
                            bf(mul(mul(c1[i] as f32, asc[r]), ws[c + 8]));
                    }
                }
            }
            i += 1;
        }
    }
    /// Double-buffered 32x64 tile; eight warps consume one slot while the
    /// next 128-wide input/weight stripe is copied into the other.
    #[kernel]
    pub fn linear_tiled_pipeline(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut A: SharedArray<u32, 2304, 16> = SharedArray::UNINIT;
        static mut B: SharedArray<u32, 4608, 16> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tiles_n = ws.len().div_ceil(64);
        let tile = thread::blockIdx_x() as usize;
        let row = tile / tiles_n * 32;
        let col = tile % tiles_n * 64;
        let wr = warp_id / 4 * 16;
        let wc = warp_id % 4 * 16;
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let a_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A) };
        let b_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B) };
        let mut c0 = [0_i32; 4];
        let mut c1 = [0_i32; 4];
        unsafe {
            stage_projection_tile::<false>(
                a_shared,
                b_shared,
                q,
                w,
                asc.len(),
                ws.len(),
                k,
                row,
                col,
                t,
                0,
            );
        }
        let mut base = 0;
        while base < k {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let slot = base / 128 % 2;
            if base + 128 < k {
                unsafe {
                    stage_projection_tile::<false>(
                        a_shared.add((1 - slot) * 1152),
                        b_shared.add((1 - slot) * 2304),
                        q,
                        w,
                        asc.len(),
                        ws.len(),
                        k,
                        row,
                        col,
                        t,
                        base + 128,
                    );
                }
            }
            let a_tile = unsafe { a_shared.add(slot * 1152) };
            let b_tile = unsafe { b_shared.add(slot * 2304) };
            let mut j = 0;
            while j < 32 {
                unsafe {
                    // Each lane supplies an aligned row of one of four 8x8
                    // b16 matrices. Their bytes are the same INT8 MMA fragments.
                    let a = cuda_device::wmma::ldmatrix_x4(
                        a_tile.add((wr + lane % 16) * 36 + j + lane / 16 * 4),
                    );
                    let b = cuda_device::wmma::ldmatrix_x4(
                        b_tile.add((wc + lane / 16 * 8 + lane % 8) * 36 + j + lane / 8 % 2 * 4),
                    );
                    let b0 = [b[0], b[1]];
                    let b1 = [b[2], b[3]];
                    c0 = cuda_device::wmma::mma_m16n8k32_s32_s8(c0, a, b0);
                    c1 = cuda_device::wmma::mma_m16n8k32_s32_s8(c1, a, b1);
                }
                j += 8;
            }
            // No thread may overwrite staging before the last warp consumes it.
            thread::sync_threads();
            base += 128;
        }
        let mut i = 0;
        while i < 4 {
            let r = row + wr + group + i / 2 * 8;
            let c = col + wc + part * 2 + i % 2;
            if r < asc.len() {
                unsafe {
                    if c < ws.len() {
                        *out.get_unchecked_mut(r * ws.len() + c) =
                            bf(mul(mul(c0[i] as f32, asc[r]), ws[c]));
                    }
                    if c + 8 < ws.len() {
                        *out.get_unchecked_mut(r * ws.len() + c + 8) =
                            bf(mul(mul(c1[i] as f32, asc[r]), ws[c + 8]));
                    }
                }
            }
            i += 1;
        }
    }
    /// Double-buffered 16x32 tile gives small row batches more independent CTAs.
    /// Four warps own disjoint 16x8 integer accumulator tiles.
    #[kernel]
    pub fn linear_narrow_pipeline(
        q: &[i8],
        asc: &[f32],
        w: &[i8],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut A: SharedArray<u32, 1152, 16> = SharedArray::UNINIT;
        static mut B: SharedArray<u32, 2304, 16> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tiles_n = ws.len().div_ceil(32);
        let tile = thread::blockIdx_x() as usize;
        let row = tile / tiles_n * 16;
        let col = tile % tiles_n * 32;
        let wr = 0;
        let wc = warp_id * 8;
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let a_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A) };
        let b_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B) };
        let mut c0 = [0_i32; 4];
        unsafe {
            stage_projection_tile::<true>(
                a_shared,
                b_shared,
                q,
                w,
                asc.len(),
                ws.len(),
                k,
                row,
                col,
                t,
                0,
            );
        }
        let mut base = 0;
        while base < k {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let slot = base / 128 % 2;
            if base + 128 < k {
                unsafe {
                    stage_projection_tile::<true>(
                        a_shared.add((1 - slot) * 576),
                        b_shared.add((1 - slot) * 1152),
                        q,
                        w,
                        asc.len(),
                        ws.len(),
                        k,
                        row,
                        col,
                        t,
                        base + 128,
                    );
                }
            }
            let a_tile = unsafe { a_shared.add(slot * 576) };
            let b_tile = unsafe { b_shared.add(slot * 1152) };
            let mut j = 0;
            while j < 32 {
                unsafe {
                    // Each lane supplies an aligned row of one of four 8x8
                    // b16 matrices. Their bytes are the same INT8 MMA fragments.
                    let a = cuda_device::wmma::ldmatrix_x4(
                        a_tile.add((wr + lane % 16) * 36 + j + lane / 16 * 4),
                    );
                    let b = cuda_device::wmma::ldmatrix_x2(
                        b_tile.add((wc + lane % 8) * 36 + j + lane / 8 % 2 * 4),
                    );
                    let b0 = [b[0], b[1]];
                    c0 = cuda_device::wmma::mma_m16n8k32_s32_s8(c0, a, b0);
                }
                j += 8;
            }
            // No thread may overwrite staging before the last warp consumes it.
            thread::sync_threads();
            base += 128;
        }
        let mut i = 0;
        while i < 4 {
            let r = row + wr + group + i / 2 * 8;
            let c = col + wc + part * 2 + i % 2;
            if r < asc.len() {
                unsafe {
                    if c < ws.len() {
                        *out.get_unchecked_mut(r * ws.len() + c) =
                            bf(mul(mul(c0[i] as f32, asc[r]), ws[c]));
                    }
                }
            }
            i += 1;
        }
    }
    /// A 64x128 output tile cooperatively stages 128 K elements at a time.
    /// Eight warps each own a 32x32 tile (eight m16n8k32 MMAs).
    #[kernel]
    pub fn linear_large(
        amap: &[u64],
        asc: &[f32],
        bmap: &[u64],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut A: SharedArray<u32, 2048, 1024> = SharedArray::UNINIT;
        static mut B: SharedArray<u32, 4096, 1024> = SharedArray::UNINIT;
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tiles_n = ws.len().div_ceil(128);
        let tile = thread::blockIdx_x() as usize;
        let row = tile / tiles_n * 64;
        let col = tile % tiles_n * 128;
        let wr = warp_id / 4 * 32;
        let wc = warp_id % 4 * 32;
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let a_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut A) };
        let b_shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut B) };
        let mut c00 = [0_i32; 4];
        let mut c01 = [0_i32; 4];
        let mut c02 = [0_i32; 4];
        let mut c03 = [0_i32; 4];
        let mut c10 = [0_i32; 4];
        let mut c11 = [0_i32; 4];
        let mut c12 = [0_i32; 4];
        let mut c13 = [0_i32; 4];
        static mut BAR: SharedArray<u64, 1, 8> = SharedArray::UNINIT;
        let bar = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BAR) }
            .cast::<cuda_device::barrier::Barrier>();
        if t == 0 {
            unsafe {
                cuda_device::barrier::mbarrier_init(bar, 1);
                cuda_device::barrier::fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();
        let mut phase = 0;
        let mut base = 0;
        while base < k {
            if t == 0 {
                unsafe {
                    cuda_device::barrier::mbarrier_arrive_expect_tx(bar, 1, 24576);
                    tma_copy(
                        a_shared.cast::<u8>(),
                        amap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                        base as i32,
                        row as i32,
                        bar,
                    );
                    tma_copy(
                        b_shared.cast::<u8>(),
                        bmap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                        base as i32,
                        col as i32,
                        bar,
                    );
                }
            }
            while !unsafe { cuda_device::barrier::mbarrier_try_wait_parity(bar, phase) } {}
            thread::sync_threads();
            let mut j = 0;
            while j < 32 {
                unsafe {
                    let a0 = cuda_device::wmma::ldmatrix_x4(
                        a_shared.add(tma_word(wr + lane % 16, j + lane / 16 * 4)),
                    );
                    let a1 = cuda_device::wmma::ldmatrix_x4(
                        a_shared.add(tma_word(wr + 16 + lane % 16, j + lane / 16 * 4)),
                    );
                    let b0 = cuda_device::wmma::ldmatrix_x4(b_shared.add(tma_word(
                        wc + lane / 16 * 8 + lane % 8,
                        j + lane / 8 % 2 * 4,
                    )));
                    let b1 = cuda_device::wmma::ldmatrix_x4(b_shared.add(tma_word(
                        wc + 16 + lane / 16 * 8 + lane % 8,
                        j + lane / 8 % 2 * 4,
                    )));
                    c00 = cuda_device::wmma::mma_m16n8k32_s32_s8(c00, a0, [b0[0], b0[1]]);
                    c01 = cuda_device::wmma::mma_m16n8k32_s32_s8(c01, a0, [b0[2], b0[3]]);
                    c02 = cuda_device::wmma::mma_m16n8k32_s32_s8(c02, a0, [b1[0], b1[1]]);
                    c03 = cuda_device::wmma::mma_m16n8k32_s32_s8(c03, a0, [b1[2], b1[3]]);
                    c10 = cuda_device::wmma::mma_m16n8k32_s32_s8(c10, a1, [b0[0], b0[1]]);
                    c11 = cuda_device::wmma::mma_m16n8k32_s32_s8(c11, a1, [b0[2], b0[3]]);
                    c12 = cuda_device::wmma::mma_m16n8k32_s32_s8(c12, a1, [b1[0], b1[1]]);
                    c13 = cuda_device::wmma::mma_m16n8k32_s32_s8(c13, a1, [b1[2], b1[3]]);
                }
                j += 8;
            }
            // No thread may overwrite staging before the last warp consumes it.
            thread::sync_threads();
            phase ^= 1;
            base += 128;
        }
        if t == 0 {
            unsafe {
                cuda_device::barrier::mbarrier_inval(bar);
            }
        }
        let mut i = 0;
        while i < 4 {
            let r = row + wr + group + i / 2 * 8;
            let c = col + wc + part * 2 + i % 2;
            unsafe {
                if r + 0 < asc.len() && c + 0 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 0) =
                        bf(mul(mul(c00[i] as f32, asc[r + 0]), ws[c + 0]));
                }
                if r + 0 < asc.len() && c + 8 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 8) =
                        bf(mul(mul(c01[i] as f32, asc[r + 0]), ws[c + 8]));
                }
                if r + 0 < asc.len() && c + 16 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 16) =
                        bf(mul(mul(c02[i] as f32, asc[r + 0]), ws[c + 16]));
                }
                if r + 0 < asc.len() && c + 24 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 24) =
                        bf(mul(mul(c03[i] as f32, asc[r + 0]), ws[c + 24]));
                }
                if r + 16 < asc.len() && c + 0 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 0) =
                        bf(mul(mul(c10[i] as f32, asc[r + 16]), ws[c + 0]));
                }
                if r + 16 < asc.len() && c + 8 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 8) =
                        bf(mul(mul(c11[i] as f32, asc[r + 16]), ws[c + 8]));
                }
                if r + 16 < asc.len() && c + 16 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 16) =
                        bf(mul(mul(c12[i] as f32, asc[r + 16]), ws[c + 16]));
                }
                if r + 16 < asc.len() && c + 24 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 24) =
                        bf(mul(mul(c13[i] as f32, asc[r + 16]), ws[c + 24]));
                }
            }
            i += 1;
        }
    }
    /// In-place head RMSNorm and RoPE with the original sum/rounding boundaries.
    #[kernel]
    pub fn norm_rope(
        mut x: DisjointSlice<f32>,
        gain: &[f32],
        table: &[f32],
        heads: u32,
        position: &[u32],
    ) {
        static mut SUM: SharedArray<f32, 128> = SharedArray::UNINIT;
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUM) };
        let h = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let value = unsafe { *x.get_unchecked_mut(h * 128 + t) };
        // Original norm used 256 slots; its upper half contained positive zero.
        unsafe {
            *p.add(t) = add(add(0.0, mul(value, value)), 0.0);
        }
        thread::sync_threads();
        if t < 32 {
            let mut sum = unsafe {
                add(
                    add(*p.add(t), *p.add(t + 64)),
                    add(*p.add(t + 32), *p.add(t + 96)),
                )
            };
            let mut d = 16;
            while d > 0 {
                sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
                d /= 2;
            }
            if t == 0 {
                unsafe {
                    *p = sum;
                }
            }
        }
        thread::sync_threads();
        let inv = rsqrt(add(div(unsafe { *p }, 128.0), 1e-6));
        let normalized = bf(mul(mul(value, inv), gain[t]));
        // All threads consume the sum before this shared allocation is reused.
        thread::sync_threads();
        unsafe {
            *p.add(t) = normalized;
        }
        thread::sync_threads();
        if t < 64 {
            let row = h / heads as usize + position[1] as usize;
            let a = unsafe { *p.add(t) };
            let b = unsafe { *p.add(t + 64) };
            let c = table[row * 128 + t];
            let s = table[row * 128 + t + 64];
            unsafe {
                *x.get_unchecked_mut(h * 128 + t) = bf(add(mul(a, c), -mul(b, s)));
                *x.get_unchecked_mut(h * 128 + t + 64) = bf(add(mul(b, c), mul(a, s)));
            }
        }
    }
    /// Q/K normalization, rotary embedding, and KV append share one launch.
    #[kernel]
    pub fn prepare_qk(
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        v: &[f32],
        qgain: &[f32],
        kgain: &[f32],
        table: &[f32],
        position: &[u32],
        mut ck: DisjointSlice<u16>,
        mut cv: DisjointSlice<u16>,
    ) {
        static mut SUM: SharedArray<f32, 128> = SharedArray::UNINIT;
        dependency_wait();
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUM) };
        let block = thread::blockIdx_x() as usize;
        let row = block / 40;
        let h = block % 40;
        let is_q = h < 32;
        let head = if is_q { h } else { h - 32 };
        let t = thread::threadIdx_x() as usize;
        let start = row * (if is_q { 4096 } else { 1024 }) + head * 128;
        let value = if is_q {
            unsafe { *q.get_unchecked_mut(start + t) }
        } else {
            unsafe { *k.get_unchecked_mut(start + t) }
        };
        // Gains and the RoPE row are cold in L2; load them before the reduction.
        let pos = row + position[1] as usize;
        let gain = if is_q { qgain[t] } else { kgain[t] };
        let (c, s) = if t < 64 {
            (table[pos * 128 + t], table[pos * 128 + t + 64])
        } else {
            (0.0, 0.0)
        };
        // Original norm used 256 slots; its upper half contained positive zero.
        unsafe {
            *p.add(t) = add(add(0.0, mul(value, value)), 0.0);
        }
        thread::sync_threads();
        if t < 32 {
            let mut sum = unsafe {
                add(
                    add(*p.add(t), *p.add(t + 64)),
                    add(*p.add(t + 32), *p.add(t + 96)),
                )
            };
            let mut d = 16;
            while d > 0 {
                sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
                d /= 2;
            }
            if t == 0 {
                unsafe {
                    *p = sum;
                }
            }
        }
        thread::sync_threads();
        let inv = rsqrt(add(div(unsafe { *p }, 128.0), 1e-6));
        let normalized = bf(mul(mul(value, inv), gain));
        // All threads consume the sum before this shared allocation is reused.
        thread::sync_threads();
        unsafe {
            *p.add(t) = normalized;
        }
        thread::sync_threads();
        if t < 64 {
            let a = unsafe { *p.add(t) };
            let b = unsafe { *p.add(t + 64) };
            let first = bf(add(mul(a, c), -mul(b, s)));
            let second = bf(add(mul(b, c), mul(a, s)));
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(start + t) = first;
                    *q.get_unchecked_mut(start + t + 64) = second;
                } else {
                    *k.get_unchecked_mut(start + t) = first;
                    *k.get_unchecked_mut(start + t + 64) = second;
                    *ck.get_unchecked_mut((head * 128 + t) * 4096 + pos) =
                        (first.to_bits() >> 16) as u16;
                    *ck.get_unchecked_mut((head * 128 + t + 64) * 4096 + pos) =
                        (second.to_bits() >> 16) as u16;
                }
            }
        }
        if !is_q {
            // Each K head also appends its matching V coordinates exactly once.
            unsafe {
                *cv.get_unchecked_mut((row + position[1] as usize) * 1024 + head * 128 + t) =
                    (v[start + t].to_bits() >> 16) as u16;
            }
        }
    }
    #[kernel]
    pub fn rope(
        x: &[f32],
        table: &[f32],
        heads: u32,
        position: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        let position = position[1];
        let i = thread::index_1d().get();
        if i >= x.len() / 2 {
            return;
        }
        let h = i / 64;
        let j = i % 64;
        let p = h / heads as usize + position as usize;
        let a = x[h * 128 + j];
        let b = x[h * 128 + j + 64];
        let c = table[p * 128 + j];
        let s = table[p * 128 + j + 64];
        // SAFETY: disjoint coordinate pairs within validated heads/positions.
        unsafe {
            *out.get_unchecked_mut(h * 128 + j) = bf(add(mul(a, c), -mul(b, s)));
            *out.get_unchecked_mut(h * 128 + j + 64) = bf(add(mul(b, c), mul(a, s)));
        }
    }
    #[kernel]
    pub fn cache(
        k: &[f32],
        v: &[f32],
        position: &[u32],
        mut ck: DisjointSlice<u16>,
        mut cv: DisjointSlice<u16>,
    ) {
        let position = position[1];
        let i = thread::index_1d().get();
        if i >= k.len() {
            return;
        }
        // SAFETY: admitted cache capacity and one writer per appended coordinate.
        unsafe {
            *ck.get_unchecked_mut(i % 1024 * 4096 + position as usize + i / 1024) =
                (k[i].to_bits() >> 16) as u16;
            *cv.get_unchecked_mut(position as usize * 1024 + i) = (v[i].to_bits() >> 16) as u16;
        }
    }
    #[kernel]
    pub fn attention(
        q: &[f32],
        k: &[u16],
        v: &[u16],
        position: &[u32],
        mut out: DisjointSlice<f32>,
        pf: &[i8],
        pf_from: u32,
        pf_to: u32,
    ) {
        if prefetch_share(pf.as_ptr().cast(), pf.len(), pf_from, pf_to, q.len() / 128) {
            return;
        }
        dependency_wait();
        let position = position[1];
        static mut SCORE: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut QUERY: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let query = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let row = query / 32;
        let head = query % 32;
        let kv = head / 4;
        let end = position as usize + row + 1;
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        let reduce = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE) };
        let query_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        unsafe {
            *query_values.add(t) = q[query * 128 + t];
        }
        let mut maximum = f32::NEG_INFINITY;
        let mut base = 0;
        while base < end {
            // A 64-key tile spans complete, aligned rows of the fixed-capacity
            // cache. Future keys may be staged, but only visible keys are used.
            unsafe {
                let mut piece = 0;
                while piece < 8 {
                    let i = t + piece * 128;
                    cuda_device::async_copy::cp_async_ca_16(
                        tile.add(i * 4),
                        k.as_ptr()
                            .add((kv * 128 + i / 8) * 4096 + base + i % 8 * 8)
                            .cast::<u32>(),
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let j = base + t % 64;
            // Preserve each thread's original t, t+128, ... maximum sequence.
            if t / 64 == base / 64 % 2 && j < end {
                let mut sum = 0.0;
                let mut c = 0;
                while c < 128 {
                    sum = add(
                        sum,
                        mul(
                            unsafe { *query_values.add(c) },
                            widen(unsafe { *halves.add(c * 64 + t % 64) }),
                        ),
                    );
                    c += 1;
                }
                let s = mul(sum, 0.08838834764831845_f32);
                maximum = maximum.max(s);
                unsafe {
                    *scores.add(j) = s;
                }
            }
            thread::sync_threads();
            base += 64;
        }
        unsafe {
            *reduce.add(t) = maximum;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = (*reduce.add(t)).max(*reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let maximum = unsafe { *reduce };
        let mut total = 0.0;
        let mut j = t;
        while j < end {
            let s = exp(add(unsafe { *scores.add(j) }, -maximum));
            unsafe {
                *scores.add(j) = s;
            }
            total = add(total, s);
            j += 128;
        }
        // All threads must finish reading maximum before reusing the reduction array.
        thread::sync_threads();
        unsafe {
            *reduce.add(t) = total;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = add(*reduce.add(t), *reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let den = unsafe { *reduce };
        // Each normalized score is shared by all 128 output coordinates. Compute
        // the same RN division once per key, preserving the PV sum's order.
        let mut j = t;
        while j < end {
            unsafe {
                *scores.add(j) = div(*scores.add(j), den);
            }
            j += 128;
        }
        thread::sync_threads();
        let mut sum = 0.0;
        let mut base = 0;
        while base < end {
            // Reuse the staging allocation for a token-major 64x128 V tile.
            unsafe {
                let mut piece = 0;
                while piece < 8 {
                    let i = t + piece * 128;
                    cuda_device::async_copy::cp_async_ca_16(
                        tile.add(i * 4),
                        v.as_ptr()
                            .add(((base + i / 16) * 8 + kv) * 128 + i % 16 * 8)
                            .cast::<u32>(),
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                sum = add(
                    sum,
                    mul(
                        unsafe { *scores.add(base + j) },
                        widen(unsafe { *halves.add(j * 128 + t) }),
                    ),
                );
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        // SAFETY: one 128-thread block owns one query/head; one coordinate per thread.
        unsafe {
            *out.get_unchecked_mut(query * 128 + t) = bf(sum);
        }
    }
    /// Four query heads share their common KV tiles. Host admits at most 512
    /// visible tokens, so the four independent score arrays fit in shared memory.
    #[kernel]
    pub fn attention_prefill(
        q: &[f32],
        k: &[u16],
        v: &[u16],
        position: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        let position = position[1];
        static mut SCORE: SharedArray<f32, 2048> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut QUERY: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let cohort = tid / 128;
        let t = tid % 128;
        let query = thread::blockIdx_x() as usize * 4 + cohort;
        let row = query / 32;
        let head = query % 32;
        let kv = head / 4;
        let end = position as usize + row + 1;
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE).add(cohort * 512) };
        let reduce = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE).add(cohort * 128) };
        let query_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY).add(cohort * 128) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        unsafe {
            *query_values.add(t) = q[query * 128 + t];
        }
        let mut maximum = f32::NEG_INFINITY;
        let mut base = 0;
        while base < end {
            // A 64-key tile spans complete, aligned rows of the fixed-capacity
            // cache. Future keys may be staged, but only visible keys are used.
            unsafe {
                let mut piece = 0;
                while piece < 2 {
                    let i = tid + piece * 512;
                    cuda_device::async_copy::cp_async_ca_16(
                        tile.add(i * 4),
                        k.as_ptr()
                            .add((kv * 128 + i / 8) * 4096 + base + i % 8 * 8)
                            .cast::<u32>(),
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let j = base + t % 64;
            // Preserve each thread's original t, t+128, ... maximum sequence.
            if t / 64 == base / 64 % 2 && j < end {
                let mut sum = 0.0;
                let mut c = 0;
                while c < 128 {
                    sum = add(
                        sum,
                        mul(
                            unsafe { *query_values.add(c) },
                            widen(unsafe { *halves.add(c * 64 + t % 64) }),
                        ),
                    );
                    c += 1;
                }
                let s = mul(sum, 0.08838834764831845_f32);
                maximum = maximum.max(s);
                unsafe {
                    *scores.add(j) = s;
                }
            }
            thread::sync_threads();
            base += 64;
        }
        unsafe {
            *reduce.add(t) = maximum;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = (*reduce.add(t)).max(*reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let maximum = unsafe { *reduce };
        let mut total = 0.0;
        let mut j = t;
        while j < end {
            let s = exp(add(unsafe { *scores.add(j) }, -maximum));
            unsafe {
                *scores.add(j) = s;
            }
            total = add(total, s);
            j += 128;
        }
        // All threads must finish reading maximum before reusing the reduction array.
        thread::sync_threads();
        unsafe {
            *reduce.add(t) = total;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = add(*reduce.add(t), *reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let den = unsafe { *reduce };
        // Each normalized score is shared by all 128 output coordinates. Compute
        // the same RN division once per key, preserving the PV sum's order.
        let mut j = t;
        while j < end {
            unsafe {
                *scores.add(j) = div(*scores.add(j), den);
            }
            j += 128;
        }
        thread::sync_threads();
        let mut sum = 0.0;
        let mut base = 0;
        while base < end {
            // Reuse the staging allocation for a token-major 64x128 V tile.
            unsafe {
                let mut piece = 0;
                while piece < 2 {
                    let i = tid + piece * 512;
                    cuda_device::async_copy::cp_async_ca_16(
                        tile.add(i * 4),
                        v.as_ptr()
                            .add(((base + i / 16) * 8 + kv) * 128 + i % 16 * 8)
                            .cast::<u32>(),
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                sum = add(
                    sum,
                    mul(
                        unsafe { *scores.add(base + j) },
                        widen(unsafe { *halves.add(j * 128 + t) }),
                    ),
                );
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        // SAFETY: one 128-thread block owns one query/head; one coordinate per thread.
        unsafe {
            *out.get_unchecked_mut(query * 128 + t) = bf(sum);
        }
    }
    /// Independent 64-key tiles expose QK parallelism across the GPU.
    #[kernel]
    pub fn attention_scores(
        q: &[f32],
        k: &[u16],
        position: &[u32],
        tiles: u32,
        mut scores: DisjointSlice<f32>,
    ) {
        dependency_wait();
        let position = position[1];
        static mut QUERY: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let query = block / tiles as usize;
        let base = block % tiles as usize * 64;
        let t = thread::threadIdx_x() as usize;
        let end = position as usize + query / 32 + 1;
        if base >= end {
            return;
        }
        let kv = query % 32 / 4;
        let query_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        unsafe {
            *query_values.add(t) = q[query * 128 + t];
            let mut piece = 0;
            while piece < 8 {
                let i = t + piece * 128;
                cuda_device::async_copy::cp_async_ca_16(
                    tile.add(i * 4),
                    k.as_ptr()
                        .add((kv * 128 + i / 8) * 4096 + base + i % 8 * 8)
                        .cast::<u32>(),
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        if t < 64 && base + t < end {
            let mut sum = 0.0;
            let mut c = 0;
            while c < 128 {
                sum = add(
                    sum,
                    mul(
                        unsafe { *query_values.add(c) },
                        widen(unsafe { *halves.add(c * 64 + t) }),
                    ),
                );
                c += 1;
            }
            unsafe {
                *scores.get_unchecked_mut(query * 4096 + base + t) =
                    mul(sum, 0.08838834764831845_f32);
            }
        }
    }
    /// Preserve the original per-thread softmax tree and ascending PV sum order.
    #[kernel]
    pub fn attention_finish(
        input_scores: &[f32],
        v: &[u16],
        position: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        let position = position[1];
        static mut SCORE: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let query = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let kv = query % 32 / 4;
        let end = position as usize + query / 32 + 1;
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        let reduce = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        let mut maximum = f32::NEG_INFINITY;
        let mut j = t;
        while j < end {
            let s = input_scores[query * 4096 + j];
            unsafe {
                *scores.add(j) = s;
            }
            maximum = maximum.max(s);
            j += 128;
        }
        unsafe {
            *reduce.add(t) = maximum;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = (*reduce.add(t)).max(*reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let maximum = unsafe { *reduce };
        let mut total = 0.0;
        let mut j = t;
        while j < end {
            let s = exp(add(unsafe { *scores.add(j) }, -maximum));
            unsafe {
                *scores.add(j) = s;
            }
            total = add(total, s);
            j += 128;
        }
        // All threads must finish reading maximum before reusing the reduction array.
        thread::sync_threads();
        unsafe {
            *reduce.add(t) = total;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = add(*reduce.add(t), *reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let den = unsafe { *reduce };
        // Each normalized score is shared by all 128 output coordinates. Compute
        // the same RN division once per key, preserving the PV sum's order.
        let mut j = t;
        while j < end {
            unsafe {
                *scores.add(j) = div(*scores.add(j), den);
            }
            j += 128;
        }
        thread::sync_threads();
        let mut sum = 0.0;
        let mut base = 0;
        while base < end {
            // Reuse the staging allocation for a token-major 64x128 V tile.
            unsafe {
                let mut piece = 0;
                while piece < 8 {
                    let i = t + piece * 128;
                    cuda_device::async_copy::cp_async_ca_16(
                        tile.add(i * 4),
                        v.as_ptr()
                            .add(((base + i / 16) * 8 + kv) * 128 + i % 16 * 8)
                            .cast::<u32>(),
                    );
                    piece += 1;
                }
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                sum = add(
                    sum,
                    mul(
                        unsafe { *scores.add(base + j) },
                        widen(unsafe { *halves.add(j * 128 + t) }),
                    ),
                );
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        // SAFETY: one 128-thread block owns one query/head; one coordinate per thread.
        unsafe {
            *out.get_unchecked_mut(query * 128 + t) = bf(sum);
        }
    }
    #[kernel]
    pub fn attention_softmax(
        mut input_scores: DisjointSlice<f32>,
        position: &[u32],
        pf: &[i8],
        pf_from: u32,
        pf_to: u32,
    ) {
        let real = input_scores.len() / 4096;
        if prefetch_share(pf.as_ptr().cast(), pf.len(), pf_from, pf_to, real) {
            return;
        }
        dependency_wait();
        let position = position[1];
        static mut SCORE: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut REDUCE: SharedArray<f32, 128> = SharedArray::UNINIT;
        let query = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let end = position as usize + query / 32 + 1;
        let scores = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        let reduce = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE) };
        let mut maximum = f32::NEG_INFINITY;
        let mut j = t;
        while j < end {
            let s = unsafe { *input_scores.get_unchecked_mut(query * 4096 + j) };
            unsafe {
                *scores.add(j) = s;
            }
            maximum = maximum.max(s);
            j += 128;
        }
        unsafe {
            *reduce.add(t) = maximum;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = (*reduce.add(t)).max(*reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let maximum = unsafe { *reduce };
        let mut total = 0.0;
        let mut j = t;
        while j < end {
            let s = exp(add(unsafe { *scores.add(j) }, -maximum));
            unsafe {
                *scores.add(j) = s;
            }
            total = add(total, s);
            j += 128;
        }
        // All threads must finish reading maximum before reusing the reduction array.
        thread::sync_threads();
        unsafe {
            *reduce.add(t) = total;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = add(*reduce.add(t), *reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let den = unsafe { *reduce };
        // Each normalized score is shared by all 128 output coordinates. Compute
        // the same RN division once per key, preserving the PV sum's order.
        let mut j = t;
        while j < end {
            unsafe {
                *input_scores.get_unchecked_mut(query * 4096 + j) = div(*scores.add(j), den);
            }
            j += 128;
        }
    }
    /// Four independent coordinate blocks per query/head, preserving key order.
    #[kernel]
    pub fn attention_values(
        scores: &[f32],
        v: &[u16],
        position: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut TILE: SharedArray<u32, 2048, 16> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, 128> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let query = block / 4;
        let column = block % 4 * 32;
        let t = thread::threadIdx_x() as usize;
        let end = position[1] as usize + query / 32 + 1;
        let kv = query % 32 / 4;
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        let probability = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PROB) };
        // Preload the first tile. Subsequent tiles are copied while the current
        // tile's ordered dot product runs; the two staging slots never alias.
        stage_values(tile, probability, scores, v, query, kv, column, t, 0, end);
        let mut sum = 0.0;
        let mut base = 0;
        while base < end {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            if base + 64 < end {
                stage_values(
                    tile,
                    probability,
                    scores,
                    v,
                    query,
                    kv,
                    column,
                    t,
                    base + 64,
                    end,
                );
            }
            let slot = base / 64 % 2;
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                sum = add(
                    sum,
                    mul(
                        unsafe { *probability.add(slot * 64 + j) },
                        widen(unsafe { *halves.add(slot * 2048 + j * 32 + t) }),
                    ),
                );
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        unsafe {
            *out.get_unchecked_mut(query * 128 + column + t) = bf(sum);
        }
    }
    #[kernel]
    pub fn residual(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < a.len() {
            unsafe {
                *out.get_unchecked_mut(i) = bf(add(a[i], b[i]));
            }
        }
    }
    #[kernel]
    pub fn swiglu(
        gate: &[f32],
        up: &[f32],
        mut out: DisjointSlice<f32>,
        mut maxima: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i < gate.len() {
            let g = gate[i];
            let value = bf(mul(div(g, add(1.0, exp(-g))), up[i]));
            unsafe {
                *out.get_unchecked_mut(i) = value;
            }
            // Decode has a full 12288-coordinate row: every warp is complete.
            if gate.len() == 12288 {
                publish_absmax(value, maxima.as_mut_ptr());
            }
        }
    }
    /// Lexicographic (largest finite value, lowest token index) reduction.
    #[kernel]
    pub fn select_token(x: &[f32], mut out: DisjointSlice<u32>) {
        static mut PARTIAL: SharedArray<u32, 96> = SharedArray::UNINIT;
        dependency_wait();
        let t = thread::threadIdx_x() as usize;
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let mut value = f32::NEG_INFINITY;
        let mut index = u32::MAX;
        let mut bad = if x.len() == 0 { 1 } else { 0 };
        let mut j = t;
        while j < x.len() {
            let v = x[j];
            if !v.is_finite() {
                bad = 1;
            }
            if v > value || (v == value && (j as u32) < index) {
                value = v;
                index = j as u32;
            }
            j += 1024;
        }
        let (value, index, bad) = best_warp(value, index, bad);
        if t % 32 == 0 {
            unsafe {
                *p.add(t / 32) = value.to_bits();
                *p.add(32 + t / 32) = index;
                *p.add(64 + t / 32) = bad;
            }
        }
        thread::sync_threads();
        if t < 32 {
            let (_, index, bad) =
                unsafe { best_warp(f32::from_bits(*p.add(t)), *p.add(32 + t), *p.add(64 + t)) };
            if t == 0 {
                unsafe {
                    *out.get_unchecked_mut(0) = if bad == 0 { index } else { u32::MAX };
                }
            }
        }
    }
    #[kernel]
    pub fn head(x: &[f32], w: &[u16], mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let row = tid / 32;
        if row >= out.len() {
            return;
        }
        let start = x.len() - 4096;
        let mut sum = 0.0;
        let mut j = lane;
        while j < 4096 {
            sum = add(sum, mul(x[start + j], widen(w[row * 4096 + j])));
            j += 32;
        }
        let mut d = 16;
        while d > 0 {
            sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
            d /= 2;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row) = bf(sum);
            }
        }
    }
    #[kernel]
    pub fn head_packed(
        x: &[f32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        mut out: DisjointSlice<f32>,
        prefetch: u32,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let row = tid / 32;
        if row < out.len() {
            // Sixteen 96-word chunks form each packed row's contiguous 6144 bytes.
            prefetch_row(
                unsafe { words.as_ptr().add(row * 1536) }.cast(),
                6144,
                prefetch,
                lane,
            );
        }
        dependency_wait();
        if row >= out.len() {
            return;
        }
        let start = x.len() - 4096;
        let mut sum = 0.0;
        // Three coalesced word planes encode eight successive terms per lane.
        // Keep the original lane-strided FP32 accumulation and warp sum order.
        let mut group = 0;
        while group < 16 {
            let chunk = row * 16 + group;
            let word = chunk * 96 + lane;
            let w0 = words[word];
            let w1 = words[word + 32];
            let w2 = words[word + 64];
            let j = group * 256 + lane;
            sum = add(
                sum,
                mul(
                    x[start + j],
                    widen(unpack_code(w0, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 32],
                    widen(unpack_code(w0 >> 12, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 64],
                    widen(unpack_code(
                        (w0 >> 24) | (w1 << 8),
                        offsets,
                        exceptions,
                        chunk,
                    )),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 96],
                    widen(unpack_code(w1 >> 4, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 128],
                    widen(unpack_code(w1 >> 16, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 160],
                    widen(unpack_code(
                        (w1 >> 28) | (w2 << 4),
                        offsets,
                        exceptions,
                        chunk,
                    )),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 192],
                    widen(unpack_code(w2 >> 8, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 224],
                    widen(unpack_code(w2 >> 20, offsets, exceptions, chunk)),
                ),
            );
            group += 1;
        }
        let mut d = 16;
        while d > 0 {
            sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
            d /= 2;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(row) = bf(sum);
            }
        }
    }
    /// Exhaustive bit-format conformance probe; inference uses head_packed.
    #[kernel]
    pub fn unpack_head(
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        mut out: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i < out.len() {
            unsafe {
                *out.get_unchecked_mut(i) = unpack_bf16(words, offsets, exceptions, i);
            }
        }
    }
    // Paged, ragged sequence kernels. Host validation owns page-table safety.
    #[kernel]
    pub fn prepare_qk_paged(
        mut q: DisjointSlice<f32>,
        mut k: DisjointSlice<f32>,
        v: &[f32],
        qgain: &[f32],
        kgain: &[f32],
        table: &[f32],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        mut ck: DisjointSlice<u16>,
        mut cv: DisjointSlice<u16>,
    ) {
        static mut SUM: SharedArray<f32, 128> = SharedArray::UNINIT;
        dependency_wait();
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUM) };
        let block = thread::blockIdx_x() as usize;
        let row = block / 40;
        let h = block % 40;
        let is_q = h < 32;
        let head = if is_q { h } else { h - 32 };
        let t = thread::threadIdx_x() as usize;
        let start = row * (if is_q { 4096 } else { 1024 }) + head * 128;
        let value = if is_q {
            unsafe { *q.get_unchecked_mut(start + t) }
        } else {
            unsafe { *k.get_unchecked_mut(start + t) }
        };
        // Gains and the RoPE row are cold in L2; load them before the reduction.
        let pos = row_info[row * 2 + 1] as usize;
        let slot = row_info[row * 2] as usize;
        let page = page_table[slot * pages_per_sequence as usize + pos / 64] as usize;
        let within = pos % 64;
        let gain = if is_q { qgain[t] } else { kgain[t] };
        let (c, s) = if t < 64 {
            (table[pos * 128 + t], table[pos * 128 + t + 64])
        } else {
            (0.0, 0.0)
        };
        // Original norm used 256 slots; its upper half contained positive zero.
        unsafe {
            *p.add(t) = add(add(0.0, mul(value, value)), 0.0);
        }
        thread::sync_threads();
        if t < 32 {
            let mut sum = unsafe {
                add(
                    add(*p.add(t), *p.add(t + 64)),
                    add(*p.add(t + 32), *p.add(t + 96)),
                )
            };
            let mut d = 16;
            while d > 0 {
                sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
                d /= 2;
            }
            if t == 0 {
                unsafe {
                    *p = sum;
                }
            }
        }
        thread::sync_threads();
        let inv = rsqrt(add(div(unsafe { *p }, 128.0), 1e-6));
        let normalized = bf(mul(mul(value, inv), gain));
        // All threads consume the sum before this shared allocation is reused.
        thread::sync_threads();
        unsafe {
            *p.add(t) = normalized;
        }
        thread::sync_threads();
        if t < 64 {
            let a = unsafe { *p.add(t) };
            let b = unsafe { *p.add(t + 64) };
            let first = bf(add(mul(a, c), -mul(b, s)));
            let second = bf(add(mul(b, c), mul(a, s)));
            unsafe {
                if is_q {
                    *q.get_unchecked_mut(start + t) = first;
                    *q.get_unchecked_mut(start + t + 64) = second;
                } else {
                    *k.get_unchecked_mut(start + t) = first;
                    *k.get_unchecked_mut(start + t + 64) = second;
                    *ck.get_unchecked_mut(page * 65536 + (head * 128 + t) * 64 + within) =
                        (first.to_bits() >> 16) as u16;
                    *ck.get_unchecked_mut(page * 65536 + (head * 128 + t + 64) * 64 + within) =
                        (second.to_bits() >> 16) as u16;
                }
            }
        }
        if !is_q {
            // Each K head also appends its matching V coordinates exactly once.
            unsafe {
                *cv.get_unchecked_mut((page * 64 + within) * 1024 + head * 128 + t) =
                    (v[start + t].to_bits() >> 16) as u16;
            }
        }
    }
    #[kernel]
    pub fn attention_scores_paged(
        q: &[f32],
        k: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        tiles: u32,
        mut scores: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut QUERY: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let query = block / tiles as usize;
        let base = block % tiles as usize * 64;
        let t = thread::threadIdx_x() as usize;
        let end = row_info[query / 32 * 2 + 1] as usize + 1;
        if base >= end {
            return;
        }
        let slot = row_info[query / 32 * 2] as usize;
        let page = page_table[slot * pages_per_sequence as usize + base / 64] as usize;
        let kv = query % 32 / 4;
        let query_values = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        unsafe {
            *query_values.add(t) = q[query * 128 + t];
            let mut piece = 0;
            while piece < 8 {
                let i = t + piece * 128;
                cuda_device::async_copy::cp_async_ca_16(
                    tile.add(i * 4),
                    k.as_ptr()
                        .add(page * 65536 + (kv * 128 + i / 8) * 64 + i % 8 * 8)
                        .cast::<u32>(),
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        if t < 64 && base + t < end {
            let mut sum = 0.0;
            let mut c = 0;
            while c < 128 {
                sum = add(
                    sum,
                    mul(
                        unsafe { *query_values.add(c) },
                        widen(unsafe { *halves.add(c * 64 + t) }),
                    ),
                );
                c += 1;
            }
            unsafe {
                *scores.get_unchecked_mut(query * stride as usize + base + t) =
                    mul(sum, 0.08838834764831845_f32);
            }
        }
    }

    /// Four GQA heads share each staged key tile while retaining each dot's order.
    #[kernel]
    pub fn attention_scores_grouped(
        q: &[f32],
        k: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        tiles: u32,
        mut scores: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut QUERY: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let group = block / tiles as usize;
        let query = group * 4;
        let base = block % tiles as usize * 64;
        let t = thread::threadIdx_x() as usize;
        let end = row_info[group / 8 * 2 + 1] as usize + 1;
        if base >= end {
            return;
        }
        let slot = row_info[group / 8 * 2] as usize;
        let page = page_table[slot * pages_per_sequence as usize + base / 64] as usize;
        let kv = group % 8;
        let queries = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        // SAFETY: 128 threads stage four complete heads and one 64-token key
        // page. The table and causal tail have the same bounds as the scalar
        // paged path. All threads wait before any staged value is consumed.
        unsafe {
            let mut head = 0;
            while head < 4 {
                *queries.add(head * 128 + t) = q[(query + head) * 128 + t];
                head += 1;
            }
            let mut piece = 0;
            while piece < 8 {
                let i = t + piece * 128;
                cuda_device::async_copy::cp_async_ca_16(
                    tile.add(i * 4),
                    k.as_ptr()
                        .add(page * 65536 + (kv * 128 + i / 8) * 64 + i % 8 * 8)
                        .cast::<u32>(),
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        if t < 64 && base + t < end {
            let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
            let mut c = 0;
            while c < 128 {
                let key = widen(unsafe { *halves.add(c * 64 + t) });
                s0 = add(s0, mul(unsafe { *queries.add(c) }, key));
                s1 = add(s1, mul(unsafe { *queries.add(128 + c) }, key));
                s2 = add(s2, mul(unsafe { *queries.add(256 + c) }, key));
                s3 = add(s3, mul(unsafe { *queries.add(384 + c) }, key));
                c += 1;
            }
            let offset = query * stride as usize + base + t;
            unsafe {
                *scores.get_unchecked_mut(offset) = mul(s0, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + stride as usize) =
                    mul(s1, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + 2 * stride as usize) =
                    mul(s2, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + 3 * stride as usize) =
                    mul(s3, 0.08838834764831845_f32);
            }
        }
    }

    /// Fixed per-query softmax tree, independent of cache layout and batch size.
    #[kernel]
    pub fn attention_softmax_paged(mut scores: DisjointSlice<f32>, row_info: &[u32], stride: u32) {
        dependency_wait();
        static mut REDUCE: SharedArray<f32, 128> = SharedArray::UNINIT;
        let query = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let end = row_info[query / 32 * 2 + 1] as usize + 1;
        let offset = query * stride as usize;
        let reduce = unsafe { SharedArray::as_raw_mut_ptr(&raw mut REDUCE) };
        let mut maximum = f32::NEG_INFINITY;
        let mut j = t;
        while j < end {
            maximum = maximum.max(unsafe { *scores.get_unchecked_mut(offset + j) });
            j += 128;
        }
        unsafe {
            *reduce.add(t) = maximum;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = (*reduce.add(t)).max(*reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let maximum = unsafe { *reduce };
        let mut total = 0.0;
        let mut j = t;
        while j < end {
            let s = exp(add(
                unsafe { *scores.get_unchecked_mut(offset + j) },
                -maximum,
            ));
            unsafe {
                *scores.get_unchecked_mut(offset + j) = s;
            }
            total = add(total, s);
            j += 128;
        }
        thread::sync_threads();
        unsafe {
            *reduce.add(t) = total;
        }
        thread::sync_threads();
        let mut d = 64;
        while d > 0 {
            if t < d {
                unsafe {
                    *reduce.add(t) = add(*reduce.add(t), *reduce.add(t + d));
                }
            }
            thread::sync_threads();
            d /= 2;
        }
        let denominator = unsafe { *reduce };
        let mut j = t;
        while j < end {
            unsafe {
                *scores.get_unchecked_mut(offset + j) =
                    div(*scores.get_unchecked_mut(offset + j), denominator);
            }
            j += 128;
        }
    }

    /// Ordered PV accumulation with a page lookup per 64-token block.
    #[kernel]
    pub fn attention_values_paged(
        scores: &[f32],
        v: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        mut out: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut TILE: SharedArray<u32, 2048, 16> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, 128> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let query = block / 4;
        let column = block % 4 * 32;
        let t = thread::threadIdx_x() as usize;
        let slot = row_info[query / 32 * 2] as usize;
        let end = row_info[query / 32 * 2 + 1] as usize + 1;
        let kv = query % 32 / 4;
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        let probability = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PROB) };
        stage_values_paged(
            tile,
            probability,
            scores,
            v,
            query,
            stride as usize,
            page_table[slot * pages_per_sequence as usize] as usize,
            kv,
            column,
            t,
            0,
            end,
        );
        let mut sum = 0.0;
        let mut base = 0;
        while base < end {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            if base + 64 < end {
                let page = page_table[slot * pages_per_sequence as usize + base / 64 + 1] as usize;
                stage_values_paged(
                    tile,
                    probability,
                    scores,
                    v,
                    query,
                    stride as usize,
                    page,
                    kv,
                    column,
                    t,
                    base + 64,
                    end,
                );
            }
            let shared_slot = base / 64 % 2;
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                sum = add(
                    sum,
                    mul(
                        unsafe { *probability.add(shared_slot * 64 + j) },
                        widen(unsafe { *halves.add(shared_slot * 2048 + j * 32 + t) }),
                    ),
                );
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        unsafe {
            *out.get_unchecked_mut(query * 128 + column + t) = bf(sum);
        }
    }
    /// Four prompt rows share a 64-key tile. A pair of warps owns each row's
    /// four GQA dots; all 256 threads contribute to the shared Q/K staging.
    #[kernel]
    pub fn attention_scores_tiled(
        q: &[f32],
        k: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        tiles: u32,
        query_tiles: &[u32],
        mut scores: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut QUERY: SharedArray<f32, 2048> = SharedArray::UNINIT;
        static mut TILE: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let group = block / tiles as usize;
        let first_row = query_tiles[group / 8 * 2] as usize;
        let count = query_tiles[group / 8 * 2 + 1] as usize;
        let kv = group % 8;
        let base = block % tiles as usize * 64;
        let max_end = row_info[(first_row + count - 1) * 2 + 1] as usize + 1;
        if base >= max_end {
            return;
        }
        let slot = row_info[first_row * 2] as usize;
        let page = page_table[slot * pages_per_sequence as usize + base / 64] as usize;
        let t = thread::threadIdx_x() as usize;
        let member = t / 64;
        let row = first_row + member.min(count - 1);
        let end = row_info[row * 2 + 1] as usize + 1;
        let key_index = t % 64;
        let queries = unsafe { SharedArray::as_raw_mut_ptr(&raw mut QUERY) };
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        unsafe {
            let mut piece = 0;
            while piece < 8 {
                let i = t + piece * 256;
                let input_row = first_row + (i / 512).min(count - 1);
                *queries.add(i) = q[(input_row * 32 + kv * 4) * 128 + i % 512];
                piece += 1;
            }
            let mut piece = 0;
            while piece < 4 {
                let i = t + piece * 256;
                cuda_device::async_copy::cp_async_ca_16(
                    tile.add(i * 4),
                    k.as_ptr()
                        .add(page * 65536 + (kv * 128 + i / 8) * 64 + i % 8 * 8)
                        .cast::<u32>(),
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
            cuda_device::async_copy::cp_async_wait_all();
        }
        thread::sync_threads();
        if member < count && base + key_index < end {
            let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
            let query = unsafe { queries.add(member * 512) };
            let mut c = 0;
            while c < 128 {
                let key = widen(unsafe { *halves.add(c * 64 + key_index) });
                s0 = add(s0, mul(unsafe { *query.add(c) }, key));
                s1 = add(s1, mul(unsafe { *query.add(128 + c) }, key));
                s2 = add(s2, mul(unsafe { *query.add(256 + c) }, key));
                s3 = add(s3, mul(unsafe { *query.add(384 + c) }, key));
                c += 1;
            }
            let offset = (row * 32 + kv * 4) * stride as usize + base + key_index;
            unsafe {
                *scores.get_unchecked_mut(offset) = mul(s0, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + stride as usize) =
                    mul(s1, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + 2 * stride as usize) =
                    mul(s2, 0.08838834764831845_f32);
                *scores.get_unchecked_mut(offset + 3 * stride as usize) =
                    mul(s3, 0.08838834764831845_f32);
            }
        }
    }

    #[inline(always)]
    unsafe fn stage_values_tiled(
        tile: *mut u32,
        probability: *mut f32,
        scores: &[f32],
        v: &[u16],
        row_info: &[u32],
        first_row: usize,
        row_count: usize,
        stride: usize,
        page: usize,
        kv: usize,
        column: usize,
        t: usize,
        base: usize,
    ) {
        let slot = base / 64 % 2;
        unsafe {
            let mut piece = 0;
            while piece < 2 {
                let i = t * 8 + piece * 1024;
                cuda_device::async_copy::cp_async_ca_16(
                    tile.add(slot * 1024 + i / 2),
                    v.as_ptr()
                        .add(page * 65536 + i / 32 * 1024 + kv * 128 + column + i % 32)
                        .cast::<u32>(),
                );
                let i = t * 4 + piece * 512;
                let head = i / 64;
                let row = first_row + (head / 4).min(row_count - 1);
                let end = row_info[row * 2 + 1] as usize + 1;
                let token = base + i % 64;
                let bytes = end.saturating_sub(token).min(4) * 4;
                let source = if bytes == 0 {
                    0
                } else {
                    (row * 32 + kv * 4 + head % 4) * stride + token
                };
                cuda_device::async_copy::cp_async_ca_zfill_16(
                    probability.add(slot * 1024 + i).cast::<u32>(),
                    scores.as_ptr().add(source).cast::<u8>(),
                    bytes as u32,
                );
                piece += 1;
            }
            cuda_device::async_copy::cp_async_commit_group();
        }
    }

    /// Four prompt rows from one sequence share each staged V tile. Query tiles
    /// are contiguous host-validated (first row, count) pairs; 1 <= count <= 4.
    #[kernel]
    pub fn attention_values_tiled(
        scores: &[f32],
        v: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        query_tiles: &[u32],
        mut out: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut TILE: SharedArray<u32, 2048, 16> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, 2048, 16> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let tile_index = block / 32;
        let first_row = query_tiles[tile_index * 2] as usize;
        let count = query_tiles[tile_index * 2 + 1] as usize;
        let kv = block / 4 % 8;
        let column = block % 4 * 32;
        let t = thread::threadIdx_x() as usize;
        let member = t / 32;
        let lane = t % 32;
        let row = first_row + member.min(count - 1);
        let slot = row_info[first_row * 2] as usize;
        let end = row_info[row * 2 + 1] as usize + 1;
        let max_end = row_info[(first_row + count - 1) * 2 + 1] as usize + 1;
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        let probability = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PROB) };
        unsafe {
            stage_values_tiled(
                tile,
                probability,
                scores,
                v,
                row_info,
                first_row,
                count,
                stride as usize,
                page_table[slot * pages_per_sequence as usize] as usize,
                kv,
                column,
                t,
                0,
            );
        }
        let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
        let mut base = 0;
        while base < max_end {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            if base + 64 < max_end {
                let page = page_table[slot * pages_per_sequence as usize + base / 64 + 1] as usize;
                unsafe {
                    stage_values_tiled(
                        tile,
                        probability,
                        scores,
                        v,
                        row_info,
                        first_row,
                        count,
                        stride as usize,
                        page,
                        kv,
                        column,
                        t,
                        base + 64,
                    );
                }
            }
            let shared_slot = base / 64 % 2;
            let probs = unsafe { probability.add(shared_slot * 1024 + member * 256) };
            let mut j = 0;
            let key_count = end.saturating_sub(base).min(64);
            #[unroll(8)]
            while j < key_count {
                let value = widen(unsafe { *halves.add(shared_slot * 2048 + j * 32 + lane) });
                s0 = add(s0, mul(unsafe { *probs.add(j) }, value));
                s1 = add(s1, mul(unsafe { *probs.add(64 + j) }, value));
                s2 = add(s2, mul(unsafe { *probs.add(128 + j) }, value));
                s3 = add(s3, mul(unsafe { *probs.add(192 + j) }, value));
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        if member < count {
            let offset = (row * 32 + kv * 4) * 128 + column + lane;
            unsafe {
                *out.get_unchecked_mut(offset) = bf(s0);
                *out.get_unchecked_mut(offset + 128) = bf(s1);
                *out.get_unchecked_mut(offset + 256) = bf(s2);
                *out.get_unchecked_mut(offset + 384) = bf(s3);
            }
        }
    }

    /// Four independent ordered PV sums reuse a single staged value tile.
    #[kernel]
    pub fn attention_values_grouped(
        scores: &[f32],
        v: &[u16],
        row_info: &[u32],
        page_table: &[u32],
        pages_per_sequence: u32,
        stride: u32,
        mut out: DisjointSlice<f32>,
    ) {
        dependency_wait();
        static mut TILE: SharedArray<u32, 2048, 16> = SharedArray::UNINIT;
        static mut PROB: SharedArray<f32, 512> = SharedArray::UNINIT;
        let block = thread::blockIdx_x() as usize;
        let group = block / 4;
        let query = group * 4;
        let column = block % 4 * 32;
        let t = thread::threadIdx_x() as usize;
        let slot = row_info[group / 8 * 2] as usize;
        let end = row_info[group / 8 * 2 + 1] as usize + 1;
        let kv = group % 8;
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        let halves = tile.cast::<u16>();
        let probability = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PROB) };
        stage_values_grouped(
            tile,
            probability,
            scores,
            v,
            query,
            stride as usize,
            page_table[slot * pages_per_sequence as usize] as usize,
            kv,
            column,
            t,
            0,
            end,
        );
        let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
        let mut base = 0;
        while base < end {
            unsafe {
                cuda_device::async_copy::cp_async_wait_all();
            }
            thread::sync_threads();
            if base + 64 < end {
                let page = page_table[slot * pages_per_sequence as usize + base / 64 + 1] as usize;
                stage_values_grouped(
                    tile,
                    probability,
                    scores,
                    v,
                    query,
                    stride as usize,
                    page,
                    kv,
                    column,
                    t,
                    base + 64,
                    end,
                );
            }
            let shared_slot = base / 64 % 2;
            let probs = unsafe { probability.add(shared_slot * 256) };
            let mut j = 0;
            let count = (end - base).min(64);
            #[unroll(8)]
            while j < count {
                let value = widen(unsafe { *halves.add(shared_slot * 2048 + j * 32 + t) });
                s0 = add(s0, mul(unsafe { *probs.add(j) }, value));
                s1 = add(s1, mul(unsafe { *probs.add(64 + j) }, value));
                s2 = add(s2, mul(unsafe { *probs.add(128 + j) }, value));
                s3 = add(s3, mul(unsafe { *probs.add(192 + j) }, value));
                j += 1;
            }
            thread::sync_threads();
            base += 64;
        }
        let offset = query * 128 + column + t;
        unsafe {
            *out.get_unchecked_mut(offset) = bf(s0);
            *out.get_unchecked_mut(offset + 128) = bf(s1);
            *out.get_unchecked_mut(offset + 256) = bf(s2);
            *out.get_unchecked_mut(offset + 384) = bf(s3);
        }
    }

    #[kernel]
    pub fn head_packed_rows(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        mut out: DisjointSlice<f32>,
        prefetch: u32,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let index = tid / 32;
        // Adjacent warps evaluate the same vocabulary row for different batch
        // members, so its packed weights stay hot in L1/L2 across those uses.
        // The output remains row-major and each dot's arithmetic is unchanged.
        let batch = index % output_rows.len();
        let row = index / output_rows.len();
        if index < out.len() {
            // Sixteen 96-word chunks form each packed row's contiguous 6144 bytes.
            prefetch_row(
                unsafe { words.as_ptr().add(row * 1536) }.cast(),
                6144,
                prefetch,
                lane,
            );
        }
        dependency_wait();
        if index >= out.len() {
            return;
        }
        let start = output_rows[batch] as usize * 4096;
        let mut sum = 0.0;
        // Three coalesced word planes encode eight successive terms per lane.
        // Keep the original lane-strided FP32 accumulation and warp sum order.
        let mut group = 0;
        while group < 16 {
            let chunk = row * 16 + group;
            let word = chunk * 96 + lane;
            let w0 = words[word];
            let w1 = words[word + 32];
            let w2 = words[word + 64];
            let j = group * 256 + lane;
            sum = add(
                sum,
                mul(
                    x[start + j],
                    widen(unpack_code(w0, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 32],
                    widen(unpack_code(w0 >> 12, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 64],
                    widen(unpack_code(
                        (w0 >> 24) | (w1 << 8),
                        offsets,
                        exceptions,
                        chunk,
                    )),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 96],
                    widen(unpack_code(w1 >> 4, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 128],
                    widen(unpack_code(w1 >> 16, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 160],
                    widen(unpack_code(
                        (w1 >> 28) | (w2 << 4),
                        offsets,
                        exceptions,
                        chunk,
                    )),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 192],
                    widen(unpack_code(w2 >> 8, offsets, exceptions, chunk)),
                ),
            );
            sum = add(
                sum,
                mul(
                    x[start + j + 224],
                    widen(unpack_code(w2 >> 20, offsets, exceptions, chunk)),
                ),
            );
            group += 1;
        }
        let mut d = 16;
        while d > 0 {
            sum = add(sum, f32::from_bits(warp::shuffle_down(sum.to_bits(), d)));
            d /= 2;
        }
        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(batch * 151936 + row) = bf(sum);
            }
        }
    }
    /// A warp shares packed-weight reconstruction across four exact head dots.
    #[kernel]
    pub fn head_packed_four(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        out: DisjointSlice<f32>,
    ) {
        head_packed_group::<false>(x, output_rows, words, offsets, exceptions, out);
    }

    /// Eight-row specialization for larger batches.
    #[kernel]
    pub fn head_packed_eight(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        out: DisjointSlice<f32>,
    ) {
        head_packed_group::<true>(x, output_rows, words, offsets, exceptions, out);
    }

    #[kernel]
    pub fn head_packed_eight_staged(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        out: DisjointSlice<f32>,
    ) {
        static mut INPUT: SharedArray<u32, 4096, 16> = SharedArray::UNINIT;
        let staged = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
        head_packed_staged::<false>(staged, x, output_rows, words, offsets, exceptions, out);
    }

    #[kernel]
    pub fn head_packed_sixteen_staged(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        out: DisjointSlice<f32>,
    ) {
        static mut INPUT: SharedArray<u32, 8192, 16> = SharedArray::UNINIT;
        let staged = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
        head_packed_staged::<true>(staged, x, output_rows, words, offsets, exceptions, out);
    }

    #[kernel]
    pub fn head_packed_thirtytwo_staged(
        x: &[f32],
        output_rows: &[u32],
        words: &[u32],
        offsets: &[u32],
        exceptions: &[u16],
        out: DisjointSlice<f32>,
    ) {
        static mut INPUT: SharedArray<u32, 8192, 16> = SharedArray::UNINIT;
        let staged = unsafe { SharedArray::as_raw_mut_ptr(&raw mut INPUT) };
        head_packed32_single(staged, x, output_rows, words, offsets, exceptions, out);
    }

    #[kernel]
    pub fn select_token_rows(x: &[f32], mut out: DisjointSlice<u32>) {
        static mut PARTIAL: SharedArray<u32, 96> = SharedArray::UNINIT;
        dependency_wait();
        let row = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        let mut value = f32::NEG_INFINITY;
        let mut index = u32::MAX;
        let mut bad = if x.len() == 0 { 1 } else { 0 };
        let mut j = t;
        while j < 151936 {
            let v = x[row * 151936 + j];
            if !v.is_finite() {
                bad = 1;
            }
            if v > value || (v == value && (j as u32) < index) {
                value = v;
                index = j as u32;
            }
            j += 1024;
        }
        let (value, index, bad) = best_warp(value, index, bad);
        if t % 32 == 0 {
            unsafe {
                *p.add(t / 32) = value.to_bits();
                *p.add(32 + t / 32) = index;
                *p.add(64 + t / 32) = bad;
            }
        }
        thread::sync_threads();
        if t < 32 {
            let (_, index, bad) =
                unsafe { best_warp(f32::from_bits(*p.add(t)), *p.add(32 + t), *p.add(64 + t)) };
            if t == 0 {
                unsafe {
                    *out.get_unchecked_mut(row) = if bad == 0 { index } else { u32::MAX };
                }
            }
        }
    }
}

/// TMA projections requiring an opt-in dynamic shared-memory resource contract.
#[cuda_module]
pub mod pipeline_device {
    use super::*;
    #[kernel]
    #[cuda_device::launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 50176, dynamic_shared_alignment = 1024)]
    pub fn linear_large_pipeline(
        amap: &[u64],
        asc: &[f32],
        bmap: &[u64],
        ws: &[f32],
        width: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let t = thread::threadIdx_x() as usize;
        let lane = t % 32;
        let warp_id = t / 32;
        let tiles_n = ws.len().div_ceil(128);
        let tile = thread::blockIdx_x() as usize;
        let row = tile / tiles_n * 64;
        let col = tile % tiles_n * 128;
        let wr = warp_id / 4 * 32;
        let wc = warp_id % 4 * 32;
        let group = lane / 4;
        let part = lane % 4;
        let k = width as usize;
        let a_shared = cuda_device::DynamicSharedArray::<u32, 1024>::get();
        let b_shared = unsafe { a_shared.add(4096) };
        let mut c00 = [0_i32; 4];
        let mut c01 = [0_i32; 4];
        let mut c02 = [0_i32; 4];
        let mut c03 = [0_i32; 4];
        let mut c10 = [0_i32; 4];
        let mut c11 = [0_i32; 4];
        let mut c12 = [0_i32; 4];
        let mut c13 = [0_i32; 4];
        let bar = unsafe { a_shared.add(12288) }.cast::<cuda_device::barrier::Barrier>();
        if t == 0 {
            unsafe {
                cuda_device::barrier::mbarrier_init(bar, 1);
                cuda_device::barrier::mbarrier_init(bar.add(1), 1);
                cuda_device::barrier::fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();
        if t == 0 {
            unsafe {
                cuda_device::barrier::mbarrier_arrive_expect_tx(bar, 1, 24576);
                tma_copy(
                    a_shared.cast::<u8>(),
                    amap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                    0,
                    row as i32,
                    bar,
                );
                tma_copy(
                    b_shared.cast::<u8>(),
                    bmap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                    0,
                    col as i32,
                    bar,
                );
            }
        }
        let mut base = 0;
        while base < k {
            let tile_index = base / 128;
            let slot = tile_index % 2;
            let phase = (tile_index / 2 % 2) as u32;
            while !unsafe { cuda_device::barrier::mbarrier_try_wait_parity(bar.add(slot), phase) } {
            }
            thread::sync_threads();
            if t == 0 && base + 128 < k {
                let next = 1 - slot;
                unsafe {
                    let next_bar = bar.add(next);
                    cuda_device::barrier::mbarrier_arrive_expect_tx(next_bar, 1, 24576);
                    tma_copy(
                        a_shared.add(next * 2048).cast::<u8>(),
                        amap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                        (base + 128) as i32,
                        row as i32,
                        next_bar,
                    );
                    tma_copy(
                        b_shared.add(next * 4096).cast::<u8>(),
                        bmap.as_ptr().cast::<cuda_device::tma::TmaDescriptor>(),
                        (base + 128) as i32,
                        col as i32,
                        next_bar,
                    );
                }
            }
            let a_tile = unsafe { a_shared.add(slot * 2048) };
            let b_tile = unsafe { b_shared.add(slot * 4096) };
            let mut j = 0;
            while j < 32 {
                unsafe {
                    let a0 = cuda_device::wmma::ldmatrix_x4(
                        a_tile.add(tma_word(wr + lane % 16, j + lane / 16 * 4)),
                    );
                    let a1 = cuda_device::wmma::ldmatrix_x4(
                        a_tile.add(tma_word(wr + 16 + lane % 16, j + lane / 16 * 4)),
                    );
                    let b0 = cuda_device::wmma::ldmatrix_x4(b_tile.add(tma_word(
                        wc + lane / 16 * 8 + lane % 8,
                        j + lane / 8 % 2 * 4,
                    )));
                    let b1 = cuda_device::wmma::ldmatrix_x4(b_tile.add(tma_word(
                        wc + 16 + lane / 16 * 8 + lane % 8,
                        j + lane / 8 % 2 * 4,
                    )));
                    c00 = cuda_device::wmma::mma_m16n8k32_s32_s8(c00, a0, [b0[0], b0[1]]);
                    c01 = cuda_device::wmma::mma_m16n8k32_s32_s8(c01, a0, [b0[2], b0[3]]);
                    c02 = cuda_device::wmma::mma_m16n8k32_s32_s8(c02, a0, [b1[0], b1[1]]);
                    c03 = cuda_device::wmma::mma_m16n8k32_s32_s8(c03, a0, [b1[2], b1[3]]);
                    c10 = cuda_device::wmma::mma_m16n8k32_s32_s8(c10, a1, [b0[0], b0[1]]);
                    c11 = cuda_device::wmma::mma_m16n8k32_s32_s8(c11, a1, [b0[2], b0[3]]);
                    c12 = cuda_device::wmma::mma_m16n8k32_s32_s8(c12, a1, [b1[0], b1[1]]);
                    c13 = cuda_device::wmma::mma_m16n8k32_s32_s8(c13, a1, [b1[2], b1[3]]);
                }
                j += 8;
            }
            // No thread may overwrite staging before the last warp consumes it.
            thread::sync_threads();
            base += 128;
        }
        if t == 0 {
            unsafe {
                cuda_device::barrier::mbarrier_inval(bar);
                cuda_device::barrier::mbarrier_inval(bar.add(1));
            }
        }
        let mut i = 0;
        while i < 4 {
            let r = row + wr + group + i / 2 * 8;
            let c = col + wc + part * 2 + i % 2;
            unsafe {
                if r + 0 < asc.len() && c + 0 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 0) =
                        bf(mul(mul(c00[i] as f32, asc[r + 0]), ws[c + 0]));
                }
                if r + 0 < asc.len() && c + 8 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 8) =
                        bf(mul(mul(c01[i] as f32, asc[r + 0]), ws[c + 8]));
                }
                if r + 0 < asc.len() && c + 16 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 16) =
                        bf(mul(mul(c02[i] as f32, asc[r + 0]), ws[c + 16]));
                }
                if r + 0 < asc.len() && c + 24 < ws.len() {
                    *out.get_unchecked_mut((r + 0) * ws.len() + c + 24) =
                        bf(mul(mul(c03[i] as f32, asc[r + 0]), ws[c + 24]));
                }
                if r + 16 < asc.len() && c + 0 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 0) =
                        bf(mul(mul(c10[i] as f32, asc[r + 16]), ws[c + 0]));
                }
                if r + 16 < asc.len() && c + 8 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 8) =
                        bf(mul(mul(c11[i] as f32, asc[r + 16]), ws[c + 8]));
                }
                if r + 16 < asc.len() && c + 16 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 16) =
                        bf(mul(mul(c12[i] as f32, asc[r + 16]), ws[c + 16]));
                }
                if r + 16 < asc.len() && c + 24 < ws.len() {
                    *out.get_unchecked_mut((r + 16) * ws.len() + c + 24) =
                        bf(mul(mul(c13[i] as f32, asc[r + 16]), ws[c + 24]));
                }
            }
            i += 1;
        }
    }
}
