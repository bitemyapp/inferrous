//! Experimental exact-output certification; the packed head remains the fallback.
use super::*;

/// Host preparation consumes original BF16 directly, avoiding a full FP32 copy.
/// Plane layout and outward metadata match the independent operator contract.
pub(crate) fn prepare_weights(weights: &[u16]) -> (Vec<i8>, Vec<f64>) {
    assert!(!weights.is_empty() && weights.len() % HIDDEN == 0);
    let nr = weights.len() / HIDDEN;
    let mut digits = vec![0_i8; 3 * weights.len()];
    let mut stats = vec![0.0_f64; nr * 5];
    for (row, values) in weights.chunks_exact(HIDDEN).enumerate() {
        let valid = values.iter().all(|&bits| {
            let magnitude = bits & 0x7fff;
            magnitude == 0 || ((77_u16 << 7)..=(157_u16 << 7)).contains(&magnitude)
        });
        let maximum = values.iter().map(|bits| bits & 0x7fff).max().unwrap();
        let mut scale = 1.0_f32;
        if valid && maximum != 0 {
            scale = f32::from_bits((((maximum as u32) >> 7) - 22) << 23);
            if f32::from_bits((maximum as u32) << 16) > 8355711.0 * scale {
                scale *= 2.0;
            }
        }
        let mut sums = [0.0_f64; 4];
        for (j, &bits) in values.iter().enumerate() {
            let value = if valid {
                f32::from_bits((bits as u32) << 16)
            } else {
                0.0
            };
            let q = (value / scale).round_ties_even() as i32;
            let middle_high = (q + 128) >> 8;
            let high = (middle_high + 128) >> 8;
            digits[row * HIDDEN + j] = (q - middle_high * 256) as i8;
            digits[weights.len() + row * HIDDEN + j] = (middle_high - high * 256) as i8;
            digits[2 * weights.len() + row * HIDDEN + j] = high as i8;
            let v = value as f64;
            let a = (q as f64) * (scale as f64);
            let e = v - a;
            sums[0] += v * v;
            sums[1] += a * a;
            sums[2] += e * e;
            sums[3] += v * v * (133 - j / 32) as f64;
        }
        stats[row * 5] = scale as f64;
        // Positive reductions: inflation exceeds gamma_4096 in FP64.
        for i in 0..4 {
            stats[row * 5 + 1 + i] = (sums[i] * (1.0 + 2.0_f64.powi(-39)))
                .next_up()
                .sqrt()
                .next_up();
        }
        if !valid {
            stats[row * 5 + 4] = f64::INFINITY;
        }
    }
    (digits, stats)
}

pub(super) struct CertifiedHead {
    pub(super) digits: DeviceBuffer<i8>,
    pub(super) stats: DeviceBuffer<f64>,
}
impl CertifiedHead {
    pub(super) fn new(stream: &Arc<CudaStream>, weights: &[u16]) -> Result<Self> {
        let (digits, stats) = prepare_weights(weights);
        Ok(Self {
            digits: DeviceBuffer::from_host(stream, &digits)?,
            stats: DeviceBuffer::from_host(stream, &stats)?,
        })
    }
}

pub(super) struct CertificateWorkspace {
    pub(super) digits: DeviceBuffer<i8>,
    pub(super) stats: DeviceBuffer<f64>,
    pub(super) center: DeviceBuffer<i64>,
    pub(super) masks: DeviceBuffer<u32>,
}
impl CertificateWorkspace {
    pub(super) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            digits: DeviceBuffer::zeroed(stream, 3 * 32 * HIDDEN)?,
            stats: DeviceBuffer::zeroed(stream, 32 * 5)?,
            center: DeviceBuffer::zeroed(stream, 32 * VOCAB)?,
            masks: DeviceBuffer::zeroed(stream, VOCAB)?,
        })
    }
}
