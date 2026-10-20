//! Explicit operator timing experiments; excluded from the ordinary test suite.
use super::*;

#[test]
#[ignore = "GPU projection timing experiment; run alone with --nocapture"]
fn projection_shapes() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = device::load(&context)?;
    let start = context.new_event(Some(0))?;
    let end = context.new_event(Some(0))?;
    let device_name = context.device_name()?;
    // Rotate through 512 MiB or more of distinct weight allocations, avoiding
    // the unrealistically hot weights of a single-matrix timing loop.
    const COPIES: usize = 32;
    for (n, k) in [(4096, 4096), (12288, 4096), (4096, 12288)] {
        let values: Vec<_> = (0..n * k)
            .map(|i| {
                (((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) >> 16) % 255) as i16
                    - 127
            })
            .map(|v| v as i8)
            .collect();
        let mut weights = Vec::new();
        let mut weight_maps = Vec::new();
        for _ in 0..COPIES {
            let weight = DeviceBuffer::from_host(&stream, &values)?;
            weight_maps.push(crate::tensor::map(&stream, &weight, n, k, 128)?);
            weights.push(weight);
        }
        let weight_scales = DeviceBuffer::from_host(&stream, &vec![0.001_f32; n])?;
        for nr in [1_usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let input: Vec<_> = (0..nr * k)
                .map(|i| ((i * 37 + 13) % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            let q = DeviceBuffer::from_host(&stream, &input)?;
            let scales = DeviceBuffer::from_host(&stream, &vec![0.003_f32; nr])?;
            let map = if nr >= 256 {
                Some(crate::tensor::map(&stream, &q, nr, k, 64)?)
            } else {
                None
            };
            let mut out = DeviceBuffer::zeroed(&stream, nr * n)?;
            // DP4A is an independent reference for the integer tensor-core paths.
            unsafe {
                module.linear(
                    &stream,
                    rows((nr * n).div_ceil(8), 256),
                    &q,
                    &scales,
                    &weights[0],
                    &weight_scales,
                    k as u32,
                    &mut out,
                )?;
            }
            let expected = out.to_host_vec(&stream)?;
            for variant in ["dp4a", "splitk", "tiled", "large"] {
                if (variant == "dp4a" && nr > 16) || (variant == "large" && nr < 256) {
                    continue;
                }
                stream.synchronize()?;
                // SAFETY: all captured allocations are retained in the enclosing
                // scopes, graph drops first, and this experiment is stream-serial.
                let graph = unsafe {
                    Graph::capture(&stream, false, || {
                        for (weight, weight_map) in weights.iter().zip(&weight_maps) {
                            match variant {
                                "dp4a" => module.linear(
                                    &stream,
                                    rows((nr * n).div_ceil(8), 256),
                                    &q,
                                    &scales,
                                    weight,
                                    &weight_scales,
                                    k as u32,
                                    &mut out,
                                )?,
                                "splitk" => module.linear_splitk(
                                    &stream,
                                    rows(nr.div_ceil(16) * n.div_ceil(8), 256),
                                    &q,
                                    &scales,
                                    weight,
                                    &weight_scales,
                                    k as u32,
                                    &mut out,
                                )?,
                                "tiled" => module.linear_tiled(
                                    &stream,
                                    rows(nr.div_ceil(32) * n.div_ceil(64), 256),
                                    &q,
                                    &scales,
                                    weight,
                                    &weight_scales,
                                    k as u32,
                                    &mut out,
                                )?,
                                "large" => module.linear_large(
                                    &stream,
                                    rows(nr.div_ceil(64) * n.div_ceil(128), 256),
                                    map.as_ref().expect("large map"),
                                    &scales,
                                    weight_map,
                                    &weight_scales,
                                    k as u32,
                                    &mut out,
                                )?,
                                _ => unreachable!(),
                            }
                        }
                        Ok(())
                    })?
                };
                let mut times = Vec::new();
                for repetition in 0..10 {
                    start.record(&stream)?;
                    unsafe {
                        graph.launch()?;
                    }
                    end.record(&stream)?;
                    let milliseconds = start.elapsed_ms(&end)? / COPIES as f32;
                    if repetition >= 3 {
                        times.push(milliseconds);
                    }
                }
                assert_eq!(
                    out.to_host_vec(&stream)?,
                    expected,
                    "{nr}x{k}x{n} {variant}"
                );
                println!(
                    "PROJECTION {}",
                    serde_json::json!({"device": device_name,
                    "rows": nr, "k": k, "n": n, "variant": variant,
                    "weight_working_set_bytes": COPIES * k * n, "milliseconds": times})
                );
            }
        }
    }
    Ok(())
}
