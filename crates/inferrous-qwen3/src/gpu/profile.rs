//! Explicit operator timing experiments; excluded from the ordinary test suite.
use super::*;

#[test]
#[ignore = "GPU head timing experiment; run alone with --nocapture and a model bundle"]
fn grouped_head_shapes() -> Result<()> {
    let bundle = std::env::var_os("CLEAN_MODEL_BUNDLE").expect("CLEAN_MODEL_BUNDLE");
    let model = Model::load(Path::new(&bundle))?;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = device::load(&context)?;
    let head = PackedHead::new(&stream, &model.head)?;
    let start = context.new_event(Some(0))?;
    let end = context.new_event(Some(0))?;
    let device_name = context.device_name()?;
    for nr in 1_usize..=64 {
        let input: Vec<f32> = (0..(nr + 2) * HIDDEN)
            .map(|i| ((i * 37 + 11) % 255) as f32 / 128.0 - 1.0)
            .collect();
        let input = DeviceBuffer::from_host(&stream, &input)?;
        let selected: Vec<u32> = (0..nr).map(|i| (nr - i) as u32).collect();
        let selected = DeviceBuffer::from_host(&stream, &selected)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, nr * VOCAB)?;
        let mut expected = None;
        for (group_size, staged) in [
            (1_usize, false),
            (4, false),
            (8, false),
            (8, true),
            (16, true),
        ] {
            stream.synchronize()?;
            // SAFETY: selected rows are in bounds, every output has a distinct
            // warp owner, and graph/allocation lifetimes are nested and fenced.
            let graph = unsafe {
                Graph::capture(&stream, false, || {
                    if group_size == 16 {
                        module.head_packed_sixteen_staged(
                            &stream,
                            rows(nr.div_ceil(16) * VOCAB / 8, 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 8 && staged {
                        module.head_packed_eight_staged(
                            &stream,
                            rows(nr.div_ceil(8) * VOCAB / 8, 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 8 {
                        module.head_packed_eight(
                            &stream,
                            rows((nr.div_ceil(8) * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 4 {
                        module.head_packed_four(
                            &stream,
                            rows((nr.div_ceil(4) * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else {
                        module.head_packed_rows(
                            &stream,
                            rows((nr * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                            0,
                        )?;
                    }
                    Ok(())
                })?
            };
            let mut times = Vec::new();
            for repetition in 0..10 {
                let _eviction = DeviceBuffer::<u8>::zeroed(&stream, 256 << 20)?;
                start.record(&stream)?;
                unsafe {
                    graph.launch()?;
                }
                end.record(&stream)?;
                let milliseconds = start.elapsed_ms(&end)?;
                if repetition >= 3 {
                    times.push(milliseconds);
                }
            }
            let actual = out.to_host_vec(&stream)?;
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "head {nr} rows");
            } else {
                expected = Some(actual);
            }
            println!(
                "HEAD {}",
                serde_json::json!({"device":device_name, "model_root":model.root,
                "rows":nr, "group_size":group_size, "staged":staged, "milliseconds":times})
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU attention timing experiment; run alone with --nocapture"]
fn grouped_attention_shapes() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = device::load(&context)?;
    let start = context.new_event(Some(0))?;
    let end = context.new_event(Some(0))?;
    let device_name = context.device_name()?;
    for stride in [64_usize, 512, 4096, 32768] {
        let pages = stride / 64;
        let table: Vec<u32> = (0..pages as u32).rev().collect();
        let table = DeviceBuffer::from_host(&stream, &table)?;
        let values: Vec<u16> = (0..stride * 1024)
            .map(|i| ((((i * 37 + 11) % 127) as f32 - 63.0) / 64.0).to_bits())
            .map(|bits| (bits >> 16) as u16)
            .collect();
        let k = DeviceBuffer::from_host(&stream, &values)?;
        let v = DeviceBuffer::from_host(&stream, &values)?;
        for nr in [1, 2, 4, 8, 16, 64, 256] {
            if nr > stride {
                continue;
            }
            let info: Vec<u32> = (0..nr)
                .flat_map(|row| [0, (stride - nr + row) as u32])
                .collect();
            let info = DeviceBuffer::from_host(&stream, &info)?;
            let query_tiles: Vec<u32> = (0..nr)
                .step_by(4)
                .flat_map(|first| [first as u32, (nr - first).min(4) as u32])
                .collect();
            let tile_count = query_tiles.len() / 2;
            let query_tiles = DeviceBuffer::from_host(&stream, &query_tiles)?;
            let q: Vec<f32> = (0..nr * 4096)
                .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let q = DeviceBuffer::from_host(&stream, &q)?;
            let mut scores = DeviceBuffer::<f32>::zeroed(&stream, nr * 32 * stride)?;
            let mut out = DeviceBuffer::<f32>::zeroed(&stream, nr * 4096)?;
            let mut expected = None;
            for (grouped_scores, grouped_values, tiled) in [
                (false, false, false),
                (true, false, false),
                (false, true, false),
                (true, true, false),
                (true, true, true),
            ] {
                stream.synchronize()?;
                // SAFETY: every query uses one fully initialized sequence; the
                // reversed table covers precisely the allocated physical pages.
                let graph = unsafe {
                    Graph::capture(&stream, false, || {
                        if tiled {
                            module.attention_scores_tiled(
                                &stream,
                                rows(tile_count * 8 * pages, 256),
                                &q,
                                &k,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                pages as u32,
                                &query_tiles,
                                &mut scores,
                            )?;
                        } else if grouped_scores {
                            module.attention_scores_grouped(
                                &stream,
                                rows(nr * 8 * pages, 128),
                                &q,
                                &k,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                pages as u32,
                                &mut scores,
                            )?;
                        } else {
                            module.attention_scores_paged(
                                &stream,
                                rows(nr * 32 * pages, 128),
                                &q,
                                &k,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                pages as u32,
                                &mut scores,
                            )?;
                        }
                        module.attention_softmax_paged(
                            &stream,
                            rows(nr * 32, 128),
                            &mut scores,
                            &info,
                            stride as u32,
                        )?;
                        if tiled {
                            module.attention_values_tiled(
                                &stream,
                                rows(tile_count * 8 * 4, 128),
                                &scores,
                                &v,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                &query_tiles,
                                &mut out,
                            )?;
                        } else if grouped_values {
                            module.attention_values_grouped(
                                &stream,
                                rows(nr * 8 * 4, 32),
                                &scores,
                                &v,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                &mut out,
                            )?;
                        } else {
                            module.attention_values_paged(
                                &stream,
                                rows(nr * 32 * 4, 32),
                                &scores,
                                &v,
                                &info,
                                &table,
                                pages as u32,
                                stride as u32,
                                &mut out,
                            )?;
                        }
                        Ok(())
                    })?
                };
                let mut times = Vec::new();
                for repetition in 0..10 {
                    // Evict the previous invocation's cache lines before timing.
                    // Real model layers have intervening projection weight reads.
                    let _eviction = DeviceBuffer::<u8>::zeroed(&stream, 256 << 20)?;
                    start.record(&stream)?;
                    unsafe {
                        graph.launch()?;
                    }
                    end.record(&stream)?;
                    let milliseconds = start.elapsed_ms(&end)?;
                    if repetition >= 3 {
                        times.push(milliseconds);
                    }
                }
                let actual = out.to_host_vec(&stream)?;
                if let Some(expected) = &expected {
                    assert_eq!(
                        &actual, expected,
                        "attention {nr}x{stride} grouped {grouped_scores}/{grouped_values}"
                    );
                } else {
                    expected = Some(actual);
                }
                println!(
                    "ATTENTION {}",
                    serde_json::json!({"device":device_name,
                    "rows":nr, "stride":stride, "grouped_scores":grouped_scores,
                    "grouped_values":grouped_values, "tiled":tiled, "milliseconds":times})
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "GPU projection timing experiment; run alone with --nocapture"]
fn projection_shapes() -> Result<()> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = device::load(&context)?;
    // SAFETY: this experiment uses the declared launch contract and retains all
    // slice allocations until its stream has completed.
    let pipeline_module = unsafe { crate::kernels::pipeline_device::load(&context)? };
    let start = context.new_event(Some(0))?;
    let end = context.new_event(Some(0))?;
    let device_name = context.device_name()?;
    // Rotate through 128 MiB or more of distinct weight allocations, avoiding
    // the unrealistically hot weights of a single-matrix timing loop.
    const COPIES: usize = 32;
    for (n, k) in [(1024, 4096), (4096, 4096), (12288, 4096), (4096, 12288)] {
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
            let map = if nr >= 16 {
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
            let pipeline =
                pipeline_module.prepare_linear_large_pipeline(cuda_core::LaunchConfig1D::new(
                    (nr.div_ceil(64) * n.div_ceil(128)) as u32,
                    256,
                    50176,
                ))?;
            for variant in [
                "dp4a",
                "splitk",
                "tiled",
                "tiled-pipeline",
                "narrow-pipeline",
                "large",
                "large-pipeline",
            ] {
                if (variant == "dp4a" && nr > 16) || (variant.starts_with("large") && nr < 16) {
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
                                "tiled-pipeline" => module.linear_tiled_pipeline(
                                    &stream,
                                    rows(nr.div_ceil(32) * n.div_ceil(64), 256),
                                    &q,
                                    &scales,
                                    weight,
                                    &weight_scales,
                                    k as u32,
                                    &mut out,
                                )?,
                                "narrow-pipeline" => module.linear_narrow_pipeline(
                                    &stream,
                                    rows(nr.div_ceil(16) * n.div_ceil(32), 128),
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
                                "large-pipeline" => pipeline_module.linear_large_pipeline(
                                    &stream,
                                    &pipeline,
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

#[test]
#[ignore = "GPU head timing experiment; run alone with --nocapture and a model bundle"]
fn head32_shapes() -> Result<()> {
    let bundle = std::env::var_os("CLEAN_MODEL_BUNDLE").expect("CLEAN_MODEL_BUNDLE");
    let model = Model::load(Path::new(&bundle))?;
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = device::load(&context)?;
    let head = PackedHead::new(&stream, &model.head)?;
    let start = context.new_event(Some(0))?;
    let end = context.new_event(Some(0))?;
    let device_name = context.device_name()?;
    for nr in [1_usize, 2, 8, 16, 17, 24, 31, 32, 33, 48, 64] {
        let input: Vec<f32> = (0..(nr + 2) * HIDDEN)
            .map(|i| ((i * 37 + 11) % 255) as f32 / 128.0 - 1.0)
            .collect();
        let input = DeviceBuffer::from_host(&stream, &input)?;
        let selected: Vec<u32> = (0..nr).map(|i| (nr - i) as u32).collect();
        let selected = DeviceBuffer::from_host(&stream, &selected)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, nr * VOCAB)?;
        let mut expected = None;
        for (group_size, staged) in [(16_usize, true), (32, true)] {
            stream.synchronize()?;
            // SAFETY: selected rows are in bounds, every output has a distinct
            // warp owner, and graph/allocation lifetimes are nested and fenced.
            let graph = unsafe {
                Graph::capture(&stream, false, || {
                    if group_size == 32 {
                        module.head_packed_thirtytwo_staged(
                            &stream,
                            rows(nr.div_ceil(32) * VOCAB / 8, 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 16 {
                        module.head_packed_sixteen_staged(
                            &stream,
                            rows(nr.div_ceil(16) * VOCAB / 8, 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 8 && staged {
                        module.head_packed_eight_staged(
                            &stream,
                            rows(nr.div_ceil(8) * VOCAB / 8, 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 8 {
                        module.head_packed_eight(
                            &stream,
                            rows((nr.div_ceil(8) * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else if group_size == 4 {
                        module.head_packed_four(
                            &stream,
                            rows((nr.div_ceil(4) * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                        )?;
                    } else {
                        module.head_packed_rows(
                            &stream,
                            rows((nr * VOCAB).div_ceil(8), 256),
                            &input,
                            &selected,
                            &head.words,
                            &head.offsets,
                            &head.exceptions,
                            &mut out,
                            0,
                        )?;
                    }
                    Ok(())
                })?
            };
            let mut times = Vec::new();
            for repetition in 0..10 {
                let _eviction = DeviceBuffer::<u8>::zeroed(&stream, 256 << 20)?;
                start.record(&stream)?;
                unsafe {
                    graph.launch()?;
                }
                end.record(&stream)?;
                let milliseconds = start.elapsed_ms(&end)?;
                if repetition >= 3 {
                    times.push(milliseconds);
                }
            }
            let actual = out.to_host_vec(&stream)?;
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "head {nr} rows");
            } else {
                expected = Some(actual);
            }
            println!(
                "HEAD32_PROFILE {}",
                serde_json::json!({"device":device_name, "model_root":model.root,
                "rows":nr, "group_size":group_size, "staged":staged, "milliseconds":times})
            );
        }
    }
    Ok(())
}
