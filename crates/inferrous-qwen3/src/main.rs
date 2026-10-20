use anyhow::{Context, Result, ensure};
use inferrous_qwen3::{
    PROFILE,
    config::{CacheMode, EngineConfig},
    gpu::{BatchEngine, GpuModel},
    model::{Model, Request},
};
use std::{path::Path, time::Instant};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        !args.is_empty(),
        "usage: inferrous-qwen3 BUNDLE (--request FILE|--prompt TEXT|--batch FILE) [--max-tokens N] [--repeat N] [--engine-config FILE] [--cache-mode reuse|recompute] [--trace DIR]"
    );
    let mut prompt = None;
    let mut request_path = None;
    let mut batch_path = None;
    let mut engine_config = None;
    let mut cache_mode = CacheMode::Reuse;
    let mut max_tokens = None;
    let mut repeats = 1;
    let mut trace = None;
    let mut i = 1;
    while i < args.len() {
        let v = args.get(i + 1).context("missing option value")?;
        match args[i].as_str() {
            "--prompt" => {
                ensure!(prompt.is_none(), "duplicate prompt");
                prompt = Some(v);
            }
            "--request" => {
                ensure!(request_path.is_none(), "duplicate request");
                request_path = Some(v);
            }
            "--batch" => {
                ensure!(batch_path.is_none(), "duplicate batch");
                batch_path = Some(v);
            }
            "--engine-config" => {
                ensure!(engine_config.is_none(), "duplicate engine config");
                engine_config = Some(v);
            }
            "--cache-mode" => {
                cache_mode = match v.as_str() {
                    "reuse" => CacheMode::Reuse,
                    "recompute" => CacheMode::Recompute,
                    _ => anyhow::bail!("cache mode must be reuse or recompute"),
                }
            }
            "--max-tokens" => {
                ensure!(max_tokens.is_none(), "duplicate max-tokens");
                max_tokens = Some(v.parse::<usize>()?);
            }
            "--repeat" => repeats = v.parse::<usize>()?,
            "--trace" => trace = Some(Path::new(v)),
            _ => anyhow::bail!("unknown option {}", args[i]),
        }
        i += 2;
    }
    ensure!(
        usize::from(prompt.is_some())
            + usize::from(request_path.is_some())
            + usize::from(batch_path.is_some())
            == 1,
        "choose exactly one prompt source"
    );
    ensure!(
        (1..=100).contains(&repeats) && (trace.is_none() || repeats == 1),
        "invalid repeat count or trace with repeats"
    );
    ensure!(
        (request_path.is_none() && batch_path.is_none()) || max_tokens.is_none(),
        "request file carries max_tokens"
    );
    let start = Instant::now();
    let model = Model::load(Path::new(&args[0]))?;
    let requests: Vec<Request> = if let Some(p) = batch_path {
        serde_json::from_slice(&std::fs::read(p)?)?
    } else if let Some(p) = request_path {
        vec![serde_json::from_slice(&std::fs::read(p)?)?]
    } else {
        vec![Request {
            prompt_ids: model
                .tokenizer
                .encode(prompt.unwrap().as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec(),
            max_tokens: max_tokens.unwrap_or(32),
            stop_ids: vec![151643, 151645],
        }]
    };
    if let Some(config_path) = engine_config {
        ensure!(
            trace.is_none(),
            "tracing is only available in the legacy engine"
        );
        let config: EngineConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let engine = BatchEngine::new(&model, config)?;
        let load_seconds = start.elapsed().as_secs_f64();
        for run in 0..repeats {
            let start = Instant::now();
            let batch = engine.generate(&model, &requests, cache_mode)?;
            println!(
                "{}",
                serde_json::json!({"backend":"inferrous-paged", "profile":PROFILE,
                "model_root":model.root, "requests":requests, "run":run,
                "load_seconds":load_seconds, "generate_seconds":start.elapsed().as_secs_f64(),
                "engine_config":engine.config(), "device_name":engine.device_name(),
                "cache_mode":cache_mode, "batch":batch})
            );
        }
        return Ok(());
    }
    ensure!(
        batch_path.is_none() && cache_mode == CacheMode::Reuse,
        "batch or recompute mode requires --engine-config FILE"
    );
    let request = requests.into_iter().next().context("empty request")?;
    request.validate()?;
    let gpu = GpuModel::new(&model)?;
    let load_seconds = start.elapsed().as_secs_f64();
    for run in 0..repeats {
        let start = Instant::now();
        let generation = gpu.generate(&model, &request, trace)?;
        let generate_seconds = start.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({"backend":"cuda-oxide-clean","profile":PROFILE,"model_root":model.root,"request":request,"run":run,"load_seconds":load_seconds,"generate_seconds":generate_seconds,"trace":trace.is_some(),"generation":generation})
        );
    }
    Ok(())
}
