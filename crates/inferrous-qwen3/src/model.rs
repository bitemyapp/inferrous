use crate::{CONTEXT, HIDDEN, MODEL_REVISION, PROFILE, VOCAB};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    dtype: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    format_version: u32,
    profile: String,
    model_revision: String,
    max_context: usize,
    config: serde_json::Value,
    tokenizer_sha256: String,
    tensors: BTreeMap<String, Tensor>,
}
pub(crate) struct Matrix {
    pub(crate) w: Vec<i8>,
    pub(crate) scale: Vec<f32>,
    pub(crate) k: usize,
    pub(crate) n: usize,
}
pub(crate) struct Layer {
    pub(crate) gains: BTreeMap<String, Vec<f32>>,
    pub(crate) weights: BTreeMap<String, Matrix>,
}
pub struct Model {
    pub(crate) layers: Vec<Layer>,
    pub(crate) embedding: Vec<u16>,
    pub(crate) head: Vec<u16>,
    pub(crate) final_gain: Vec<f32>,
    pub(crate) rope: Vec<f32>,
    pub root: String,
    pub tokenizer: tokenizers::Tokenizer,
}
pub fn sha(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path.join("manifest.json"))?;
        let m: Manifest = serde_json::from_slice(&bytes)?;
        ensure!(
            m.format_version == 1
                && m.profile == PROFILE
                && m.model_revision == MODEL_REVISION
                && m.max_context == CONTEXT,
            "unsupported clean model profile"
        );
        for (key, n) in [
            ("hidden_size", 4096),
            ("intermediate_size", 12288),
            ("num_hidden_layers", 36),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("vocab_size", 151936),
        ] {
            ensure!(m.config[key].as_u64() == Some(n), "unsupported {key}");
        }
        ensure!(
            m.config["rms_norm_eps"].as_f64() == Some(1e-6)
                && m.config["rope_theta"].as_u64() == Some(1_000_000)
                && m.config["rope_scaling"].is_null()
                && m.config["attention_bias"] == false
                && m.config["use_sliding_window"] == false
                && m.config["tie_word_embeddings"] == false
                && m.config["hidden_act"] == "silu",
            "unsupported model operators"
        );
        let read = |name: &str, dtype: &str, shape: &[usize]| -> Result<Vec<u8>> {
            let t = m
                .tensors
                .get(name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(
                t.dtype == dtype && t.shape == shape,
                "{name}: wrong type or shape"
            );
            let bytes = fs::read(path.join(format!("{name}.bin")))?;
            let size = match dtype {
                "|i1" => 1,
                "<u2" => 2,
                "<f4" => 4,
                _ => unreachable!(),
            };
            ensure!(
                bytes.len() == shape.iter().product::<usize>() * size && sha(&bytes) == t.sha256,
                "{name}: size or hash mismatch"
            );
            Ok(bytes)
        };
        let floats = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let a: Vec<_> = read(name, "<f4", shape)?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(a.iter().all(|x| x.is_finite()), "{name}: nonfinite float");
            Ok(a)
        };
        let bfloats = |name: &str| -> Result<Vec<u16>> {
            let a: Vec<_> = read(name, "<u2", &[VOCAB, HIDDEN])?
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(
                a.iter().all(|&x| x & 0x7f80 != 0x7f80),
                "{name}: nonfinite BF16"
            );
            Ok(a)
        };
        let mut layers = Vec::new();
        for l in 0..36 {
            let mut gains = BTreeMap::new();
            let mut weights = BTreeMap::new();
            for (name, n) in [
                ("norm1", 4096),
                ("norm2", 4096),
                ("qnorm", 128),
                ("knorm", 128),
            ] {
                gains.insert(name.into(), floats(&format!("layer.{l}.{name}"), &[n])?);
            }
            for (name, n, k) in [
                ("q", 4096, 4096),
                ("k", 1024, 4096),
                ("v", 1024, 4096),
                ("o", 4096, 4096),
                ("gate", 12288, 4096),
                ("up", 12288, 4096),
                ("down", 4096, 12288),
            ] {
                let w = read(&format!("layer.{l}.{name}.w"), "|i1", &[n, k])?
                    .into_iter()
                    .map(|b| b as i8)
                    .collect::<Vec<_>>();
                ensure!(!w.contains(&-128), "invalid quantized weight");
                let scale = floats(&format!("layer.{l}.{name}.scale"), &[n])?;
                ensure!(scale.iter().all(|&s| s > 0.0), "invalid weight scale");
                weights.insert(name.into(), Matrix { w, scale, k, n });
            }
            layers.push(Layer { gains, weights });
        }
        let tokenizer_bytes = fs::read(path.join("tokenizer.json"))?;
        ensure!(
            sha(&tokenizer_bytes) == m.tokenizer_sha256,
            "tokenizer hash mismatch"
        );
        Ok(Self {
            layers,
            embedding: bfloats("embedding")?,
            head: bfloats("head")?,
            final_gain: floats("final_gain", &[HIDDEN])?,
            rope: floats("rope", &[CONTEXT, 128])?,
            root: sha(&bytes),
            tokenizer: tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    #[serde(default)]
    pub stop_ids: Vec<u32>,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.prompt_ids.is_empty()
                && self.max_tokens > 0
                && self
                    .prompt_ids
                    .len()
                    .checked_add(self.max_tokens)
                    .is_some_and(|n| n <= CONTEXT),
            "request exceeds context or is empty"
        );
        ensure!(
            self.prompt_ids
                .iter()
                .chain(&self.stop_ids)
                .all(|&id| (id as usize) < VOCAB),
            "token outside vocabulary"
        );
        Ok(())
    }
}
#[derive(Serialize)]
pub struct Generation {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: String,
    pub step_seconds: Vec<f64>,
    pub request_seconds: f64,
    pub logit_sha256: Vec<String>,
}
pub fn argmax(x: &[f32]) -> Result<u32> {
    ensure!(
        !x.is_empty() && x.iter().all(|v| v.is_finite()),
        "nonfinite or empty logits"
    );
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_and_ties() {
        let valid = Request {
            prompt_ids: vec![0],
            max_tokens: CONTEXT - 1,
            stop_ids: vec![(VOCAB - 1) as u32],
        };
        assert!(valid.validate().is_ok());
        for request in [
            Request {
                prompt_ids: vec![],
                ..valid.clone()
            },
            Request {
                max_tokens: 0,
                ..valid.clone()
            },
            Request {
                max_tokens: usize::MAX,
                ..valid.clone()
            },
            Request {
                stop_ids: vec![VOCAB as u32],
                ..valid.clone()
            },
        ] {
            assert!(request.validate().is_err());
        }
        assert!(
            Request {
                prompt_ids: vec![1],
                max_tokens: CONTEXT,
                stop_ids: vec![]
            }
            .validate()
            .is_err()
        );
        assert!(
            Request {
                prompt_ids: vec![VOCAB as u32],
                max_tokens: 1,
                stop_ids: vec![]
            }
            .validate()
            .is_err()
        );
        assert_eq!(argmax(&[1.0, 3.0, 3.0]).unwrap(), 1);
        assert!(argmax(&[f32::NAN]).is_err());
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[f32::INFINITY]).is_err());
        assert_eq!(argmax(&[-0.0, 0.0]).unwrap(), 0);
    }
}
