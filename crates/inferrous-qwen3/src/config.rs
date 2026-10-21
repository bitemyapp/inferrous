//! Explicit hardware and execution policy; benchmark output records all fields.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const NATIVE_CONTEXT: usize = 32768;
pub const PAGE_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum HardwareProfile {
    Rtx5090,
    #[serde(rename = "rtx-pro-6000")]
    RtxPro6000,
}

impl HardwareProfile {
    pub fn validate_device(self, name: &str) -> Result<()> {
        let matches = match self {
            Self::Rtx5090 => name.contains("RTX 5090"),
            Self::RtxPro6000 => name.contains("RTX PRO 6000"),
        };
        ensure!(
            matches,
            "hardware profile {self:?} does not match device {name}"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum CacheMode {
    Reuse,
    Recompute,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    pub hardware: HardwareProfile,
    pub max_context: usize,
    pub max_batch: usize,
    pub prefill_chunk: usize,
    /// Bound each score workspace while allowing larger short-context chunks.
    #[serde(default = "default_score_scratch")]
    pub score_scratch_bytes: usize,
    pub kv_cache_bytes: usize,
    pub prefix_cache_entries: usize,
    pub verify_logits: bool,
    #[serde(default = "enabled")]
    pub cuda_graphs: bool,
    /// Permit the measured hardware-specific projection dispatch. Disable for A/B runs.
    #[serde(default = "enabled")]
    pub tuned_projections: bool,
    /// Share K/V staging between GQA heads at measured profitable row counts.
    #[serde(default = "enabled")]
    pub tuned_attention: bool,
    /// Reuse packed head reconstruction across output rows on the RTX 5090.
    #[serde(default = "enabled")]
    pub tuned_head: bool,
}

fn enabled() -> bool {
    true
}

fn default_score_scratch() -> usize {
    1 << 30
}

impl EngineConfig {
    /// Capacity defaults, not a claim of measured optimal kernel tuning.
    pub fn for_hardware(hardware: HardwareProfile, max_context: usize) -> Self {
        let (max_batch, prefill_chunk, kv_gib, score_gib) = match hardware {
            HardwareProfile::Rtx5090 => (16, 1024, 12, 1),
            HardwareProfile::RtxPro6000 => (64, 512, 64, 4),
        };
        Self {
            hardware,
            max_context,
            max_batch,
            prefill_chunk,
            score_scratch_bytes: score_gib << 30,
            kv_cache_bytes: kv_gib * (1 << 30),
            prefix_cache_entries: 32,
            verify_logits: true,
            cuda_graphs: true,
            tuned_projections: hardware == HardwareProfile::Rtx5090,
            tuned_attention: hardware == HardwareProfile::Rtx5090,
            tuned_head: hardware == HardwareProfile::Rtx5090,
        }
    }

    pub fn validate(&self, model_context: usize) -> Result<()> {
        ensure!(
            (1..=NATIVE_CONTEXT).contains(&self.max_context),
            "invalid native context limit"
        );
        ensure!(
            self.max_context <= model_context,
            "bundle RoPE table is too short; extend the bundle context first"
        );
        ensure!(
            (1..=64).contains(&self.max_batch),
            "batch capacity must be 1..=64"
        );
        ensure!(
            (self.max_batch..=1024).contains(&self.prefill_chunk),
            "prefill chunk must cover the batch and be at most 1024 tokens"
        );
        let score_row_bytes = 32 * self.max_context.next_power_of_two().max(PAGE_TOKENS) * 4;
        ensure!(
            self.score_scratch_bytes / score_row_bytes >= self.max_batch,
            "score workspace budget cannot hold one query per admitted sequence"
        );
        ensure!(
            self.prefix_cache_entries <= 1024,
            "prefix cache entry limit exceeds 1024"
        );
        ensure!(
            self.kv_cache_bytes >= PAGE_TOKENS * crate::cache::KV_BYTES_PER_TOKEN,
            "KV budget cannot hold one page"
        );
        let pages = self.kv_cache_bytes / (PAGE_TOKENS * crate::cache::KV_BYTES_PER_TOKEN);
        ensure!(
            pages <= u32::MAX as usize / (PAGE_TOKENS * 1024),
            "KV layer addresses exceed the device index range"
        );
        Ok(())
    }
}
