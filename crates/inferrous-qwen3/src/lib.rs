//! Standard Qwen3 with per-channel INT8 weights and dynamic per-token INT8 activations.
pub mod cache;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
mod graph;
#[cfg(feature = "gpu")]
mod kernels;
pub mod model;
#[cfg(feature = "gpu")]
mod packing;
#[cfg(feature = "gpu")]
mod tensor;

pub const MODEL_REVISION: &str = "b968826d9c46dd6066d109eabc6255188de91218";
pub const PROFILE: &str = "qwen3-8b-int8-channel-token-bf16-softmax-v1";
pub const HIDDEN: usize = 4096;
pub const VOCAB: usize = 151936;
pub const CONTEXT: usize = 4096;

#[cfg(all(test, feature = "gpu"))]
mod conformance;
