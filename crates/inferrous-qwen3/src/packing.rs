//! Lossless storage for BF16 head weights; this does not quantize model values.

pub(crate) struct PackedBf16 {
    pub words: Vec<u32>,
    pub offsets: Vec<u32>,
    pub exceptions: Vec<u16>,
}

impl PackedBf16 {
    /// Fixed model rows are multiples of 256; each chunk needs at most 256 escapes.
    pub fn new(input: &[u16]) -> Self {
        assert!(input.len().is_multiple_of(256));
        assert!(input.len() <= u32::MAX as usize);
        let mut packed = Self {
            words: vec![0; input.len() / 8 * 3],
            offsets: Vec::with_capacity(input.len() / 256),
            exceptions: Vec::new(),
        };
        for (chunk_index, chunk) in input.chunks_exact(256).enumerate() {
            let base = packed.exceptions.len();
            packed.offsets.push(base as u32);
            for (local, &value) in chunk.iter().enumerate() {
                let exponent = (value >> 7) & 255;
                let code = if (109..124).contains(&exponent) {
                    ((exponent - 109) << 8) | ((value >> 8) & 128) | (value & 127)
                } else {
                    let code = 0xf00 | (packed.exceptions.len() - base) as u16;
                    packed.exceptions.push(value);
                    code
                } as u32;
                // Interleave 32 lane streams, each containing eight 12-bit codes.
                let word = chunk_index * 96 + local % 32 + (local / 32 * 12 / 32) * 32;
                let shift = (local / 32 * 12) % 32;
                packed.words[word] |= code << shift;
                if shift > 20 {
                    packed.words[word + 32] |= code >> (32 - shift);
                }
            }
        }
        // Keep the device allocation nonempty even when there are no exceptions.
        if packed.exceptions.is_empty() {
            packed.exceptions.push(0);
        }
        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_bf16_bits_round_trip() {
        let mut input: Vec<u16> = (0..=u16::MAX).collect();
        // Mix rare/common values within chunks as well as testing all-escape chunks.
        input.extend((0..65536).map(|i| (i as u16).wrapping_mul(40503)));
        let packed = PackedBf16::new(&input);
        for (i, &expected) in input.iter().enumerate() {
            // Read two words as a 64-bit window independently of the writer's
            // crossing-word branch. All arithmetic here runs on the host.
            let word = i / 256 * 96 + i % 32 + ((i % 256 / 32) * 12 / 32) * 32;
            let pair = u64::from(packed.words[word])
                | (u64::from(*packed.words.get(word + 32).unwrap_or(&0)) << 32);
            let code = ((pair >> ((i % 256 / 32) * 12 % 32)) & 4095) as u16;
            let actual = if code >> 8 == 15 {
                packed.exceptions[packed.offsets[i / 256] as usize + (code & 255) as usize]
            } else {
                ((code & 128) << 8) | (((code >> 8) + 109) << 7) | (code & 127)
            };
            assert_eq!(actual, expected, "index {i}");
        }
    }
}
