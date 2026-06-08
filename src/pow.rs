use anyhow::{Context, Result, anyhow, bail};
use ripemd::Ripemd160;
use sha2::{Digest, Sha256, Sha512};

pub const LBRY_HEADER_LEN: usize = 112;
const LBRY_DIFF_MULTIPLIER2: f64 = 256.0;
const TRUE_DIFF_ONE: f64 = 26959535291011309493156476344723991336010898738574164086137773096960.0;
const BITS_192: f64 = 6277101735386680763835789423207666416102355444464034512896.0;
const BITS_128: f64 = 340282366920938463463374607431768211456.0;
const BITS_64: f64 = 18446744073709551616.0;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BlockHeader {
    pub version: i32,
    pub prev_block: [u8; 32],
    pub merkle_root: [u8; 32],
    pub claim_trie_root: [u8; 32],
    pub time: u32,
    pub bits: u32,
    pub nonce: u32,
}

impl BlockHeader {
    pub fn serialize(&self) -> HeaderBytes {
        let mut out = [0u8; LBRY_HEADER_LEN];

        out[0..4].copy_from_slice(&self.version.to_le_bytes());
        out[4..36].copy_from_slice(&self.prev_block);
        out[36..68].copy_from_slice(&self.merkle_root);
        out[68..100].copy_from_slice(&self.claim_trie_root);
        out[100..104].copy_from_slice(&self.time.to_le_bytes());
        out[104..108].copy_from_slice(&self.bits.to_le_bytes());
        out[108..112].copy_from_slice(&self.nonce.to_le_bytes());

        HeaderBytes(out)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HeaderBytes([u8; LBRY_HEADER_LEN]);

impl HeaderBytes {
    pub fn from_hex(input: &str) -> Result<Self> {
        let decoded = hex::decode(input.trim()).context("header hex is invalid")?;
        Self::try_from_slice(&decoded)
    }

    pub fn try_from_slice(input: &[u8]) -> Result<Self> {
        if input.len() != LBRY_HEADER_LEN {
            bail!(
                "LBRY block header must be {LBRY_HEADER_LEN} bytes, got {}",
                input.len()
            );
        }

        let mut header = [0u8; LBRY_HEADER_LEN];
        header.copy_from_slice(input);
        Ok(Self(header))
    }

    pub fn as_bytes(&self) -> &[u8; LBRY_HEADER_LEN] {
        &self.0
    }
}

pub fn lbry_hash(header: &[u8; LBRY_HEADER_LEN]) -> [u8; 32] {
    let sha_b = double_sha256(header);
    let sha512 = Sha512::digest(sha_b);

    let ripemd_left = Ripemd160::digest(&sha512[..32]);
    let ripemd_right = Ripemd160::digest(&sha512[32..64]);

    let mut joined = [0u8; 40];
    joined[..20].copy_from_slice(&ripemd_left);
    joined[20..].copy_from_slice(&ripemd_right);

    double_sha256(&joined)
}

pub fn lbry_work_hash(work_data: &[u8; LBRY_HEADER_LEN]) -> [u8; 32] {
    lbry_hash(&wordswap_112(work_data))
}

pub fn double_sha256(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    Sha256::digest(first).into()
}

pub fn wordswap_112(input: &[u8; LBRY_HEADER_LEN]) -> [u8; LBRY_HEADER_LEN] {
    let mut out = [0u8; LBRY_HEADER_LEN];
    for (src, dst) in input.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }
    out
}

pub fn compact_to_target(bits: u32) -> Result<[u8; 32]> {
    let exponent = (bits >> 24) as usize;
    let mantissa = bits & 0x007f_ffff;

    if bits & 0x0080_0000 != 0 {
        return Err(anyhow!("negative compact target is invalid"));
    }

    let mut target = [0u8; 32];
    if exponent <= 3 {
        let value = mantissa >> (8 * (3 - exponent));
        target[0..4].copy_from_slice(&value.to_le_bytes());
        return Ok(target);
    }

    let offset = exponent - 3;
    if offset + 3 > 32 {
        return Err(anyhow!("compact target overflows 256 bits"));
    }

    target[offset] = (mantissa & 0xff) as u8;
    target[offset + 1] = ((mantissa >> 8) & 0xff) as u8;
    target[offset + 2] = ((mantissa >> 16) & 0xff) as u8;
    Ok(target)
}

pub fn lbry_share_target(difficulty: f64) -> Result<[u8; 32]> {
    if !difficulty.is_finite() || difficulty <= 0.0 {
        bail!("difficulty must be a positive finite number");
    }

    let mut value = LBRY_DIFF_MULTIPLIER2 * TRUE_DIFF_ONE / difficulty;
    let mut target = [0u8; 32];

    for (offset, divisor) in [(24, BITS_192), (16, BITS_128), (8, BITS_64), (0, 1.0)] {
        let chunk = (value / divisor) as u64;
        target[offset..offset + 8].copy_from_slice(&chunk.to_le_bytes());
        value -= (chunk as f64) * divisor;
    }

    Ok(target)
}

pub fn hash_meets_target(hash_le: &[u8; 32], target_le: &[u8; 32]) -> bool {
    for (hash, target) in hash_le.iter().zip(target_le.iter()).rev() {
        if hash < target {
            return true;
        }
        if hash > target {
            return false;
        }
    }

    true
}

pub fn target_tail64(target_le: &[u8; 32]) -> u64 {
    u64::from_le_bytes(target_le[24..32].try_into().expect("slice length is fixed"))
}

pub fn lbry_hash_difficulty(hash_le: &[u8; 32]) -> f64 {
    let value = le256_to_f64(hash_le);
    if value == 0.0 {
        f64::INFINITY
    } else {
        LBRY_DIFF_MULTIPLIER2 * TRUE_DIFF_ONE / value
    }
}

pub fn kernel_candidate_nonce(global_id: u32) -> u32 {
    global_id.swap_bytes()
}

fn le256_to_f64(value: &[u8; 32]) -> f64 {
    let mut out = 0.0;
    for (offset, divisor) in [(24, BITS_192), (16, BITS_128), (8, BITS_64), (0, 1.0)] {
        let chunk = u64::from_le_bytes(value[offset..offset + 8].try_into().expect("fixed chunk"));
        out += (chunk as f64) * divisor;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_lbry_header_to_112_bytes() {
        let header = BlockHeader {
            version: 0x20,
            prev_block: [1; 32],
            merkle_root: [2; 32],
            claim_trie_root: [3; 32],
            time: 0x6a20_714e,
            bits: 0x1a0e_988c,
            nonce: 42,
        };

        let bytes = header.serialize();
        assert_eq!(bytes.as_bytes().len(), LBRY_HEADER_LEN);
        assert_eq!(&bytes.as_bytes()[0..4], &0x20i32.to_le_bytes());
        assert_eq!(&bytes.as_bytes()[100..104], &0x6a20_714eu32.to_le_bytes());
        assert_eq!(&bytes.as_bytes()[104..108], &0x1a0e_988cu32.to_le_bytes());
        assert_eq!(&bytes.as_bytes()[108..112], &42u32.to_le_bytes());
    }

    #[test]
    fn rejects_wrong_header_size() {
        let err = HeaderBytes::try_from_slice(&[0u8; 111]).unwrap_err();
        assert!(err.to_string().contains("112 bytes"));
    }

    #[test]
    fn wordswap_matches_sgminer_flip112_shape() {
        let mut header = [0u8; LBRY_HEADER_LEN];
        header[0..8].copy_from_slice(&[0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13]);

        let swapped = wordswap_112(&header);
        assert_eq!(
            &swapped[0..8],
            &[0x03, 0x02, 0x01, 0x00, 0x13, 0x12, 0x11, 0x10]
        );
    }

    #[test]
    fn compact_target_compares_little_endian_hashes() {
        let target = compact_to_target(0x1d00ffff).unwrap();
        let mut below = [0u8; 32];
        below[26] = 0xfe;

        let mut above = [0u8; 32];
        above[28] = 0x01;

        assert!(hash_meets_target(&below, &target));
        assert!(!hash_meets_target(&above, &target));
        assert_eq!(target_tail64(&target), 0x0000_0000_ffff_0000);
    }

    #[test]
    fn lbry_share_target_matches_sgminer_for_pool_diff() {
        let target = lbry_share_target(75000.0).unwrap();
        assert_eq!(
            hex::encode(target),
            "000000000000000000000000000000000000000038ab3e575bb1df0000000000"
        );
        assert_eq!(target_tail64(&target), 0x0000_0000_00df_b15b);
    }

    #[test]
    fn estimates_hash_difficulty_from_little_endian_hash() {
        let target = lbry_share_target(1000.0).unwrap();
        let difficulty = lbry_hash_difficulty(&target);
        assert!((difficulty - 1000.0).abs() < 0.001, "{difficulty}");
    }

    #[test]
    fn kernel_candidate_nonce_matches_sgminer_lbry_output() {
        assert_eq!(kernel_candidate_nonce(0), 0);
        assert_eq!(kernel_candidate_nonce(1), 0x0100_0000);
        assert_eq!(kernel_candidate_nonce(0x1234_5678), 0x7856_3412);
    }

    #[test]
    fn hashes_zero_header_with_lbry_pow_chain() {
        let header = [0u8; LBRY_HEADER_LEN];
        let hash = lbry_hash(&header);
        assert_eq!(
            hex::encode(hash),
            "e3a4ac103496ea346431fb03e4011fba24d07dd829dfe802aecb69546796c79f"
        );
    }
}
