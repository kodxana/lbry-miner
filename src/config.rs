use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    pub mine: MineConfig,
}

impl ConfigFile {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse JSON config file {}", path.display()))
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MineConfig {
    pub backend: Option<String>,
    pub url: Option<String>,
    pub wallet: Option<String>,
    #[serde(alias = "password")]
    pub worker: Option<String>,
    pub seconds: Option<u64>,
    pub platform: Option<usize>,
    pub device: Option<usize>,
    pub hip_arch: Option<String>,
    pub work_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub start_nonce: Option<u32>,
    pub dry_run: Option<bool>,
    pub ui: Option<String>,
    pub target_mhs: Option<f64>,
    pub batch_sleep_ms: Option<u64>,
    pub connect_timeout: Option<u64>,
    pub reconnect_delay: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mine_config_with_password_alias() {
        let config: ConfigFile = serde_json::from_str(
            r#"{
                "mine": {
                    "backend": "opencl",
                    "url": "stratum+tcp://lbrypool.net:3334",
                    "wallet": "test-wallet",
                    "password": "d=1",
                    "seconds": 30,
                    "platform": 0,
                    "device": 1,
                    "hip_arch": "gfx1201",
                    "work_size": 256,
                    "batch_size": 1048576,
                    "start_nonce": 7,
                    "dry_run": true,
                    "ui": "tui",
                    "target_mhs": 300.0,
                    "batch_sleep_ms": 2,
                    "connect_timeout": 20,
                    "reconnect_delay": 2
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.mine.backend.as_deref(), Some("opencl"));
        assert_eq!(config.mine.wallet.as_deref(), Some("test-wallet"));
        assert_eq!(config.mine.worker.as_deref(), Some("d=1"));
        assert_eq!(config.mine.device, Some(1));
        assert_eq!(config.mine.hip_arch.as_deref(), Some("gfx1201"));
        assert_eq!(config.mine.dry_run, Some(true));
        assert_eq!(config.mine.ui.as_deref(), Some("tui"));
        assert_eq!(config.mine.target_mhs, Some(300.0));
        assert_eq!(config.mine.batch_sleep_ms, Some(2));
    }
}
