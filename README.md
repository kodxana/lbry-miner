# lbry-miner
Experimental Rust OpenCL/HIP miner for the LBRY proof-of-work algorithm.

## Notice

The initial OpenCL LBRY kernels are vendored from `sgminer-gm` under
`third_party/sgminer-gm-kernels`, so this project is licensed as
GPL-3.0-or-later.

## Status

Working pieces:

- Stratum subscribe/authorize/job parsing.
- LBRY proof-of-work verification and compact target utilities.
- OpenCL and HIP GPU backends.
- Local share validation before submit.
- Live mining loop with accepted/rejected share tracking.
- Reconnect handling and stale candidate discard.
- JSON config files.
- Log and TUI output modes.
- Basic heat/power throttling with hashrate caps or batch sleeps.

## Commands

```powershell
cargo run -- stratum-smoke --wallet <LBC_WALLET_ADDRESS> --worker rust --seconds 20
cargo run -- work-smoke --wallet <LBC_WALLET_ADDRESS> --worker rust --seconds 20
cargo run -- hash-header --header-hex <112-byte-header-hex>
cargo run -- suggest-diff --mhs 361 --share-seconds 45 --difficulty 1000

cargo run --features opencl -- list-opencl
cargo run --features opencl -- gpu-self-test
cargo run --features opencl -- gpu-work-smoke --wallet <LBC_WALLET_ADDRESS> --worker rust

cargo run --features opencl,hip -- list-hip --hip-arch gfx1201
cargo run --features opencl,hip -- hip-self-test --hip-arch gfx1201
cargo run --release --features opencl,hip -- mine --config .\configs\lbrypool.local.json --seconds 30 --dry-run
```

`--password` is accepted as an alias for `--worker` for compatibility with pool
terminology and older examples. On `lbrypool.net`, the password field can also
request lower share difficulty, for example `--worker d=1`.

## Config

`mine --config <path>` reads strict JSON. CLI flags override config values, and
missing values fall back to defaults. The included
[configs/lbrypool.json](configs/lbrypool.json) file is an example; copy it to a
local ignored file before adding your payout address.

```json
{
  "mine": {
    "backend": "hip",
    "url": "stratum+tcp://lbrypool.net:3334",
    "wallet": "YOUR_LBC_WALLET_ADDRESS",
    "password": "rust-miner",
    "seconds": 0,
    "platform": 0,
    "device": 0,
    "hip_arch": "gfx1201",
    "work_size": 256,
    "batch_size": 1048576,
    "start_nonce": 0,
    "dry_run": false,
    "ui": "tui",
    "target_mhs": null,
    "batch_sleep_ms": 0,
    "connect_timeout": 20,
    "reconnect_delay": 5
  }
}
```

`backend` can be `opencl`, `hip`, or `rocm` (`rocm` is an alias for `hip`). The
HIP backend compiles its kernel with HIPRTC at startup.

HIP diagnostic command:

```powershell
cargo run --features opencl,hip -- list-hip --hip-arch gfx1201
```

## Efficiency Controls

The miner can deliberately leave idle gaps between GPU batches to reduce heat
and power.

```powershell
.\target\release\lbc-miner.exe mine --config .\configs\lbrypool.local.json --target-mhs 300
.\target\release\lbc-miner.exe mine --config .\configs\lbrypool.local.json --batch-sleep-ms 2
```

`--target-mhs` is an effective hashrate cap. It computes how long each batch
should take for the requested hashrate and sleeps after fast batches.
`--batch-sleep-ms` always sleeps a fixed amount after each batch. You can combine
them.

Pool difficulty does not change hashrate or heat; it only changes share submit
frequency. Use `suggest-diff` to estimate share cadence:

```powershell
.\target\release\lbc-miner.exe suggest-diff --mhs 250 --share-seconds 45 --difficulty 1000
```

## Security Notes

- This is experimental mining software. Watch GPU temperatures and power limits.

