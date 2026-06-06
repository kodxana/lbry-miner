<div align="center">

# lbry-miner

**Experimental Rust GPU miner for the LBRY proof-of-work algorithm.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)
![Backend: OpenCL](https://img.shields.io/badge/backend-OpenCL-2ea44f.svg)
![Backend: HIP/ROCm](https://img.shields.io/badge/backend-HIP%2FROCm-cc342d.svg)
![Status: Experimental](https://img.shields.io/badge/status-experimental-yellow.svg)
[![Last commit](https://img.shields.io/github/last-commit/kodxana/lbry-miner)](https://github.com/kodxana/lbry-miner/commits)
[![Issues](https://img.shields.io/github/issues/kodxana/lbry-miner)](https://github.com/kodxana/lbry-miner/issues)

</div>

`lbry-miner` is a work-in-progress Rust miner targeting LBRY proof-of-work with
OpenCL and HIP/ROCm GPU backends. It includes Stratum pool support, local share
validation, JSON configuration, reconnect handling, and basic heat/power control
through hashrate caps or batch sleeps.

> [!WARNING]
> This is experimental mining software. It may be unstable, inefficient, or
> incompatible with your hardware/driver setup. Monitor GPU temperature, power
> draw, fan behavior, and system stability while testing.

## Project status

Working pieces:

- Stratum subscribe, authorize, and job parsing.
- LBRY proof-of-work verification and compact target utilities.
- OpenCL and HIP GPU backends.
- Local share validation before submit.
- Live mining loop with accepted/rejected share tracking.
- Reconnect handling and stale candidate discard.
- Strict JSON config files.
- Log and TUI output modes.
- Basic heat/power throttling with hashrate caps or batch sleeps.

Still experimental:

- Cross-GPU performance tuning.
- Driver compatibility.
- Long-running stability.
- Better documentation for tested hardware and known-good driver versions.

## Requirements

- Rust toolchain with `cargo`.
- A GPU supported by at least one backend:
  - OpenCL for the `opencl` feature.
  - HIP/ROCm for the `hip` feature.
- A valid LBC wallet address for pool mining.
- Pool connection details, for example `lbrypool.net`.

## Quick start

Clone the repo and create a local config from the example:

```bash
git clone https://github.com/kodxana/lbry-miner.git
cd lbry-miner
cp configs/lbrypool.json configs/lbrypool.local.json
```

Edit `configs/lbrypool.local.json` and replace `YOUR_LBC_WALLET_ADDRESS` with
your payout address.

Check that the miner can see your GPU backend:

```powershell
cargo run --features opencl -- list-opencl
cargo run --features opencl,hip -- list-hip --hip-arch gfx1201
```

Run a short dry-run mining test:

```powershell
cargo run --release --features opencl,hip -- mine --config .\configs\lbrypool.local.json --seconds 30 --dry-run
```

Start mining without `--dry-run`:

```powershell
cargo run --release --features opencl,hip -- mine --config .\configs\lbrypool.local.json
```

## Commands

| Command | Purpose |
| --- | --- |
| `stratum-smoke` | Test Stratum subscribe/authorize flow against a pool. |
| `work-smoke` | Request work from the pool and validate basic job handling. |
| `hash-header` | Hash a raw 112-byte LBRY header. |
| `suggest-diff` | Estimate pool difficulty for a target share cadence. |
| `list-opencl` | List available OpenCL platforms/devices. |
| `gpu-self-test` | Run an OpenCL GPU self-test. |
| `gpu-work-smoke` | Run a short OpenCL GPU work test against a pool. |
| `list-hip` | List available HIP devices for the selected architecture. |
| `hip-self-test` | Run a HIP backend self-test. |
| `mine` | Start the live mining loop. |

Example command set:

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

## Configuration

`mine --config <path>` reads strict JSON. CLI flags override config values, and
missing values fall back to defaults.

The included [`configs/lbrypool.json`](configs/lbrypool.json) file is an example.
Copy it to a local ignored file before adding your payout address.

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

### Backend options

`backend` can be one of:

| Value | Meaning |
| --- | --- |
| `opencl` | Use the OpenCL backend. |
| `hip` | Use the HIP/ROCm backend. |
| `rocm` | Alias for `hip`. |

The HIP backend compiles its kernel with HIPRTC at startup.

HIP diagnostic command:

```powershell
cargo run --features opencl,hip -- list-hip --hip-arch gfx1201
```

## Efficiency controls

The miner can deliberately leave idle gaps between GPU batches to reduce heat
and power.

```powershell
.\target\release\lbc-miner.exe mine --config .\configs\lbrypool.local.json --target-mhs 300
.\target\release\lbc-miner.exe mine --config .\configs\lbrypool.local.json --batch-sleep-ms 2
```

`--target-mhs` is an effective hashrate cap. It computes how long each batch
should take for the requested hashrate and sleeps after fast batches.

`--batch-sleep-ms` always sleeps a fixed amount after each batch.

You can combine both options.

Pool difficulty does **not** change hashrate or heat. It only changes share
submit frequency. Use `suggest-diff` to estimate share cadence:

```powershell
.\target\release\lbc-miner.exe suggest-diff --mhs 250 --share-seconds 45 --difficulty 1000
```

## Troubleshooting

| Problem | Things to check |
| --- | --- |
| No GPU devices are listed | Run `list-opencl` or `list-hip`; check GPU drivers and OpenCL/HIP runtime installation. |
| HIP startup fails | Verify `--hip-arch`, for example `gfx1201`, and confirm that HIPRTC is available. |
| GPU gets too hot | Lower power limit externally, set `--target-mhs`, or add `--batch-sleep-ms`. |
| Shares are rejected | Check wallet address, pool URL, worker/password field, and requested difficulty. |
| Frequent stale work | Check network stability and pool latency; stale candidates should be discarded after reconnects. |
| Config fails to load | Validate that the JSON is strict JSON: no comments, trailing commas, or unquoted values. |

## Tested hardware

Known-good hardware and driver combinations should be documented here as testing
continues.

| GPU | Backend | OS | Driver/ROCm version | Status |
| --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO |

## Security notes

- Mining stresses hardware. Watch temperatures, power limits, and fan behavior.
- Do not commit your local config if it contains a real wallet address or private notes.
- Prefer local config files such as `configs/lbrypool.local.json` for personal settings.
- Treat pool URLs, binaries, and third-party kernels as code you should audit before trusting.

## License

The initial OpenCL LBRY kernels are vendored from `sgminer-gm` under
`third_party/sgminer-gm-kernels`, so this project is licensed as
**GPL-3.0-or-later**.
