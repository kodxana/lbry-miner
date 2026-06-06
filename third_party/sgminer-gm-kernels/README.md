These OpenCL kernels are vendored from `sgminer-gm` to bootstrap the Rust LBRY miner GPU path.

Files:

- `lbry.cl`
- `sha256.cl`
- `wolf-sha512.cl`
- `ripemd160.cl`

The original project is GPLv3-or-later; see `COPYING` in this directory. Because these kernels are compiled into the miner at runtime, this Rust project is licensed as GPL-3.0-or-later while they are included.

