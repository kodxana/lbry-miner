use anyhow::Result;

use crate::pow::LBRY_HEADER_LEN;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub platform_index: usize,
    pub device_index: usize,
    pub work_size: usize,
    pub batch_size: usize,
}

#[cfg(feature = "opencl")]
mod gpu {
    use std::fmt::Display;
    use std::ptr;

    use anyhow::{Context as AnyhowContext, Result, anyhow, bail};
    use opencl3::command_queue::CommandQueue;
    use opencl3::context::Context;
    use opencl3::device::{
        CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT,
        CL_DEVICE_TYPE_GPU, Device,
    };
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
    use opencl3::platform::get_platforms;
    use opencl3::program::Program;
    use opencl3::types::{CL_BLOCKING, cl_device_type, cl_uint, cl_ulong};

    use crate::opencl::SearchConfig;
    use crate::pow::{
        HeaderBytes, LBRY_HEADER_LEN, kernel_candidate_nonce, lbry_hash, wordswap_112,
    };

    const FOUND_INDEX: usize = 0xff;
    const OUTPUT_WORDS: usize = 0x100;
    const SHA256_KERNEL: &str = include_str!("../third_party/sgminer-gm-kernels/sha256.cl");
    const SHA512_KERNEL: &str = include_str!("../third_party/sgminer-gm-kernels/wolf-sha512.cl");
    const RIPEMD160_KERNEL: &str = include_str!("../third_party/sgminer-gm-kernels/ripemd160.cl");
    const LBRY_KERNEL: &str = include_str!("../third_party/sgminer-gm-kernels/lbry.cl");

    pub struct LbryOpenCl {
        queue: CommandQueue,
        search: Kernel,
        search1: Kernel,
        search2: Kernel,
        input: Buffer<cl_uint>,
        ctx: Buffer<cl_uint>,
        output: Buffer<cl_uint>,
        batch_size: usize,
        work_size: usize,
    }

    impl LbryOpenCl {
        pub fn new(
            platform_index: usize,
            device_index: usize,
            work_size: usize,
            batch_size: usize,
        ) -> Result<Self> {
            if work_size == 0 {
                bail!("work_size must be greater than zero");
            }
            if batch_size == 0 {
                bail!("batch_size must be greater than zero");
            }
            if batch_size % work_size != 0 {
                bail!("batch_size must be a multiple of work_size");
            }

            let platforms = get_platforms()?;
            let platform = platforms
                .get(platform_index)
                .ok_or_else(|| {
                    anyhow!(
                        "OpenCL platform {platform_index} was not found; run list-opencl to see available platform indexes"
                    )
                })?;
            let device_ids = platform.get_devices(CL_DEVICE_TYPE_ALL)?;
            let device_id = *device_ids
                .get(device_index)
                .ok_or_else(|| {
                    anyhow!(
                        "OpenCL device {device_index} was not found on platform {platform_index}; run list-opencl to see available device indexes"
                    )
                })?;
            let device = Device::new(device_id);
            let context = Context::from_device(&device)?;
            let queue = CommandQueue::create_default(&context, 0)?;

            let source = kernel_source(work_size);
            let program = Program::create_and_build_from_source(&context, &source, "")
                .map_err(|err| anyhow!("failed to build LBRY OpenCL kernel: {err}"))?;
            let search = Kernel::create(&program, "search")?;
            let search1 = Kernel::create(&program, "search1")?;
            let search2 = Kernel::create(&program, "search2")?;

            let input = unsafe {
                Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 28, ptr::null_mut())?
            };
            let ctx = unsafe {
                Buffer::<cl_uint>::create(
                    &context,
                    CL_MEM_READ_WRITE,
                    batch_size * 8,
                    ptr::null_mut(),
                )?
            };
            let output = unsafe {
                Buffer::<cl_uint>::create(
                    &context,
                    CL_MEM_WRITE_ONLY,
                    OUTPUT_WORDS,
                    ptr::null_mut(),
                )?
            };

            Ok(Self {
                queue,
                search,
                search1,
                search2,
                input,
                ctx,
                output,
                batch_size,
                work_size,
            })
        }

        pub fn search_batch(
            &mut self,
            header: &[u8; LBRY_HEADER_LEN],
            start_nonce: u32,
            target_tail: u64,
        ) -> Result<Vec<u32>> {
            let input_words = header_input_words(header);
            let mut output_words = [0u32; OUTPUT_WORDS];

            unsafe {
                self.queue.enqueue_write_buffer(
                    &mut self.input,
                    CL_BLOCKING,
                    0,
                    &input_words,
                    &[],
                )?;
                self.queue.enqueue_write_buffer(
                    &mut self.output,
                    CL_BLOCKING,
                    0,
                    &output_words,
                    &[],
                )?;

                ExecuteKernel::new(&self.search)
                    .set_arg(&self.input)
                    .set_arg(&self.ctx)
                    .set_global_work_offset(start_nonce as usize)
                    .set_global_work_size(self.batch_size)
                    .set_local_work_size(self.work_size)
                    .enqueue_nd_range(&self.queue)?
                    .wait()?;

                ExecuteKernel::new(&self.search1)
                    .set_arg(&self.ctx)
                    .set_global_work_offset(start_nonce as usize)
                    .set_global_work_size(self.batch_size)
                    .set_local_work_size(self.work_size)
                    .enqueue_nd_range(&self.queue)?
                    .wait()?;

                let target_tail: cl_ulong = target_tail;
                ExecuteKernel::new(&self.search2)
                    .set_arg(&self.ctx)
                    .set_arg(&self.output)
                    .set_arg(&target_tail)
                    .set_global_work_offset(start_nonce as usize)
                    .set_global_work_size(self.batch_size)
                    .set_local_work_size(self.work_size)
                    .enqueue_nd_range(&self.queue)?
                    .wait()?;

                self.queue.enqueue_read_buffer(
                    &self.output,
                    CL_BLOCKING,
                    0,
                    &mut output_words,
                    &[],
                )?;
            }

            let count = output_words[FOUND_INDEX] as usize;
            let safe_count = count.min(FOUND_INDEX);
            Ok(output_words[..safe_count].to_vec())
        }
    }

    pub fn list_devices() -> Result<()> {
        let platforms = get_platforms()?;
        if platforms.is_empty() {
            println!("No OpenCL platforms were found.");
            println!(
                "Install a GPU driver with OpenCL support, then re-run: cargo run --features opencl -- list-opencl"
            );
            return Ok(());
        }

        for (platform_index, platform) in platforms.iter().enumerate() {
            println!(
                "platform[{platform_index}]: {}",
                format_info(platform.name())
            );
            println!("  vendor: {}", format_info(platform.vendor()));
            println!("  version: {}", format_info(platform.version()));
            println!("  profile: {}", format_info(platform.profile()));

            let device_ids = platform.get_devices(CL_DEVICE_TYPE_ALL)?;
            if device_ids.is_empty() {
                println!("  devices: <none>");
                continue;
            }

            for (device_index, device_id) in device_ids.iter().enumerate() {
                let device = Device::new(*device_id);
                let vendor = format_info(device.vendor());
                println!("  device[{device_index}]: {}", format_info(device.name()));
                println!(
                    "    selection: --backend opencl --platform {platform_index} --device {device_index}"
                );
                println!("    type: {}", format_device_type(device.dev_type()));
                println!("    vendor: {vendor}");
                println!("    driver: {}", format_info(device.driver_version()));
                println!("    device OpenCL: {}", format_info(device.version()));
                println!("    OpenCL C: {}", format_info(device.opencl_c_version()));
                println!(
                    "    compute units: {}",
                    format_info(device.max_compute_units())
                );
                println!(
                    "    max work-group size: {}",
                    format_info(device.max_work_group_size())
                );
                println!(
                    "    preferred work-group multiple: {}",
                    format_optional_info(device.preferred_work_group_size_multiple())
                );
                println!(
                    "    max work-item sizes: {}",
                    format_vec_info(device.max_work_item_sizes())
                );
                println!(
                    "    global memory: {}",
                    format_bytes_info(device.global_mem_size())
                );
                println!(
                    "    max allocation: {}",
                    format_bytes_info(device.max_mem_alloc_size())
                );
                println!(
                    "    local memory: {}",
                    format_bytes_info(device.local_mem_size())
                );

                if vendor.to_ascii_lowercase().contains("nvidia") {
                    println!(
                        "    note: NVIDIA OpenCL is intended but not yet maintainer hardware-validated"
                    );
                }
            }
        }

        Ok(())
    }

    pub fn scan_batch(
        config: &SearchConfig,
        header: &[u8; LBRY_HEADER_LEN],
        start_nonce: u32,
        target_tail: u64,
    ) -> Result<Vec<u32>> {
        let mut miner = LbryOpenCl::new(
            config.platform_index,
            config.device_index,
            config.work_size,
            config.batch_size,
        )?;
        miner.search_batch(header, start_nonce, target_tail)
    }

    pub fn gpu_self_test(
        platform_index: usize,
        device_index: usize,
        work_size: usize,
        batch_size: usize,
        start_nonce: u32,
    ) -> Result<()> {
        let header = HeaderBytes::try_from_slice(&[0u8; LBRY_HEADER_LEN])?;
        let mut miner = LbryOpenCl::new(platform_index, device_index, work_size, batch_size)?;
        let easy_target = batch_size <= FOUND_INDEX;
        let target_tail = if easy_target { u64::MAX } else { 0 };
        let candidates = miner
            .search_batch(header.as_bytes(), start_nonce, target_tail)
            .context("OpenCL LBRY self-test search failed")?;

        if easy_target && candidates.len() != batch_size {
            bail!(
                "expected {batch_size} candidates with max target, got {}",
                candidates.len()
            );
        }

        if easy_target {
            let mut actual = candidates.clone();
            actual.sort_unstable();

            let mut expected = (0..batch_size)
                .map(|index| kernel_candidate_nonce(start_nonce.wrapping_add(index as u32)))
                .collect::<Vec<_>>();
            expected.sort_unstable();

            if actual != expected {
                bail!("candidate nonce set mismatch: got {actual:x?}, expected {expected:x?}");
            }
        }

        let first_hash = lbry_hash(header.as_bytes());
        println!(
            "gpu self-test ok: {} candidates, mode={}, first_nonce={}, zero_header_hash={}",
            candidates.len(),
            if easy_target {
                "easy-target"
            } else {
                "hard-target-smoke"
            },
            candidates
                .first()
                .map(|nonce| format!("{nonce:#010x}"))
                .unwrap_or_else(|| "none".to_owned()),
            hex::encode(first_hash)
        );

        Ok(())
    }

    fn header_input_words(header: &[u8; LBRY_HEADER_LEN]) -> [cl_uint; 28] {
        let swapped = wordswap_112(header);
        let mut words = [0u32; 28];
        for (chunk, word) in swapped.chunks_exact(4).zip(words.iter_mut()) {
            *word = u32::from_ne_bytes(chunk.try_into().expect("chunk length is fixed"));
        }
        words
    }

    fn kernel_source(work_size: usize) -> String {
        let lbry_without_includes = LBRY_KERNEL
            .lines()
            .filter(|line| !line.trim_start().starts_with("#include"))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "#define WORKSIZE {work_size}\n{SHA256_KERNEL}\n{SHA512_KERNEL}\n{RIPEMD160_KERNEL}\n{lbry_without_includes}\n"
        )
    }

    fn format_info<T, E>(result: std::result::Result<T, E>) -> String
    where
        T: Display,
        E: Display,
    {
        result
            .map(|value| value.to_string())
            .unwrap_or_else(|err| format!("<error: {err}>"))
    }

    fn format_vec_info<T, E>(result: std::result::Result<Vec<T>, E>) -> String
    where
        T: Display,
        E: Display,
    {
        result
            .map(|values| {
                values
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" x ")
            })
            .unwrap_or_else(|err| format!("<error: {err}>"))
    }

    fn format_optional_info<T, E>(result: std::result::Result<T, E>) -> String
    where
        T: Display,
        E: Display,
    {
        result
            .map(|value| value.to_string())
            .unwrap_or_else(|err| format!("<unavailable: {err}>"))
    }

    fn format_bytes_info<E>(result: std::result::Result<cl_ulong, E>) -> String
    where
        E: Display,
    {
        result
            .map(format_bytes)
            .unwrap_or_else(|err| format!("<error: {err}>"))
    }

    fn format_bytes(bytes: cl_ulong) -> String {
        const KIB: f64 = 1024.0;
        const MIB: f64 = KIB * 1024.0;
        const GIB: f64 = MIB * 1024.0;

        let bytes_f = bytes as f64;
        if bytes_f >= GIB {
            format!("{:.2} GiB ({bytes} bytes)", bytes_f / GIB)
        } else if bytes_f >= MIB {
            format!("{:.2} MiB ({bytes} bytes)", bytes_f / MIB)
        } else if bytes_f >= KIB {
            format!("{:.2} KiB ({bytes} bytes)", bytes_f / KIB)
        } else {
            format!("{bytes} bytes")
        }
    }

    fn format_device_type<E>(result: std::result::Result<cl_device_type, E>) -> String
    where
        E: Display,
    {
        let device_type = match result {
            Ok(device_type) => device_type,
            Err(err) => return format!("<error: {err}>"),
        };

        let mut names = Vec::new();
        if device_type & CL_DEVICE_TYPE_DEFAULT != 0 {
            names.push("DEFAULT");
        }
        if device_type & CL_DEVICE_TYPE_CPU != 0 {
            names.push("CPU");
        }
        if device_type & CL_DEVICE_TYPE_GPU != 0 {
            names.push("GPU");
        }
        if device_type & CL_DEVICE_TYPE_ACCELERATOR != 0 {
            names.push("ACCELERATOR");
        }

        if names.is_empty() {
            format!("0x{device_type:x}")
        } else {
            names.join("|")
        }
    }
}

#[cfg(feature = "opencl")]
pub fn list_devices() -> Result<()> {
    gpu::list_devices()
}

#[cfg(not(feature = "opencl"))]
pub fn list_devices() -> Result<()> {
    println!("OpenCL support is disabled. Re-run with: cargo run --features opencl -- list-opencl");
    Ok(())
}

#[cfg(feature = "opencl")]
pub fn gpu_self_test(
    platform_index: usize,
    device_index: usize,
    work_size: usize,
    batch_size: usize,
    start_nonce: u32,
) -> Result<()> {
    gpu::gpu_self_test(
        platform_index,
        device_index,
        work_size,
        batch_size,
        start_nonce,
    )
}

#[cfg(feature = "opencl")]
pub fn scan_batch(
    config: &SearchConfig,
    header: &[u8; LBRY_HEADER_LEN],
    start_nonce: u32,
    target_tail: u64,
) -> Result<Vec<u32>> {
    gpu::scan_batch(config, header, start_nonce, target_tail)
}

#[cfg(feature = "opencl")]
pub struct Scanner {
    inner: gpu::LbryOpenCl,
}

#[cfg(feature = "opencl")]
impl Scanner {
    pub fn new(config: &SearchConfig) -> Result<Self> {
        Ok(Self {
            inner: gpu::LbryOpenCl::new(
                config.platform_index,
                config.device_index,
                config.work_size,
                config.batch_size,
            )?,
        })
    }

    pub fn search_batch(
        &mut self,
        header: &[u8; LBRY_HEADER_LEN],
        start_nonce: u32,
        target_tail: u64,
    ) -> Result<Vec<u32>> {
        self.inner.search_batch(header, start_nonce, target_tail)
    }
}

#[cfg(not(feature = "opencl"))]
pub fn gpu_self_test(
    _platform_index: usize,
    _device_index: usize,
    _work_size: usize,
    _batch_size: usize,
    _start_nonce: u32,
) -> Result<()> {
    println!(
        "OpenCL support is disabled. Re-run with: cargo run --features opencl -- gpu-self-test"
    );
    Ok(())
}

#[cfg(not(feature = "opencl"))]
pub fn scan_batch(
    _config: &SearchConfig,
    _header: &[u8; LBRY_HEADER_LEN],
    _start_nonce: u32,
    _target_tail: u64,
) -> Result<Vec<u32>> {
    println!("OpenCL support is disabled. Re-run with: cargo run --features opencl -- ...");
    Ok(Vec::new())
}

#[cfg(not(feature = "opencl"))]
pub struct Scanner;

#[cfg(not(feature = "opencl"))]
impl Scanner {
    pub fn new(_config: &SearchConfig) -> Result<Self> {
        anyhow::bail!("OpenCL support is disabled. Re-run with --features opencl")
    }

    pub fn search_batch(
        &mut self,
        _header: &[u8; LBRY_HEADER_LEN],
        _start_nonce: u32,
        _target_tail: u64,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("OpenCL support is disabled. Re-run with --features opencl")
    }
}
