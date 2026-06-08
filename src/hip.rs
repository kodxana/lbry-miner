use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::mem;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};

use crate::opencl::SearchConfig;
use crate::pow::{LBRY_HEADER_LEN, kernel_candidate_nonce, lbry_work_hash, wordswap_112};

const FOUND_INDEX: usize = 0xff;
const OUTPUT_WORDS: usize = 0x100;
const HIP_SUCCESS: HipError = 0;
const HIPRTC_SUCCESS: HiprtcResult = 0;
const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[derive(Debug, Clone)]
struct ToolProbe {
    name: &'static str,
    found: bool,
    ok: bool,
    summary: String,
}

#[derive(Debug, Clone)]
struct LibraryProbe {
    name: &'static str,
    path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    feature_enabled: bool,
    hip_path: Option<String>,
    rocm_path: Option<String>,
    requested_arch: Option<String>,
    tools: Vec<ToolProbe>,
    libraries: Vec<LibraryProbe>,
}

impl Diagnostics {
    pub fn detect(requested_arch: Option<&str>) -> Self {
        Self {
            feature_enabled: cfg!(feature = "hip"),
            hip_path: env::var("HIP_PATH").ok(),
            rocm_path: env::var("ROCM_PATH").ok(),
            requested_arch: requested_arch.map(str::to_owned),
            tools: vec![
                probe_tool("hipcc", &["--version"]),
                probe_tool("hipconfig", &["--full"]),
                probe_tool("rocminfo", &[]),
                probe_tool("hipInfo", &[]),
            ],
            libraries: probe_libraries(),
        }
    }

    fn has_required_runtime(&self) -> bool {
        self.libraries.iter().any(|probe| {
            probe.path.is_some()
                && matches!(
                    probe.name,
                    "amdhip64.dll" | "amdhip64_6.dll" | "amdhip64_7.dll" | "libamdhip64.so"
                )
        })
    }

    fn has_hiprtc(&self) -> bool {
        self.libraries.iter().any(|probe| {
            probe.path.is_some()
                && matches!(
                    probe.name,
                    "hiprtc.dll" | "hiprtc0507.dll" | "hiprtc0701.dll" | "libhiprtc.so"
                )
        })
    }

    fn has_compiler(&self) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.name == "hipcc" && tool.ok)
    }

    fn missing_summary(&self) -> String {
        let mut missing = Vec::new();
        if !self.feature_enabled {
            missing.push("cargo feature 'hip'");
        }
        if !self.has_compiler() {
            missing.push("hipcc");
        }
        if !self.has_required_runtime() {
            missing.push("HIP runtime library");
        }
        if !self.has_hiprtc() {
            missing.push("HIPRTC library");
        }
        if self.requested_arch.is_none() {
            missing.push("hip_arch, for example gfx1201");
        }

        if missing.is_empty() {
            "HIP toolchain detected, but the HIP scanner runtime is not implemented yet".to_owned()
        } else {
            format!("missing {}", missing.join(", "))
        }
    }

    pub fn print(&self) {
        println!("HIP diagnostics");
        println!("  cargo feature hip: {}", self.feature_enabled);
        println!(
            "  requested arch: {}",
            self.requested_arch.as_deref().unwrap_or("<not set>")
        );
        println!(
            "  HIP_PATH: {}",
            self.hip_path.as_deref().unwrap_or("<unset>")
        );
        println!(
            "  ROCM_PATH: {}",
            self.rocm_path.as_deref().unwrap_or("<unset>")
        );

        println!("  tools:");
        for tool in &self.tools {
            println!(
                "    {}: {} ({})",
                tool.name,
                if tool.ok {
                    "ok"
                } else if tool.found {
                    "failed"
                } else {
                    "missing"
                },
                tool.summary
            );
        }

        println!("  libraries:");
        for library in &self.libraries {
            println!(
                "    {}: {}",
                library.name,
                library
                    .path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<not found>".to_owned())
            );
        }
    }
}

pub fn list_devices(requested_arch: Option<&str>) -> Result<()> {
    let diagnostics = Diagnostics::detect(requested_arch);
    diagnostics.print();
    println!("  status: {}", diagnostics.missing_summary());
    Ok(())
}

pub fn gpu_self_test(
    device_index: usize,
    hip_arch: &str,
    work_size: usize,
    batch_size: usize,
    start_nonce: u32,
) -> Result<()> {
    if batch_size > FOUND_INDEX {
        bail!("HIP self-test batch_size must be <= {FOUND_INDEX}");
    }

    let config = SearchConfig {
        platform_index: 0,
        device_index,
        work_size,
        batch_size,
    };
    let mut scanner = Scanner::new(&config, Some(hip_arch))?;
    let header = [0u8; LBRY_HEADER_LEN];
    let (first_nonce, first_hash, first_tail) =
        run_self_test_case(&mut scanner, &header, start_nonce, batch_size, "zero")?;

    let mut patterned_header = [0u8; LBRY_HEADER_LEN];
    for (index, byte) in patterned_header[..108].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    let patterned_start = if start_nonce == 0x0102_0304 {
        start_nonce.wrapping_add(0x1010_1010)
    } else {
        0x0102_0304
    };
    run_self_test_case(
        &mut scanner,
        &patterned_header,
        patterned_start,
        batch_size,
        "patterned",
    )?;

    println!(
        "HIP self-test ok: cases=2 batch_size={} first_nonce={first_nonce:#010x}, first_hash={}, first_tail={first_tail:#018x}",
        batch_size,
        hex::encode(first_hash)
    );
    Ok(())
}

fn run_self_test_case(
    scanner: &mut Scanner,
    header: &[u8; LBRY_HEADER_LEN],
    start_nonce: u32,
    batch_size: usize,
    label: &str,
) -> Result<(u32, [u8; 32], u64)> {
    let mut candidates = scanner.search_batch(header, start_nonce, u64::MAX)?;
    candidates.sort_unstable();

    let mut expected = (0..batch_size)
        .map(|index| kernel_candidate_nonce(start_nonce.wrapping_add(index as u32)))
        .collect::<Vec<_>>();
    expected.sort_unstable();

    if candidates != expected {
        bail!(
            "HIP {label} max-target nonce set mismatch: got {candidates:x?}, expected {expected:x?}"
        );
    }

    let first_nonce = kernel_candidate_nonce(start_nonce);
    let mut first_header = *header;
    first_header[108..112].copy_from_slice(&first_nonce.to_le_bytes());
    let first_hash = lbry_work_hash(&first_header);
    let hip_first_hash = scanner.hash_one(header, start_nonce)?;
    if hip_first_hash != first_hash {
        bail!(
            "HIP {label} hash mismatch for nonce {first_nonce:#010x}: gpu={}, cpu={}",
            hex::encode(hip_first_hash),
            hex::encode(first_hash)
        );
    }

    let first_tail = u64::from_le_bytes(first_hash[24..32].try_into().expect("fixed tail"));
    let exact_candidates = scanner.search_batch(header, start_nonce, first_tail)?;
    if !exact_candidates.contains(&first_nonce) {
        bail!(
            "HIP {label} hash-tail smoke did not return first nonce {first_nonce:#010x} for tail {first_tail:#018x}"
        );
    }

    Ok((first_nonce, first_hash, first_tail))
}

const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

fn sha256_first_block_midstate(header_input: &[u8; LBRY_HEADER_LEN]) -> [u32; 8] {
    let mut block = [0u8; 64];
    block.copy_from_slice(&header_input[..64]);
    let mut state = SHA256_IV;
    sha256_compress_block(&mut state, &block);
    state
}

fn sha256_compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (word, chunk) in w[..16].iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes(chunk.try_into().expect("fixed SHA256 word"));
    }

    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    fn sha256_112_from_midstate_for_test(header_input: &[u8; LBRY_HEADER_LEN]) -> [u8; 32] {
        let mut state = sha256_first_block_midstate(header_input);
        let mut block = [0u8; 64];
        block[..48].copy_from_slice(&header_input[64..112]);
        block[48] = 0x80;
        block[62] = 0x03;
        block[63] = 0x80;
        sha256_compress_block(&mut state, &block);

        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    #[test]
    fn first_block_midstate_matches_sha256_for_112_byte_headers() {
        let mut header = [0u8; LBRY_HEADER_LEN];
        for (index, byte) in header.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(19).wrapping_add(7);
        }
        header[108..112].copy_from_slice(&0x7856_3412u32.to_le_bytes());

        let header_input = wordswap_112(&header);
        let actual = sha256_112_from_midstate_for_test(&header_input);
        let expected: [u8; 32] = Sha256::digest(header_input).into();

        assert_eq!(actual, expected);
    }
}

pub struct Scanner {
    _module: HipModule,
    header_buffer: DeviceBuffer,
    midstate_buffer: DeviceBuffer,
    output_buffer: DeviceBuffer,
    hash_output_buffer: DeviceBuffer,
    runtime: HipRuntime,
    function: HipFunction,
    hash_one_function: HipFunction,
    config: SearchConfig,
    _requested_arch: String,
}

impl Scanner {
    pub fn new(config: &SearchConfig, requested_arch: Option<&str>) -> Result<Self> {
        let diagnostics = Diagnostics::detect(requested_arch);
        if !diagnostics.feature_enabled
            || !diagnostics.has_compiler()
            || !diagnostics.has_required_runtime()
            || !diagnostics.has_hiprtc()
            || diagnostics.requested_arch.is_none()
        {
            diagnostics.print();
            bail!(
                "HIP backend is not ready on this system: {}",
                diagnostics.missing_summary()
            );
        }

        diagnostics.print();
        let arch = requested_arch
            .context("HIP backend requires hip_arch, for example gfx1201")?
            .to_owned();
        let runtime = HipRuntime::load()?;
        runtime.check(
            unsafe { (runtime.set_device)(config.device_index as c_int) },
            "hipSetDevice",
        )?;

        let rtc = HipRtc::load()?;
        let source = hip_kernel_source();
        let code = rtc.compile(&source, &arch, config.work_size)?;
        let module = runtime.load_module(&code)?;
        let function = runtime.get_function(module.raw, "lbry_search")?;
        let hash_one_function = runtime.get_function(module.raw, "lbry_hash_one")?;
        let header_buffer = DeviceBuffer::zeroed::<u8>(&runtime, LBRY_HEADER_LEN)?;
        let midstate_buffer = DeviceBuffer::zeroed::<u32>(&runtime, 8)?;
        let output_buffer = DeviceBuffer::zeroed::<u32>(&runtime, OUTPUT_WORDS)?;
        let hash_output_buffer = DeviceBuffer::zeroed::<u8>(&runtime, 32)?;

        Ok(Self {
            _module: module,
            header_buffer,
            midstate_buffer,
            output_buffer,
            hash_output_buffer,
            runtime,
            function,
            hash_one_function,
            config: config.clone(),
            _requested_arch: arch,
        })
    }

    pub fn search_batch(
        &mut self,
        header: &[u8; LBRY_HEADER_LEN],
        start_nonce: u32,
        target_tail: u64,
    ) -> Result<Vec<u32>> {
        if self.config.work_size == 0 {
            bail!("work_size must be greater than zero");
        }
        if self.config.batch_size == 0 {
            bail!("batch_size must be greater than zero");
        }

        let header_input = wordswap_112(header);
        let midstate = sha256_first_block_midstate(&header_input);
        let mut output_words = [0u32; OUTPUT_WORDS];
        self.header_buffer
            .copy_from_host(&self.runtime, &header_input)?;
        self.midstate_buffer
            .copy_from_host(&self.runtime, &midstate)?;
        self.output_buffer
            .copy_from_host(&self.runtime, &output_words)?;

        let batch_size_u32 =
            u32::try_from(self.config.batch_size).context("HIP batch_size must fit in 32 bits")?;
        let grid_size = self.config.batch_size.div_ceil(self.config.work_size) as c_uint;
        let block_size = self.config.work_size as c_uint;

        let mut header_ptr = self.header_buffer.ptr;
        let mut midstate_ptr = self.midstate_buffer.ptr;
        let mut output_ptr = self.output_buffer.ptr;
        let mut start_nonce_arg = start_nonce;
        let mut target_tail_arg = target_tail;
        let mut batch_size_arg = batch_size_u32;
        let mut params = [
            (&mut header_ptr as *mut *mut c_void).cast::<c_void>(),
            (&mut midstate_ptr as *mut *mut c_void).cast::<c_void>(),
            (&mut start_nonce_arg as *mut u32).cast::<c_void>(),
            (&mut target_tail_arg as *mut u64).cast::<c_void>(),
            (&mut batch_size_arg as *mut u32).cast::<c_void>(),
            (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
        ];

        self.runtime.check(
            unsafe {
                (self.runtime.module_launch_kernel)(
                    self.function.raw,
                    grid_size,
                    1,
                    1,
                    block_size,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )?;
        self.runtime.check(
            unsafe { (self.runtime.device_synchronize)() },
            "hipDeviceSynchronize",
        )?;
        self.output_buffer
            .copy_to_host(&self.runtime, &mut output_words)?;

        let count = output_words[FOUND_INDEX] as usize;
        let safe_count = count.min(FOUND_INDEX);
        Ok(output_words[..safe_count].to_vec())
    }

    fn hash_one(&mut self, header: &[u8; LBRY_HEADER_LEN], start_nonce: u32) -> Result<[u8; 32]> {
        let header_input = wordswap_112(header);
        let midstate = sha256_first_block_midstate(&header_input);
        self.header_buffer
            .copy_from_host(&self.runtime, &header_input)?;
        self.midstate_buffer
            .copy_from_host(&self.runtime, &midstate)?;

        let mut header_ptr = self.header_buffer.ptr;
        let mut midstate_ptr = self.midstate_buffer.ptr;
        let mut output_ptr = self.hash_output_buffer.ptr;
        let mut start_nonce_arg = start_nonce;
        let mut params = [
            (&mut header_ptr as *mut *mut c_void).cast::<c_void>(),
            (&mut midstate_ptr as *mut *mut c_void).cast::<c_void>(),
            (&mut start_nonce_arg as *mut u32).cast::<c_void>(),
            (&mut output_ptr as *mut *mut c_void).cast::<c_void>(),
        ];

        self.runtime.check(
            unsafe {
                (self.runtime.module_launch_kernel)(
                    self.hash_one_function.raw,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel lbry_hash_one",
        )?;
        self.runtime.check(
            unsafe { (self.runtime.device_synchronize)() },
            "hipDeviceSynchronize lbry_hash_one",
        )?;

        let mut out = [0u8; 32];
        self.hash_output_buffer
            .copy_to_host(&self.runtime, &mut out)?;
        Ok(out)
    }
}

type HipError = c_int;
type HiprtcResult = c_int;
type HiprtcProgram = *mut c_void;
type HipModuleRaw = *mut c_void;
type HipFunctionRaw = *mut c_void;
type HipStream = *mut c_void;

struct HipRuntime {
    _lib: dylib::Library,
    set_device: unsafe extern "C" fn(c_int) -> HipError,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError,
    free: unsafe extern "C" fn(*mut c_void) -> HipError,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipError,
    device_synchronize: unsafe extern "C" fn() -> HipError,
    get_error_string: unsafe extern "C" fn(HipError) -> *const c_char,
    module_load_data: unsafe extern "C" fn(*mut HipModuleRaw, *const c_void) -> HipError,
    module_get_function:
        unsafe extern "C" fn(*mut HipFunctionRaw, HipModuleRaw, *const c_char) -> HipError,
    module_launch_kernel: unsafe extern "C" fn(
        HipFunctionRaw,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipError,
    module_unload: unsafe extern "C" fn(HipModuleRaw) -> HipError,
}

impl HipRuntime {
    fn load() -> Result<Self> {
        let path = find_first_library(&["amdhip64_7.dll", "amdhip64.dll", "amdhip64_6.dll"])
            .context("failed to find HIP runtime DLL")?;
        let lib = dylib::Library::load(&path)?;
        unsafe {
            Ok(Self {
                set_device: lib.symbol("hipSetDevice")?,
                malloc: lib.symbol("hipMalloc")?,
                free: lib.symbol("hipFree")?,
                memcpy: lib.symbol("hipMemcpy")?,
                device_synchronize: lib.symbol("hipDeviceSynchronize")?,
                get_error_string: lib.symbol("hipGetErrorString")?,
                module_load_data: lib.symbol("hipModuleLoadData")?,
                module_get_function: lib.symbol("hipModuleGetFunction")?,
                module_launch_kernel: lib.symbol("hipModuleLaunchKernel")?,
                module_unload: lib.symbol("hipModuleUnload")?,
                _lib: lib,
            })
        }
    }

    fn check(&self, code: HipError, operation: &str) -> Result<()> {
        if code == HIP_SUCCESS {
            return Ok(());
        }

        let message = unsafe {
            let ptr = (self.get_error_string)(code);
            if ptr.is_null() {
                format!("HIP error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(anyhow!("{operation} failed: {message} ({code})"))
    }

    fn load_module(&self, code: &[u8]) -> Result<HipModule> {
        let mut module = ptr::null_mut();
        self.check(
            unsafe { (self.module_load_data)(&mut module, code.as_ptr().cast::<c_void>()) },
            "hipModuleLoadData",
        )?;
        Ok(HipModule {
            raw: module,
            unload: self.module_unload,
        })
    }

    fn get_function(&self, module: HipModuleRaw, name: &str) -> Result<HipFunction> {
        let name = CString::new(name)?;
        let mut function = ptr::null_mut();
        self.check(
            unsafe { (self.module_get_function)(&mut function, module, name.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(HipFunction { raw: function })
    }
}

struct HipModule {
    raw: HipModuleRaw,
    unload: unsafe extern "C" fn(HipModuleRaw) -> HipError,
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { (self.unload)(self.raw) };
        }
    }
}

struct HipFunction {
    raw: HipFunctionRaw,
}

struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
    free: unsafe extern "C" fn(*mut c_void) -> HipError,
}

impl DeviceBuffer {
    fn zeroed<T>(runtime: &HipRuntime, count: usize) -> Result<Self> {
        let bytes = count
            .checked_mul(mem::size_of::<T>())
            .context("device allocation size overflow")?;
        let mut ptr = ptr::null_mut();
        runtime.check(unsafe { (runtime.malloc)(&mut ptr, bytes) }, "hipMalloc")?;
        Ok(Self {
            ptr,
            bytes,
            free: runtime.free,
        })
    }

    fn copy_from_host<T>(&self, runtime: &HipRuntime, slice: &[T]) -> Result<()> {
        let bytes = slice
            .len()
            .checked_mul(mem::size_of::<T>())
            .context("host-to-device copy size overflow")?;
        if bytes > self.bytes {
            bail!("host-to-device copy exceeds device buffer");
        }
        runtime.check(
            unsafe {
                (runtime.memcpy)(
                    self.ptr,
                    slice.as_ptr().cast::<c_void>(),
                    bytes,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy HostToDevice",
        )
    }

    fn copy_to_host<T>(&self, runtime: &HipRuntime, out: &mut [T]) -> Result<()> {
        let bytes = out
            .len()
            .checked_mul(mem::size_of::<T>())
            .context("device-to-host copy size overflow")?;
        if bytes > self.bytes {
            bail!("device-to-host copy exceeds device buffer");
        }
        runtime.check(
            unsafe {
                (runtime.memcpy)(
                    out.as_mut_ptr().cast::<c_void>(),
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy DeviceToHost",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { (self.free)(self.ptr) };
        }
    }
}

struct HipRtc {
    _lib: dylib::Library,
    create_program: unsafe extern "C" fn(
        *mut HiprtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> HiprtcResult,
    destroy_program: unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult,
    compile_program:
        unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HiprtcResult,
    get_program_log_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    get_program_log: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    get_code_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    get_code: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    get_error_string: unsafe extern "C" fn(HiprtcResult) -> *const c_char,
}

impl HipRtc {
    fn load() -> Result<Self> {
        let path = find_first_library(&["hiprtc0701.dll", "hiprtc.dll", "hiprtc0507.dll"])
            .context("failed to find HIPRTC DLL")?;
        let lib = dylib::Library::load(&path)?;
        unsafe {
            Ok(Self {
                create_program: lib.symbol("hiprtcCreateProgram")?,
                destroy_program: lib.symbol("hiprtcDestroyProgram")?,
                compile_program: lib.symbol("hiprtcCompileProgram")?,
                get_program_log_size: lib.symbol("hiprtcGetProgramLogSize")?,
                get_program_log: lib.symbol("hiprtcGetProgramLog")?,
                get_code_size: lib.symbol("hiprtcGetCodeSize")?,
                get_code: lib.symbol("hiprtcGetCode")?,
                get_error_string: lib.symbol("hiprtcGetErrorString")?,
                _lib: lib,
            })
        }
    }

    fn compile(&self, source: &str, arch: &str, work_size: usize) -> Result<Vec<u8>> {
        let source = CString::new(source)?;
        let name = CString::new("lbry_hip_scalar.cpp")?;
        let mut program = ptr::null_mut();
        self.check(
            unsafe {
                (self.create_program)(
                    &mut program,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "hiprtcCreateProgram",
        )?;
        let mut program = HipRtcProgram {
            rtc: self,
            raw: program,
        };

        let arch_option = CString::new(format!("--gpu-architecture={arch}"))?;
        let std_option = CString::new("-std=c++17")?;
        let opt_option = CString::new("-O3")?;
        let work_size_option = CString::new(format!("-DWORKSIZE={work_size}"))?;
        let options = [
            arch_option.as_ptr(),
            std_option.as_ptr(),
            opt_option.as_ptr(),
            work_size_option.as_ptr(),
        ];

        let compile_result = unsafe {
            (self.compile_program)(program.raw, options.len() as c_int, options.as_ptr())
        };
        if compile_result != HIPRTC_SUCCESS {
            let log = program
                .log()
                .unwrap_or_else(|err| format!("failed to read log: {err}"));
            self.check(compile_result, &format!("hiprtcCompileProgram\n{log}"))?;
        }

        let log = program.log()?;
        if !log.trim().is_empty() {
            println!("HIPRTC compile log:\n{log}");
        }

        program.code()
    }

    fn check(&self, code: HiprtcResult, operation: &str) -> Result<()> {
        if code == HIPRTC_SUCCESS {
            return Ok(());
        }

        let message = unsafe {
            let ptr = (self.get_error_string)(code);
            if ptr.is_null() {
                format!("HIPRTC error {code}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(anyhow!("{operation} failed: {message} ({code})"))
    }
}

struct HipRtcProgram<'a> {
    rtc: &'a HipRtc,
    raw: HiprtcProgram,
}

impl HipRtcProgram<'_> {
    fn log(&mut self) -> Result<String> {
        let mut size = 0usize;
        self.rtc.check(
            unsafe { (self.rtc.get_program_log_size)(self.raw, &mut size) },
            "hiprtcGetProgramLogSize",
        )?;
        if size == 0 {
            return Ok(String::new());
        }

        let mut bytes = vec![0u8; size];
        self.rtc.check(
            unsafe { (self.rtc.get_program_log)(self.raw, bytes.as_mut_ptr().cast::<c_char>()) },
            "hiprtcGetProgramLog",
        )?;
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn code(&mut self) -> Result<Vec<u8>> {
        let mut size = 0usize;
        self.rtc.check(
            unsafe { (self.rtc.get_code_size)(self.raw, &mut size) },
            "hiprtcGetCodeSize",
        )?;
        let mut code = vec![0u8; size];
        self.rtc.check(
            unsafe { (self.rtc.get_code)(self.raw, code.as_mut_ptr().cast::<c_char>()) },
            "hiprtcGetCode",
        )?;
        Ok(code)
    }
}

impl Drop for HipRtcProgram<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { (self.rtc.destroy_program)(&mut self.raw) };
        }
    }
}

fn probe_tool(name: &'static str, args: &[&str]) -> ToolProbe {
    let command = find_executable(name).unwrap_or_else(|| PathBuf::from(name));
    match Command::new(&command).args(args).output() {
        Ok(output) => {
            let mut text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if text.is_empty() {
                text = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            }
            if text.len() > 140 {
                text.truncate(140);
                text.push_str("...");
            }
            ToolProbe {
                name,
                found: true,
                ok: output.status.success(),
                summary: if text.is_empty() {
                    format!("{}; exit status {}", command.display(), output.status)
                } else {
                    format!("{}; {text}", command.display())
                },
            }
        }
        Err(err) => ToolProbe {
            name,
            found: false,
            ok: false,
            summary: err.to_string(),
        },
    }
}

fn probe_libraries() -> Vec<LibraryProbe> {
    runtime_library_names()
        .iter()
        .map(|name| LibraryProbe {
            name,
            path: find_library(name),
        })
        .collect()
}

#[cfg(windows)]
fn runtime_library_names() -> &'static [&'static str] {
    &[
        "amdhip64.dll",
        "amdhip64_6.dll",
        "amdhip64_7.dll",
        "hiprtc.dll",
        "hiprtc0507.dll",
        "hiprtc0701.dll",
        "hiprtc-builtins0701.dll",
    ]
}

#[cfg(not(windows))]
fn runtime_library_names() -> &'static [&'static str] {
    &["libamdhip64.so", "libhiprtc.so"]
}

fn find_library(name: &str) -> Option<PathBuf> {
    let dirs = if is_hip_runtime_library(name) {
        hip_runtime_candidate_dirs()
    } else {
        candidate_dirs()
    };
    find_library_in_dirs(name, dirs)
}

fn find_first_library(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|name| find_library(name))
}

fn find_library_in_dirs(name: &str, dirs: Vec<PathBuf>) -> Option<PathBuf> {
    dirs.into_iter()
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

fn is_hip_runtime_library(name: &str) -> bool {
    matches!(
        name,
        "amdhip64.dll" | "amdhip64_6.dll" | "amdhip64_7.dll" | "libamdhip64.so"
    )
}

fn find_executable(name: &str) -> Option<PathBuf> {
    executable_names(name)
        .into_iter()
        .flat_map(|candidate| {
            candidate_dirs()
                .into_iter()
                .map(move |dir| dir.join(&candidate))
        })
        .find(|path| path.is_file())
}

#[cfg(windows)]
fn executable_names(name: &str) -> Vec<String> {
    if name.ends_with(".exe") || name.ends_with(".bat") {
        vec![name.to_owned()]
    } else {
        vec![
            format!("{name}.exe"),
            format!("{name}.bat"),
            name.to_owned(),
        ]
    }
}

#[cfg(not(windows))]
fn executable_names(name: &str) -> Vec<String> {
    vec![name.to_owned()]
}

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    for var in ["HIP_PATH", "ROCM_PATH"] {
        if let Ok(value) = env::var(var) {
            let root = PathBuf::from(value);
            dirs.push(root.clone());
            dirs.push(root.join("bin"));
            dirs.push(root.join("lib"));
            dirs.push(root.join("lib64"));
        }
    }

    if let Some(paths) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&paths));
    }

    #[cfg(windows)]
    {
        dirs.push(PathBuf::from(r"C:\Program Files\AMD\ROCm\bin"));
        if let Ok(entries) = std::fs::read_dir(r"C:\Program Files\AMD\ROCm") {
            for entry in entries.flatten() {
                let path = entry.path();
                dirs.push(path.clone());
                dirs.push(path.join("bin"));
            }
        }
    }

    #[cfg(not(windows))]
    {
        dirs.push(PathBuf::from("/opt/rocm/lib"));
        dirs.push(PathBuf::from("/opt/rocm/lib64"));
        dirs.push(PathBuf::from("/opt/rocm/bin"));
    }

    dedup_existing_dirs(dirs)
}

#[cfg(windows)]
fn hip_runtime_candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    for var in ["SystemRoot", "WINDIR"] {
        if let Ok(value) = env::var(var) {
            dirs.push(PathBuf::from(value).join("System32"));
        }
    }

    if let Some(paths) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&paths));
    }

    for var in ["HIP_PATH", "ROCM_PATH"] {
        if let Ok(value) = env::var(var) {
            let root = PathBuf::from(value);
            dirs.push(root.clone());
            dirs.push(root.join("bin"));
        }
    }

    dirs.push(PathBuf::from(r"C:\Program Files\AMD\ROCm\bin"));
    if let Ok(entries) = std::fs::read_dir(r"C:\Program Files\AMD\ROCm") {
        for entry in entries.flatten() {
            let path = entry.path();
            dirs.push(path.clone());
            dirs.push(path.join("bin"));
        }
    }

    dedup_existing_dirs(dirs)
}

#[cfg(not(windows))]
fn hip_runtime_candidate_dirs() -> Vec<PathBuf> {
    candidate_dirs()
}

fn dedup_existing_dirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        if !out
            .iter()
            .any(|existing| same_path(existing.as_path(), &dir))
        {
            out.push(dir);
        }
    }
    out
}

fn same_path(left: &Path, right: &Path) -> bool {
    if cfg!(windows) {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    } else {
        left == right
    }
}

fn hip_kernel_source() -> String {
    HIP_SCALAR_KERNEL.to_owned()
}

const HIP_SCALAR_KERNEL: &str = r#"
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32U - n)); }
__device__ __forceinline__ u64 rotr64(u64 x, u32 n) { return (x >> n) | (x << (64U - n)); }
__device__ __forceinline__ u32 bswap32(u32 x) { return __byte_perm(x, 0U, 0x0123); }
__device__ __forceinline__ u32 read_be32(const u8 *p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3]; }
__device__ __forceinline__ u32 read_le32(const u8 *p) { return ((u32)p[3] << 24) | ((u32)p[2] << 16) | ((u32)p[1] << 8) | (u32)p[0]; }
__device__ __forceinline__ void write_be32(u8 *p, u32 x) { p[0] = (u8)(x >> 24); p[1] = (u8)(x >> 16); p[2] = (u8)(x >> 8); p[3] = (u8)x; }
__device__ __forceinline__ void write_le32(u8 *p, u32 x) { p[0] = (u8)x; p[1] = (u8)(x >> 8); p[2] = (u8)(x >> 16); p[3] = (u8)(x >> 24); }
__device__ __forceinline__ u64 read_le64(const u8 *p) {
    return ((u64)p[0]) | ((u64)p[1] << 8) | ((u64)p[2] << 16) | ((u64)p[3] << 24) |
           ((u64)p[4] << 32) | ((u64)p[5] << 40) | ((u64)p[6] << 48) | ((u64)p[7] << 56);
}

__device__ __constant__ u32 SHA256_K[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

__device__ void sha256_transform(const u8 block[64], u32 state[8]) {
    u32 w[64];
    for (int i = 0; i < 16; ++i) w[i] = read_be32(block + i * 4);
    for (int i = 16; i < 64; ++i) {
        u32 s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        u32 s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    u32 a = state[0], b = state[1], c = state[2], d = state[3];
    u32 e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        u32 S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 temp1 = h + S1 + ch + SHA256_K[i] + w[i];
        u32 S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha256_bytes(const u8 *data, int len, u8 out[32]) {
    u32 state[8] = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    u8 padded[128];
    for (int i = 0; i < 128; ++i) padded[i] = 0;
    for (int i = 0; i < len; ++i) padded[i] = data[i];
    padded[len] = 0x80U;
    u64 bit_len = (u64)len * 8ULL;
    int total = ((len + 9 + 63) / 64) * 64;
    for (int i = 0; i < 8; ++i) padded[total - 1 - i] = (u8)(bit_len >> (8 * i));
    sha256_transform(padded, state);
    if (total > 64) sha256_transform(padded + 64, state);
    for (int i = 0; i < 8; ++i) write_be32(out + i * 4, state[i]);
}

__device__ void double_sha256_bytes(const u8 *data, int len, u8 out[32]) {
    u8 first[32];
    sha256_bytes(data, len, first);
    sha256_bytes(first, 32, out);
}

__device__ void sha256_112_from_midstate(const u8 header[112], const u32 midstate[8], u8 out[32]) {
    u32 state[8];
    for (int i = 0; i < 8; ++i) state[i] = midstate[i];

    u8 block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 48; ++i) block[i] = header[64 + i];
    block[48] = 0x80U;
    block[62] = 0x03U;
    block[63] = 0x80U;

    sha256_transform(block, state);
    for (int i = 0; i < 8; ++i) write_be32(out + i * 4, state[i]);
}

__device__ __constant__ u64 SHA512_K[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

__device__ u64 read_be64(const u8 *p) {
    return ((u64)p[0] << 56) | ((u64)p[1] << 48) | ((u64)p[2] << 40) | ((u64)p[3] << 32) |
           ((u64)p[4] << 24) | ((u64)p[5] << 16) | ((u64)p[6] << 8) | (u64)p[7];
}
__device__ void write_be64(u8 *p, u64 x) {
    for (int i = 0; i < 8; ++i) p[i] = (u8)(x >> (56 - 8 * i));
}

__device__ void sha512_32(const u8 data[32], u8 out[64]) {
    u64 w[80];
    u8 block[128];
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80U;
    block[126] = 0x01U;
    block[127] = 0x00U;
    for (int i = 0; i < 16; ++i) w[i] = read_be64(block + i * 8);
    for (int i = 16; i < 80; ++i) {
        u64 s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        u64 s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u64 h[8] = {
        0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL
    };
    u64 a=h[0], b=h[1], c=h[2], d=h[3], e=h[4], f=h[5], g=h[6], hh=h[7];
    for (int i = 0; i < 80; ++i) {
        u64 S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        u64 ch = (e & f) ^ ((~e) & g);
        u64 t1 = hh + S1 + ch + SHA512_K[i] + w[i];
        u64 S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        u64 maj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    for (int i = 0; i < 8; ++i) write_be64(out + i * 8, h[i]);
}

__device__ __forceinline__ u32 rol32(u32 x, u32 n) { return (x << n) | (x >> (32U - n)); }
__device__ __forceinline__ u32 rmd_f(int j, u32 x, u32 y, u32 z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}
__device__ __forceinline__ u32 rmd_k(int j) {
    if (j < 16) return 0x00000000U;
    if (j < 32) return 0x5a827999U;
    if (j < 48) return 0x6ed9eba1U;
    if (j < 64) return 0x8f1bbcdcU;
    return 0xa953fd4eU;
}
__device__ __forceinline__ u32 rmd_kp(int j) {
    if (j < 16) return 0x50a28be6U;
    if (j < 32) return 0x5c4dd124U;
    if (j < 48) return 0x6d703ef3U;
    if (j < 64) return 0x7a6d76e9U;
    return 0x00000000U;
}
__device__ __constant__ int RMD_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__device__ __constant__ int RMD_RP[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__device__ __constant__ int RMD_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__device__ __constant__ int RMD_SP[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

__device__ void ripemd160_32(const u8 data[32], u8 out[20]) {
    u8 block[64];
    for (int i = 0; i < 64; ++i) block[i] = 0;
    for (int i = 0; i < 32; ++i) block[i] = data[i];
    block[32] = 0x80U;
    block[56] = 0x00U; block[57] = 0x01U;
    u32 x[16];
    for (int i = 0; i < 16; ++i) x[i] = read_le32(block + i * 4);
    u32 h0=0x67452301U, h1=0xefcdab89U, h2=0x98badcfeU, h3=0x10325476U, h4=0xc3d2e1f0U;
    u32 al=h0, bl=h1, cl=h2, dl=h3, el=h4;
    u32 ar=h0, br=h1, cr=h2, dr=h3, er=h4;
    for (int j = 0; j < 80; ++j) {
        u32 t = rol32(al + rmd_f(j, bl, cl, dl) + x[RMD_R[j]] + rmd_k(j), RMD_S[j]) + el;
        al=el; el=dl; dl=rol32(cl, 10); cl=bl; bl=t;
        t = rol32(ar + rmd_f(79 - j, br, cr, dr) + x[RMD_RP[j]] + rmd_kp(j), RMD_SP[j]) + er;
        ar=er; er=dr; dr=rol32(cr, 10); cr=br; br=t;
    }
    u32 t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;
    write_le32(out, h0); write_le32(out + 4, h1); write_le32(out + 8, h2); write_le32(out + 12, h3); write_le32(out + 16, h4);
}

__device__ void lbry_hash_112(const u8 header[112], const u32 midstate[8], u8 out[32]) {
    u8 sha_b[32];
    u8 first_sha[32];
    sha256_112_from_midstate(header, midstate, first_sha);
    sha256_bytes(first_sha, 32, sha_b);
    u8 sha512[64];
    sha512_32(sha_b, sha512);
    u8 joined[40];
    ripemd160_32(sha512, joined);
    ripemd160_32(sha512 + 32, joined + 20);
    double_sha256_bytes(joined, 40, out);
}

extern "C" __global__
void lbry_hash_one(const u8 *header, const u32 *midstate, u32 start_nonce, u8 *hash_out) {
    u32 gid = start_nonce;
    u8 candidate[112];
    for (int i = 0; i < 108; ++i) candidate[i] = header[i];
    candidate[108] = (u8)gid;
    candidate[109] = (u8)(gid >> 8);
    candidate[110] = (u8)(gid >> 16);
    candidate[111] = (u8)(gid >> 24);

    u8 hash[32];
    lbry_hash_112(candidate, midstate, hash);
    for (int i = 0; i < 32; ++i) hash_out[i] = hash[i];
}

extern "C" __global__ __launch_bounds__(WORKSIZE)
void lbry_search(const u8 *header, const u32 *midstate, u32 start_nonce, u64 target_tail, u32 batch_size, u32 *output) {
    u32 local = (u32)(blockIdx.x * blockDim.x + threadIdx.x);
    if (local >= batch_size) return;
    u32 gid = start_nonce + local;
    u32 nonce = bswap32(gid);

    u8 candidate[112];
    for (int i = 0; i < 108; ++i) candidate[i] = header[i];
    candidate[108] = (u8)gid;
    candidate[109] = (u8)(gid >> 8);
    candidate[110] = (u8)(gid >> 16);
    candidate[111] = (u8)(gid >> 24);

    u8 hash[32];
    lbry_hash_112(candidate, midstate, hash);
    if (read_le64(hash + 24) <= target_tail) {
        u32 slot = atomicAdd(output + 0xff, 1U);
        if (slot < 0xff) output[slot] = nonce;
    }
}
"#;

#[cfg(windows)]
mod dylib {
    use std::ffi::{CString, c_char, c_void};
    use std::mem;
    use std::path::Path;

    use anyhow::{Context, Result, anyhow};

    type HModule = *mut c_void;
    type FarProc = *mut c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> HModule;
        fn GetProcAddress(module: HModule, name: *const c_char) -> FarProc;
        fn FreeLibrary(module: HModule) -> i32;
    }

    pub struct Library {
        handle: HModule,
    }

    impl Library {
        pub fn load(path: &Path) -> Result<Self> {
            let path = CString::new(path.to_string_lossy().as_bytes())
                .context("library path contains NUL byte")?;
            let handle = unsafe { LoadLibraryA(path.as_ptr()) };
            if handle.is_null() {
                return Err(anyhow!("failed to load {}", path.to_string_lossy()));
            }
            Ok(Self { handle })
        }

        pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
            let name = CString::new(name)?;
            let ptr = unsafe { GetProcAddress(self.handle, name.as_ptr()) };
            if ptr.is_null() {
                return Err(anyhow!("missing symbol {}", name.to_string_lossy()));
            }
            Ok(unsafe { mem::transmute_copy(&ptr) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                let _ = unsafe { FreeLibrary(self.handle) };
            }
        }
    }
}

#[cfg(not(windows))]
mod dylib {
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::mem;
    use std::path::Path;

    use anyhow::{Context, Result, anyhow};

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    pub struct Library {
        handle: *mut c_void,
    }

    impl Library {
        pub fn load(path: &Path) -> Result<Self> {
            let path = CString::new(path.to_string_lossy().as_bytes())
                .context("library path contains NUL byte")?;
            let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                return Err(anyhow!("failed to load {}", path.to_string_lossy()));
            }
            Ok(Self { handle })
        }

        pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
            let name = CString::new(name)?;
            let ptr = unsafe { dlsym(self.handle, name.as_ptr()) };
            if ptr.is_null() {
                return Err(anyhow!("missing symbol {}", name.to_string_lossy()));
            }
            Ok(unsafe { mem::transmute_copy(&ptr) })
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                let _ = unsafe { dlclose(self.handle) };
            }
        }
    }
}
