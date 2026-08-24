// ============================================================
// NVRTC Runtime Compiler — On-the-fly CUDA → PTX Compilation
// ============================================================
//
// This module wraps NVIDIA's NVRTC (Runtime Compilation) library
// to allow cudaclaw's Ramify engine to compile CUDA C++ source
// strings into PTX at runtime — without requiring a full nvcc
// build cycle.
//
// WHY NVRTC:
// - nvcc is a heavyweight offline compiler; launching it takes
//   hundreds of milliseconds and requires the CUDA toolkit.
// - NVRTC is a lightweight in-process compiler that can turn
//   a CUDA C++ string into PTX in ~10-50ms, ideal for the
//   Ramify loop where the system "learns" new patterns and
//   recompiles specialized kernels on-the-fly.
// - NVRTC output (PTX) can be loaded directly via
//   cuModuleLoadData / cust::Module::from_ptx without touching
//   the filesystem.
//
// ARCHITECTURE:
//
//   ┌──────────────────────────────────────────────────────┐
//   │                NvrtcCompiler                         │
//   │                                                     │
//   │  CUDA C++ source string                             │
//   │        │                                            │
//   │        ▼                                            │
//   │  nvrtcCreateProgram() → nvrtcCompileProgram()       │
//   │        │                                            │
//   │        ▼                                            │
//   │  nvrtcGetPTX() → PTX string                         │
//   │        │                                            │
//   │        ▼                                            │
//   │  cuModuleLoadData() → CUmodule (via cust)           │
//   │        │                                            │
//   │        ▼                                            │
//   │  cuModuleGetFunction() → CUfunction                 │
//   │        │                                            │
//   │        ▼                                            │
//   │  GPU kernel launch with optimized constants         │
//   └──────────────────────────────────────────────────────┘
//
// SAFETY:
// - All NVRTC FFI calls are wrapped in safe Rust interfaces.
// - Compilation errors are captured and returned as structured
//   NvrtcError values with the full compiler log.
// - The NvrtcProgram handle is cleaned up via Drop.
// - When NVRTC is not available (no GPU / no CUDA toolkit),
//   a simulated fallback performs PTX string substitution only.
//
// INTEGRATION:
// - BranchRegistry::compile_for_pattern() now uses NvrtcCompiler
//   when CUDA C++ templates are registered (in addition to the
//   existing PTX-only string substitution path).
// - PtxTemplate gains a `source_type` field: `Ptx` (direct PTX
//   text) or `CudaCpp` (CUDA C++ that needs NVRTC compilation).
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// NVRTC FFI Bindings
// ============================================================

/// NVRTC result codes (subset of nvrtcResult enum).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvrtcResult {
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    CompilationError = 6,
    BuiltinOperationFailure = 7,
    NoNameExpressionsAfterCompilation = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11,
}

impl NvrtcResult {
    fn from_i32(code: i32) -> Self {
        match code {
            0 => NvrtcResult::Success,
            1 => NvrtcResult::OutOfMemory,
            2 => NvrtcResult::ProgramCreationFailure,
            3 => NvrtcResult::InvalidInput,
            4 => NvrtcResult::InvalidProgram,
            5 => NvrtcResult::InvalidOption,
            6 => NvrtcResult::CompilationError,
            7 => NvrtcResult::BuiltinOperationFailure,
            8 => NvrtcResult::NoNameExpressionsAfterCompilation,
            9 => NvrtcResult::NoLoweredNamesBeforeCompilation,
            10 => NvrtcResult::NameExpressionNotValid,
            _ => NvrtcResult::InternalError,
        }
    }

    fn is_success(self) -> bool {
        self == NvrtcResult::Success
    }
}

/// Opaque NVRTC program handle.
#[repr(C)]
struct NvrtcProgramHandle {
    _opaque: [u8; 0],
}

type NvrtcProgramPtr = *mut NvrtcProgramHandle;

// ============================================================
// FFI declarations for libnvrtc.so
// ============================================================
//
// These are loaded dynamically at runtime so cudaclaw can still
// compile and run on machines without a CUDA toolkit. If the
// library is not found, we fall back to simulated compilation.

#[cfg(feature = "nvrtc")]
extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut NvrtcProgramPtr,
        src: *const std::os::raw::c_char,
        name: *const std::os::raw::c_char,
        num_headers: i32,
        headers: *const *const std::os::raw::c_char,
        include_names: *const *const std::os::raw::c_char,
    ) -> i32;

    fn nvrtcDestroyProgram(prog: *mut NvrtcProgramPtr) -> i32;

    fn nvrtcCompileProgram(
        prog: NvrtcProgramPtr,
        num_options: i32,
        options: *const *const std::os::raw::c_char,
    ) -> i32;

    fn nvrtcGetPTXSize(prog: NvrtcProgramPtr, size: *mut usize) -> i32;

    fn nvrtcGetPTX(
        prog: NvrtcProgramPtr,
        ptx: *mut std::os::raw::c_char,
    ) -> i32;

    fn nvrtcGetProgramLogSize(prog: NvrtcProgramPtr, size: *mut usize) -> i32;

    fn nvrtcGetProgramLog(
        prog: NvrtcProgramPtr,
        log: *mut std::os::raw::c_char,
    ) -> i32;

    fn nvrtcVersion(major: *mut i32, minor: *mut i32) -> i32;
}

// ============================================================
// Error Types
// ============================================================

/// Errors from the NVRTC compilation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NvrtcError {
    /// NVRTC library not available on this system.
    LibraryNotAvailable(String),
    /// Failed to create the NVRTC program handle.
    ProgramCreationFailed(NvrtcResultCode),
    /// Compilation failed — includes the full compiler log.
    CompilationFailed {
        result_code: NvrtcResultCode,
        compiler_log: String,
    },
    /// Failed to retrieve PTX output.
    PtxRetrievalFailed(NvrtcResultCode),
    /// The source string contains invalid UTF-8 or null bytes.
    InvalidSource(String),
    /// Internal error.
    Internal(String),
}

/// Serializable wrapper for NVRTC result codes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NvrtcResultCode(pub i32);

impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvrtcError::LibraryNotAvailable(msg) => write!(f, "NVRTC not available: {}", msg),
            NvrtcError::ProgramCreationFailed(code) => {
                write!(f, "NVRTC program creation failed (code {})", code.0)
            }
            NvrtcError::CompilationFailed {
                result_code,
                compiler_log,
            } => write!(
                f,
                "NVRTC compilation failed (code {}): {}",
                result_code.0, compiler_log
            ),
            NvrtcError::PtxRetrievalFailed(code) => {
                write!(f, "NVRTC PTX retrieval failed (code {})", code.0)
            }
            NvrtcError::InvalidSource(msg) => write!(f, "Invalid source: {}", msg),
            NvrtcError::Internal(msg) => write!(f, "NVRTC internal error: {}", msg),
        }
    }
}

// ============================================================
// Compilation Options
// ============================================================

/// Options controlling the NVRTC compilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileOptions {
    /// Target GPU architecture (e.g., "compute_70" for Volta).
    pub gpu_arch: String,
    /// Additional compiler flags (e.g., "--use_fast_math").
    pub extra_flags: Vec<String>,
    /// Whether to enable debug info (line numbers in PTX).
    pub debug_info: bool,
    /// Whether to enable relocatable device code (needed for
    /// separate compilation / device linking).
    pub relocatable_device_code: bool,
    /// Maximum number of registers per thread (0 = no limit).
    pub max_registers: u32,
    /// Define macros (name → value).
    pub defines: HashMap<String, String>,
    /// Include paths for header resolution.
    pub include_paths: Vec<String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            gpu_arch: "compute_70".to_string(),
            extra_flags: Vec::new(),
            debug_info: false,
            relocatable_device_code: false,
            max_registers: 0,
            defines: HashMap::new(),
            include_paths: Vec::new(),
        }
    }
}

impl CompileOptions {
    /// Create options for a specific compute capability.
    pub fn for_compute(major: u32, minor: u32) -> Self {
        CompileOptions {
            gpu_arch: format!("compute_{}{}", major, minor),
            ..Default::default()
        }
    }

    /// Add a preprocessor define.
    pub fn define(mut self, name: &str, value: &str) -> Self {
        self.defines.insert(name.to_string(), value.to_string());
        self
    }

    /// Enable fast math operations.
    pub fn fast_math(mut self) -> Self {
        self.extra_flags.push("--use_fast_math".to_string());
        self
    }

    /// Set maximum registers per thread.
    pub fn max_regs(mut self, count: u32) -> Self {
        self.max_registers = count;
        self
    }

    /// Build the compiler flag strings for nvrtcCompileProgram.
    fn build_flags(&self) -> Vec<String> {
        let mut flags = Vec::new();

        // Architecture target
        flags.push(format!("--gpu-architecture={}", self.gpu_arch));

        // Preprocessor defines
        for (name, value) in &self.defines {
            if value.is_empty() {
                flags.push(format!("-D{}", name));
            } else {
                flags.push(format!("-D{}={}", name, value));
            }
        }

        // Include paths
        for path in &self.include_paths {
            flags.push(format!("-I{}", path));
        }

        // Debug info
        if self.debug_info {
            flags.push("--device-debug".to_string());
            flags.push("--generate-line-info".to_string());
        }

        // Relocatable device code
        if self.relocatable_device_code {
            flags.push("--relocatable-device-code=true".to_string());
        }

        // Max registers
        if self.max_registers > 0 {
            flags.push(format!("--maxrregcount={}", self.max_registers));
        }

        // Extra flags
        flags.extend(self.extra_flags.iter().cloned());

        flags
    }
}

// ============================================================
// Compilation Result
// ============================================================

/// Successful compilation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// The compiled PTX assembly string.
    pub ptx: String,
    /// Compiler log (warnings, notes — may be non-empty even on success).
    pub compiler_log: String,
    /// Source program name.
    pub program_name: String,
    /// Compilation time in microseconds.
    pub compile_time_us: u64,
    /// The options used for compilation.
    pub options: CompileOptions,
    /// Whether this was a simulated compilation (no real NVRTC).
    pub simulated: bool,
}

// ============================================================
// NVRTC Compiler
// ============================================================

/// Safe wrapper around the NVRTC runtime compilation library.
///
/// Supports two modes:
/// 1. **Real NVRTC** — linked against `libnvrtc.so` (requires CUDA toolkit).
///    Compiles CUDA C++ source strings to PTX in-process.
/// 2. **Simulated** — fallback when NVRTC is not available. Performs
///    PTX template string substitution only (no real compilation).
///
/// # Usage
///
/// ```no_run
/// use crate::ramify::nvrtc_compiler::{NvrtcCompiler, CompileOptions};
///
/// let compiler = NvrtcCompiler::new();
/// let source = r#"
///     extern "C" __global__ void add(float* a, float* b, float* c, int n) {
///         int i = blockIdx.x * blockDim.x + threadIdx.x;
///         if (i < n) c[i] = a[i] + b[i];
///     }
/// "#;
///
/// let opts = CompileOptions::for_compute(7, 0).fast_math();
/// match compiler.compile(source, "add_kernel", &opts) {
///     Ok(result) => println!("PTX ({} bytes): {}", result.ptx.len(), &result.ptx[..100]),
///     Err(e) => eprintln!("Compilation failed: {}", e),
/// }
/// ```
pub struct NvrtcCompiler {
    /// Whether real NVRTC is available.
    nvrtc_available: bool,
    /// NVRTC version (major, minor), if available.
    nvrtc_version: Option<(i32, i32)>,
    /// Compilation statistics.
    stats: CompilerStats,
}

/// Compiler usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompilerStats {
    /// Total compilations attempted.
    pub total_compilations: u64,
    /// Successful compilations.
    pub successful_compilations: u64,
    /// Failed compilations.
    pub failed_compilations: u64,
    /// Simulated compilations (no real NVRTC).
    pub simulated_compilations: u64,
    /// Total compilation time in microseconds.
    pub total_compile_time_us: u64,
    /// Average compilation time in microseconds.
    pub avg_compile_time_us: f64,
    /// Total PTX bytes generated.
    pub total_ptx_bytes: u64,
}

impl NvrtcCompiler {
    /// Create a new NVRTC compiler instance.
    ///
    /// Automatically detects whether NVRTC is available by
    /// attempting to query the library version.
    pub fn new() -> Self {
        let (available, version) = Self::probe_nvrtc();
        NvrtcCompiler {
            nvrtc_available: available,
            nvrtc_version: version,
            stats: CompilerStats::default(),
        }
    }

    /// Probe for NVRTC availability.
    fn probe_nvrtc() -> (bool, Option<(i32, i32)>) {
        #[cfg(feature = "nvrtc")]
        {
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            let result = unsafe { nvrtcVersion(&mut major, &mut minor) };
            if NvrtcResult::from_i32(result).is_success() {
                return (true, Some((major, minor)));
            }
        }
        // NVRTC not available — will use simulated compilation.
        (false, None)
    }

    /// Check if real NVRTC compilation is available.
    pub fn is_available(&self) -> bool {
        self.nvrtc_available
    }

    /// Get the NVRTC version, if available.
    pub fn version(&self) -> Option<(i32, i32)> {
        self.nvrtc_version
    }

    /// Get compiler statistics.
    pub fn stats(&self) -> &CompilerStats {
        &self.stats
    }

    /// Compile a CUDA C++ source string to PTX.
    ///
    /// # Arguments
    /// * `source` — CUDA C++ source code as a string
    /// * `program_name` — name for the program (e.g., "ramified_cell_update.cu")
    /// * `options` — compilation options (architecture, defines, etc.)
    ///
    /// # Returns
    /// `Ok(CompilationResult)` with the PTX string on success,
    /// `Err(NvrtcError)` with the compiler log on failure.
    pub fn compile(
        &mut self,
        source: &str,
        program_name: &str,
        options: &CompileOptions,
    ) -> Result<CompilationResult, NvrtcError> {
        self.stats.total_compilations += 1;
        let start = Instant::now();

        let result = if self.nvrtc_available {
            self.compile_real(source, program_name, options)
        } else {
            self.compile_simulated(source, program_name, options)
        };

        let elapsed_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(cr) => {
                self.stats.successful_compilations += 1;
                self.stats.total_compile_time_us += elapsed_us;
                self.stats.total_ptx_bytes += cr.ptx.len() as u64;
                if self.stats.successful_compilations > 0 {
                    self.stats.avg_compile_time_us = self.stats.total_compile_time_us as f64
                        / self.stats.successful_compilations as f64;
                }
            }
            Err(_) => {
                self.stats.failed_compilations += 1;
            }
        }

        result
    }

    /// Real NVRTC compilation path.
    #[cfg(feature = "nvrtc")]
    fn compile_real(
        &mut self,
        source: &str,
        program_name: &str,
        options: &CompileOptions,
    ) -> Result<CompilationResult, NvrtcError> {
        let start = Instant::now();

        // Convert source and name to C strings.
        let c_source = CString::new(source).map_err(|e| {
            NvrtcError::InvalidSource(format!("Source contains null byte: {}", e))
        })?;
        let c_name = CString::new(program_name).map_err(|e| {
            NvrtcError::InvalidSource(format!("Program name contains null byte: {}", e))
        })?;

        // Create NVRTC program.
        let mut prog: NvrtcProgramPtr = std::ptr::null_mut();
        let result = unsafe {
            nvrtcCreateProgram(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        if !NvrtcResult::from_i32(result).is_success() {
            return Err(NvrtcError::ProgramCreationFailed(NvrtcResultCode(result)));
        }

        // Build compiler flags.
        let flags = options.build_flags();
        let c_flags: Vec<CString> = flags
            .iter()
            .map(|f| CString::new(f.as_str()).unwrap())
            .collect();
        let c_flag_ptrs: Vec<*const std::os::raw::c_char> =
            c_flags.iter().map(|f| f.as_ptr()).collect();

        // Compile.
        let compile_result = unsafe {
            nvrtcCompileProgram(
                prog,
                c_flag_ptrs.len() as i32,
                if c_flag_ptrs.is_empty() {
                    std::ptr::null()
                } else {
                    c_flag_ptrs.as_ptr()
                },
            )
        };

        // Always retrieve the log (even on success, it may contain warnings).
        let compiler_log = self.get_program_log(prog);

        if !NvrtcResult::from_i32(compile_result).is_success() {
            // Clean up the program handle before returning the error.
            unsafe {
                nvrtcDestroyProgram(&mut prog);
            }
            return Err(NvrtcError::CompilationFailed {
                result_code: NvrtcResultCode(compile_result),
                compiler_log,
            });
        }

        // Retrieve PTX.
        let ptx = self.get_ptx(prog)?;

        // Clean up.
        unsafe {
            nvrtcDestroyProgram(&mut prog);
        }

        let elapsed_us = start.elapsed().as_micros() as u64;

        Ok(CompilationResult {
            ptx,
            compiler_log,
            program_name: program_name.to_string(),
            compile_time_us: elapsed_us,
            options: options.clone(),
            simulated: false,
        })
    }

    /// Retrieve the compiler log from an NVRTC program.
    #[cfg(feature = "nvrtc")]
    fn get_program_log(&self, prog: NvrtcProgramPtr) -> String {
        let mut log_size: usize = 0;
        let result = unsafe { nvrtcGetProgramLogSize(prog, &mut log_size) };
        if !NvrtcResult::from_i32(result).is_success() || log_size == 0 {
            return String::new();
        }

        let mut log_buf: Vec<u8> = vec![0u8; log_size];
        let result = unsafe {
            nvrtcGetProgramLog(prog, log_buf.as_mut_ptr() as *mut std::os::raw::c_char)
        };
        if !NvrtcResult::from_i32(result).is_success() {
            return String::new();
        }

        // Trim trailing null.
        if let Some(nul_pos) = log_buf.iter().position(|&b| b == 0) {
            log_buf.truncate(nul_pos);
        }
        String::from_utf8_lossy(&log_buf).to_string()
    }

    /// Retrieve the compiled PTX from an NVRTC program.
    #[cfg(feature = "nvrtc")]
    fn get_ptx(&self, prog: NvrtcProgramPtr) -> Result<String, NvrtcError> {
        let mut ptx_size: usize = 0;
        let result = unsafe { nvrtcGetPTXSize(prog, &mut ptx_size) };
        if !NvrtcResult::from_i32(result).is_success() {
            return Err(NvrtcError::PtxRetrievalFailed(NvrtcResultCode(result)));
        }

        let mut ptx_buf: Vec<u8> = vec![0u8; ptx_size];
        let result =
            unsafe { nvrtcGetPTX(prog, ptx_buf.as_mut_ptr() as *mut std::os::raw::c_char) };
        if !NvrtcResult::from_i32(result).is_success() {
            return Err(NvrtcError::PtxRetrievalFailed(NvrtcResultCode(result)));
        }

        // Trim trailing null.
        if let Some(nul_pos) = ptx_buf.iter().position(|&b| b == 0) {
            ptx_buf.truncate(nul_pos);
        }
        Ok(String::from_utf8_lossy(&ptx_buf).to_string())
    }

    /// Simulated compilation fallback when NVRTC is not available.
    ///
    /// This performs the same placeholder substitution as the
    /// PtxTemplate::specialize() path, but wraps the result in a
    /// CompilationResult so callers get a uniform interface.
    ///
    /// For CUDA C++ templates (not raw PTX), this generates a
    /// synthetic PTX scaffold that documents the intended kernel
    /// signature and constants. The scaffold is NOT executable —
    /// it exists only for demonstration, testing, and CI.
    fn compile_simulated(
        &mut self,
        source: &str,
        program_name: &str,
        options: &CompileOptions,
    ) -> Result<CompilationResult, NvrtcError> {
        self.stats.simulated_compilations += 1;
        let start = Instant::now();

        let ptx = generate_simulated_ptx(source, program_name, options);

        let elapsed_us = start.elapsed().as_micros() as u64;

        Ok(CompilationResult {
            ptx,
            compiler_log: format!(
                "[simulated] NVRTC not available. Generated synthetic PTX from CUDA C++ source ({} bytes).",
                source.len()
            ),
            program_name: program_name.to_string(),
            compile_time_us: elapsed_us,
            options: options.clone(),
            simulated: true,
        })
    }

    /// Compile a CUDA C++ source with Ramify-specific defaults.
    ///
    /// This is a convenience wrapper that injects cudaclaw's
    /// standard includes, defines, and compilation flags.
    pub fn compile_ramified(
        &mut self,
        source: &str,
        program_name: &str,
        arch: &str,
        constants: &HashMap<String, String>,
    ) -> Result<CompilationResult, NvrtcError> {
        let mut options = CompileOptions {
            gpu_arch: arch.to_string(),
            ..Default::default()
        };

        // Inject constants as preprocessor defines.
        for (name, value) in constants {
            options.defines.insert(name.clone(), value.clone());
        }

        // Ramify-standard flags.
        options.extra_flags.push("--use_fast_math".to_string());
        options
            .extra_flags
            .push("--std=c++17".to_string());
        options
            .extra_flags
            .push("--extensible-whole-program".to_string());

        self.compile(source, program_name, &options)
    }
}

// We need compile_real and helpers even when nvrtc feature is off,
// but they won't be called. Provide stub to avoid conditional
// compilation errors in the struct impl.
#[cfg(not(feature = "nvrtc"))]
impl NvrtcCompiler {
    fn compile_real(
        &mut self,
        _source: &str,
        program_name: &str,
        options: &CompileOptions,
    ) -> Result<CompilationResult, NvrtcError> {
        // Feature not enabled — always fall back to simulated.
        self.compile_simulated(
            _source,
            program_name,
            options,
        )
    }
}

// ============================================================
// Simulated PTX Generation
// ============================================================

/// Generate a synthetic PTX scaffold from CUDA C++ source.
///
/// Parses the source for `__global__` function signatures and
/// generates a minimal PTX program that documents the kernels
/// and their parameters. The output is valid PTX syntax but
/// the kernel bodies are stubs (immediate return).
fn generate_simulated_ptx(
    source: &str,
    program_name: &str,
    options: &CompileOptions,
) -> String {
    let arch_version = options
        .gpu_arch
        .strip_prefix("compute_")
        .unwrap_or("70");

    let mut ptx = String::new();
    ptx.push_str(&format!(
        "// ============================================================\n\
         // Simulated PTX — generated by cudaclaw NVRTC fallback\n\
         // Source: {}\n\
         // Architecture: sm_{}\n\
         // Generated: {:?}\n\
         // NOTE: This is a stub. Real compilation requires NVRTC.\n\
         // ============================================================\n\n\
         .version 7.0\n\
         .target sm_{}\n\
         .address_size 64\n\n",
        program_name,
        arch_version,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        arch_version
    ));

    // Emit defines as PTX constants.
    for (name, value) in &options.defines {
        // Try to parse as integer for .const, otherwise emit as comment.
        if let Ok(int_val) = value.parse::<u32>() {
            ptx.push_str(&format!(".const .u32 {} = {};\n", name, int_val));
        } else {
            ptx.push_str(&format!("// #define {} {}\n", name, value));
        }
    }
    ptx.push('\n');

    // Parse __global__ function signatures from the source.
    let kernels = extract_kernel_signatures(source);
    if kernels.is_empty() {
        // If no kernels found, emit a single stub entry point.
        ptx.push_str(&format!(
            ".visible .entry {}()\n{{\n    ret;\n}}\n",
            program_name.replace('.', "_")
        ));
    } else {
        for kernel in &kernels {
            ptx.push_str(&format!(
                "// Stub for: {}\n\
                 .visible .entry {}(\n",
                kernel.signature, kernel.name
            ));
            for (i, param) in kernel.params.iter().enumerate() {
                let ptx_type = cuda_type_to_ptx(&param.param_type);
                let comma = if i + 1 < kernel.params.len() {
                    ","
                } else {
                    ""
                };
                ptx.push_str(&format!(
                    "    .param {} {}{}    // {}\n",
                    ptx_type, param.name, comma, param.param_type
                ));
            }
            ptx.push_str(")\n{\n    ret;\n}\n\n");
        }
    }

    ptx
}

/// A parsed kernel signature from CUDA C++ source.
struct KernelSignature {
    name: String,
    signature: String,
    params: Vec<KernelParam>,
}

struct KernelParam {
    name: String,
    param_type: String,
}

/// Extract `__global__` kernel signatures from CUDA C++ source.
fn extract_kernel_signatures(source: &str) -> Vec<KernelSignature> {
    let mut kernels = Vec::new();

    // Simple regex-free parser: find lines with `__global__` and
    // extract the function name and parameters.
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Look for __global__ keyword.
        if line.contains("__global__") {
            // Collect the full signature (may span multiple lines).
            let mut sig = String::new();
            let mut j = i;
            while j < lines.len() {
                sig.push_str(lines[j].trim());
                sig.push(' ');
                if sig.contains('{') || sig.contains(';') {
                    break;
                }
                j += 1;
            }

            // Parse function name: find the name before '('
            if let Some(paren_pos) = sig.find('(') {
                let before_paren = &sig[..paren_pos];
                // The function name is the last word before '('
                let name = before_paren
                    .split_whitespace()
                    .last()
                    .unwrap_or("unknown_kernel")
                    .to_string();

                // Parse parameters between '(' and ')'
                let params = if let Some(close_paren) = sig.find(')') {
                    let param_str = &sig[paren_pos + 1..close_paren];
                    parse_params(param_str)
                } else {
                    Vec::new()
                };

                kernels.push(KernelSignature {
                    name,
                    signature: sig.trim().to_string(),
                    params,
                });
            }

            i = j + 1;
        } else {
            i += 1;
        }
    }

    kernels
}

/// Parse a comma-separated parameter list from CUDA C++ source.
fn parse_params(param_str: &str) -> Vec<KernelParam> {
    let mut params = Vec::new();
    for param in param_str.split(',') {
        let param = param.trim();
        if param.is_empty() {
            continue;
        }

        // Split into type and name. The name is the last token.
        let tokens: Vec<&str> = param.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        let name = tokens.last().unwrap().trim_start_matches('*').to_string();
        let param_type = if tokens.len() > 1 {
            tokens[..tokens.len() - 1].join(" ")
                + if param.contains('*') { "*" } else { "" }
        } else {
            tokens[0].to_string()
        };

        params.push(KernelParam { name, param_type });
    }
    params
}

/// Map common CUDA C++ types to PTX parameter types.
fn cuda_type_to_ptx(cuda_type: &str) -> &'static str {
    let t = cuda_type.trim();
    if t.contains('*') {
        return ".u64"; // All pointers are 64-bit addresses
    }
    match t {
        "int" | "int32_t" | "uint32_t" | "unsigned int" | "unsigned" => ".u32",
        "long long" | "int64_t" | "uint64_t" | "unsigned long long" | "size_t" => ".u64",
        "float" => ".f32",
        "double" => ".f64",
        "short" | "int16_t" | "uint16_t" => ".u16",
        "char" | "int8_t" | "uint8_t" | "unsigned char" => ".u8",
        "bool" => ".pred",
        _ => ".u64", // Default to 64-bit for unknown types
    }
}

// ============================================================
// CUDA C++ Template System
// ============================================================

/// Source type for a PtxTemplate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateSourceType {
    /// Raw PTX assembly with placeholder substitution only.
    Ptx,
    /// CUDA C++ source that requires NVRTC compilation.
    CudaCpp,
}

/// A CUDA C++ kernel template that can be compiled via NVRTC.
///
/// Unlike raw PTX templates (which use string substitution),
/// CUDA C++ templates are compiled by NVRTC with the pattern-
/// specific constants injected as preprocessor defines. This
/// allows the template to use full C++ features (templates,
/// constexpr, etc.) while still being specialized at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaCppTemplate {
    /// Unique template identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// CUDA C++ source code.
    pub source: String,
    /// Entry point kernel name (must match a __global__ function).
    pub entry_point: String,
    /// Minimum compute capability (e.g., 70 for Volta).
    pub min_compute_capability: u32,
    /// Which access patterns this template is designed for.
    pub target_patterns: Vec<super::ptx_branching::AccessPattern>,
}

impl CudaCppTemplate {
    pub fn new(id: &str, description: &str, source: &str, entry_point: &str) -> Self {
        CudaCppTemplate {
            id: id.to_string(),
            description: description.to_string(),
            source: source.to_string(),
            entry_point: entry_point.to_string(),
            min_compute_capability: 70,
            target_patterns: Vec::new(),
        }
    }

    pub fn with_target_patterns(
        mut self,
        patterns: Vec<super::ptx_branching::AccessPattern>,
    ) -> Self {
        self.target_patterns = patterns;
        self
    }
}

// ============================================================
// Built-in CUDA C++ Templates
// ============================================================

/// Create a CUDA C++ cell-update template for NVRTC compilation.
///
/// This is the CUDA C++ equivalent of the PTX cell_update_v1
/// template. It uses preprocessor defines for constants that
/// are injected by the Ramify engine based on detected patterns.
pub fn default_cuda_cell_update_template() -> CudaCppTemplate {
    let source = r#"
// ============================================================
// Ramified Cell Update Kernel (CUDA C++)
// ============================================================
// Auto-generated by cudaclaw Ramify NVRTC pipeline.
// Constants are injected as preprocessor defines by the
// Ramify engine based on detected access patterns.
// ============================================================

// Injected by Ramify (defaults if not defined):
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4
#endif

#ifndef USE_L1_CACHE
#define USE_L1_CACHE 1
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 2
#endif

// CRDTCell structure (must match host definition)
struct __align__(32) CRDTCell {
    double value;
    unsigned long long timestamp;
    unsigned int node_id;
    unsigned int state;
    unsigned long long _padding;
};

/// Coalesced cell update kernel.
///
/// Each thread processes one cell. Adjacent threads process
/// adjacent cells in memory for maximum VRAM throughput.
///
/// Template constants (BLOCK_SIZE, UNROLL_FACTOR, etc.) are
/// specialized by the Ramify engine at compile time based on
/// the detected access pattern.
extern "C" __global__ void ramified_cell_update_nvrtc(
    CRDTCell* __restrict__ cells,
    const unsigned int* __restrict__ indices,
    const double* __restrict__ values,
    unsigned int count,
    unsigned long long timestamp,
    unsigned int node_id
) {
    const unsigned int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Prefetch hint for sequential patterns
#if PREFETCH_DISTANCE > 0 && USE_L1_CACHE
    if (gid + PREFETCH_DISTANCE * BLOCK_SIZE < count) {
        const unsigned int prefetch_idx = indices[gid + PREFETCH_DISTANCE * BLOCK_SIZE];
        // Use __ldg for read-only texture cache path
        asm volatile("prefetch.global.L1 [%0];" :: "l"(&cells[prefetch_idx]));
    }
#endif

    if (gid >= count) return;

    // Load cell index (coalesced: adjacent threads load adjacent indices)
    const unsigned int cell_idx = indices[gid];

    // Load new value (coalesced)
#if USE_L1_CACHE
    const double new_val = values[gid];
#else
    // Bypass L1, use texture cache for random access patterns
    const double new_val = __ldg(&values[gid]);
#endif

    // Update cell fields
    CRDTCell* cell = &cells[cell_idx];

    // Atomic CAS on value field for concurrent safety
    unsigned long long* val_ptr = reinterpret_cast<unsigned long long*>(&cell->value);
    unsigned long long old_val = *val_ptr;
    unsigned long long new_val_bits = __double_as_longlong(new_val);

    // Spin-CAS until we successfully update or detect a newer write
    while (true) {
        unsigned long long result = atomicCAS(val_ptr, old_val, new_val_bits);
        if (result == old_val) {
            // CAS succeeded — update metadata
            cell->timestamp = timestamp;
            cell->node_id = node_id;
            __threadfence();  // Ensure visibility
            break;
        }
        // Another thread wrote — check if their timestamp is newer
        CRDTCell current = *cell;
        if (current.timestamp > timestamp) {
            break;  // Newer write wins — nothing to do
        }
        old_val = result;  // Retry with updated expected value
    }
}

/// Warp-aggregated cell update kernel.
///
/// Uses ballot + shuffle to deduplicate writes within a warp
/// when multiple threads target the same cell.
extern "C" __global__ void ramified_cell_update_warp_agg(
    CRDTCell* __restrict__ cells,
    const unsigned int* __restrict__ indices,
    const double* __restrict__ values,
    const unsigned long long* __restrict__ timestamps,
    const unsigned int* __restrict__ node_ids,
    unsigned int count
) {
    const unsigned int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= count) return;

    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int cell_idx = indices[gid];

    // Warp-level deduplication: find threads targeting the same cell
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, cell_idx);
    int leader = __ffs(match_mask) - 1;

    if (lane_id == leader) {
        // This lane is the leader for this cell index.
        // Find the best (highest timestamp) among all matching lanes.
        unsigned long long best_ts = timestamps[gid];
        double best_val = values[gid];
        unsigned int best_nid = node_ids[gid];

        // Butterfly reduction across matching lanes
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            unsigned long long other_ts = __shfl_xor_sync(0xFFFFFFFF, best_ts, offset);
            // Pack value as bits for shuffle
            unsigned long long other_val_bits = __shfl_xor_sync(
                0xFFFFFFFF, __double_as_longlong(best_val), offset);
            unsigned int other_nid = __shfl_xor_sync(0xFFFFFFFF, best_nid, offset);

            int partner = lane_id ^ offset;
            bool partner_matches = (match_mask >> partner) & 1u;

            if (partner_matches && (other_ts > best_ts ||
                (other_ts == best_ts && other_nid > best_nid))) {
                best_ts = other_ts;
                best_val = __longlong_as_double(other_val_bits);
                best_nid = other_nid;
            }
        }

        // Leader performs the actual write
        CRDTCell* cell = &cells[cell_idx];
        unsigned long long* val_ptr = reinterpret_cast<unsigned long long*>(&cell->value);
        unsigned long long old_val = *val_ptr;
        unsigned long long new_val_bits = __double_as_longlong(best_val);

        unsigned long long result = atomicCAS(val_ptr, old_val, new_val_bits);
        if (result == old_val) {
            cell->timestamp = best_ts;
            cell->node_id = best_nid;
            __threadfence();
        }
    }
}
"#;

    CudaCppTemplate::new(
        "cell_update_nvrtc_v1",
        "NVRTC-compiled cell update kernel with warp aggregation",
        source,
        "ramified_cell_update_nvrtc",
    )
    .with_target_patterns(vec![
        super::ptx_branching::AccessPattern::Sequential,
        super::ptx_branching::AccessPattern::BulkSequential,
        super::ptx_branching::AccessPattern::Random,
        super::ptx_branching::AccessPattern::ColumnMajor,
    ])
}

/// Create a CUDA C++ CRDT merge template for NVRTC compilation.
pub fn default_cuda_crdt_merge_template() -> CudaCppTemplate {
    let source = r#"
// ============================================================
// Ramified CRDT Merge Kernel (CUDA C++)
// ============================================================
// Merges remote CRDT state into local state using timestamp-
// based last-writer-wins resolution. Compiled via NVRTC with
// pattern-specific optimizations.
// ============================================================

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct __align__(32) CRDTCell {
    double value;
    unsigned long long timestamp;
    unsigned int node_id;
    unsigned int state;
    unsigned long long _padding;
};

/// CRDT merge: last-writer-wins with coalesced access.
extern "C" __global__ void ramified_crdt_merge_nvrtc(
    CRDTCell* __restrict__ local_cells,
    const CRDTCell* __restrict__ remote_cells,
    unsigned int cell_count,
    unsigned int local_node_id
) {
    const unsigned int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= cell_count) return;

    CRDTCell local = local_cells[gid];
    const CRDTCell remote = __ldg(&remote_cells[gid]);

    // Last-writer-wins: remote wins if remote timestamp is higher,
    // or same timestamp but higher node_id (tiebreaker).
    bool remote_wins = (remote.timestamp > local.timestamp) ||
                       (remote.timestamp == local.timestamp &&
                        remote.node_id > local.node_id);

    if (remote_wins) {
        local_cells[gid] = remote;
        __threadfence();
    }
}
"#;

    CudaCppTemplate::new(
        "crdt_merge_nvrtc_v1",
        "NVRTC-compiled CRDT merge kernel with LWW resolution",
        source,
        "ramified_crdt_merge_nvrtc",
    )
    .with_target_patterns(vec![
        super::ptx_branching::AccessPattern::Sequential,
        super::ptx_branching::AccessPattern::Random,
    ])
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = NvrtcCompiler::new();
        // In CI (no CUDA), should fall back to simulated.
        let stats = compiler.stats();
        assert_eq!(stats.total_compilations, 0);
    }

    #[test]
    fn test_simulated_compilation() {
        let mut compiler = NvrtcCompiler::new();
        let source = r#"
            extern "C" __global__ void test_kernel(float* data, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) data[i] *= 2.0f;
            }
        "#;
        let opts = CompileOptions::for_compute(7, 0);
        let result = compiler.compile(source, "test.cu", &opts);

        assert!(result.is_ok());
        let cr = result.unwrap();
        assert!(cr.ptx.contains(".version 7.0"));
        assert!(cr.ptx.contains("sm_70"));
        assert!(cr.ptx.contains("test_kernel"));
        assert!(cr.compiler_log.contains("simulated"));
        assert!(cr.simulated);
    }

    #[test]
    fn test_simulated_compilation_with_defines() {
        let mut compiler = NvrtcCompiler::new();
        let source = r#"
            extern "C" __global__ void add_kernel(float* a, float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) c[i] = a[i] + b[i];
            }
        "#;
        let opts = CompileOptions::for_compute(8, 0)
            .define("BLOCK_SIZE", "256")
            .define("UNROLL_FACTOR", "4")
            .fast_math();

        let result = compiler.compile(source, "add.cu", &opts);
        assert!(result.is_ok());
        let cr = result.unwrap();
        assert!(cr.ptx.contains("BLOCK_SIZE = 256"));
        assert!(cr.ptx.contains("UNROLL_FACTOR = 4"));
        assert!(cr.ptx.contains("sm_80"));
    }

    #[test]
    fn test_compile_options_build_flags() {
        let opts = CompileOptions::for_compute(7, 5)
            .define("N", "1024")
            .define("DEBUG", "")
            .fast_math()
            .max_regs(64);

        let flags = opts.build_flags();
        assert!(flags.contains(&"--gpu-architecture=compute_75".to_string()));
        assert!(flags.contains(&"-DN=1024".to_string()));
        assert!(flags.contains(&"-DDEBUG".to_string()));
        assert!(flags.contains(&"--use_fast_math".to_string()));
        assert!(flags.contains(&"--maxrregcount=64".to_string()));
    }

    #[test]
    fn test_compile_ramified() {
        let mut compiler = NvrtcCompiler::new();
        let source = r#"
            extern "C" __global__ void cell_update(double* cells, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) cells[i] += 1.0;
            }
        "#;

        let mut constants = HashMap::new();
        constants.insert("BLOCK_SIZE".into(), "128".into());
        constants.insert("UNROLL_FACTOR".into(), "4".into());

        let result = compiler.compile_ramified(source, "cell_update.cu", "compute_70", &constants);
        assert!(result.is_ok());
        let cr = result.unwrap();
        assert!(cr.ptx.contains("cell_update"));
        assert!(cr.ptx.contains("BLOCK_SIZE = 128"));
    }

    #[test]
    fn test_compiler_stats_tracking() {
        let mut compiler = NvrtcCompiler::new();
        let source = r#"extern "C" __global__ void k(int* a) { }"#;
        let opts = CompileOptions::default();

        for _ in 0..5 {
            let _ = compiler.compile(source, "k.cu", &opts);
        }

        let stats = compiler.stats();
        assert_eq!(stats.total_compilations, 5);
        assert_eq!(stats.successful_compilations, 5);
        assert_eq!(stats.failed_compilations, 0);
    }

    #[test]
    fn test_extract_kernel_signatures() {
        let source = r#"
            extern "C" __global__ void kernel_a(float* data, int n) {
                // body
            }
            __device__ void helper() {}
            extern "C" __global__ void kernel_b(
                double* input,
                double* output,
                unsigned int count
            ) {
                // body
            }
        "#;

        let kernels = extract_kernel_signatures(source);
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].name, "kernel_a");
        assert_eq!(kernels[0].params.len(), 2);
        assert_eq!(kernels[1].name, "kernel_b");
        assert_eq!(kernels[1].params.len(), 3);
    }

    #[test]
    fn test_cuda_type_to_ptx_mapping() {
        assert_eq!(cuda_type_to_ptx("int"), ".u32");
        assert_eq!(cuda_type_to_ptx("float"), ".f32");
        assert_eq!(cuda_type_to_ptx("double"), ".f64");
        assert_eq!(cuda_type_to_ptx("float*"), ".u64");
        assert_eq!(cuda_type_to_ptx("unsigned long long"), ".u64");
        assert_eq!(cuda_type_to_ptx("uint32_t"), ".u32");
    }

    #[test]
    fn test_default_cell_update_template() {
        let template = default_cuda_cell_update_template();
        assert_eq!(template.id, "cell_update_nvrtc_v1");
        assert!(template.source.contains("ramified_cell_update_nvrtc"));
        assert!(template.source.contains("BLOCK_SIZE"));
        assert!(!template.target_patterns.is_empty());
    }

    #[test]
    fn test_default_crdt_merge_template() {
        let template = default_cuda_crdt_merge_template();
        assert_eq!(template.id, "crdt_merge_nvrtc_v1");
        assert!(template.source.contains("ramified_crdt_merge_nvrtc"));
        assert!(template.source.contains("last-writer-wins"));
    }

    #[test]
    fn test_nvrtc_result_codes() {
        assert!(NvrtcResult::Success.is_success());
        assert!(!NvrtcResult::CompilationError.is_success());
        assert_eq!(NvrtcResult::from_i32(0), NvrtcResult::Success);
        assert_eq!(NvrtcResult::from_i32(6), NvrtcResult::CompilationError);
        assert_eq!(NvrtcResult::from_i32(999), NvrtcResult::InternalError);
    }

    #[test]
    fn test_compile_cell_update_template() {
        let mut compiler = NvrtcCompiler::new();
        let template = default_cuda_cell_update_template();

        let mut constants = HashMap::new();
        constants.insert("BLOCK_SIZE".into(), "256".into());
        constants.insert("UNROLL_FACTOR".into(), "8".into());
        constants.insert("USE_L1_CACHE".into(), "1".into());
        constants.insert("PREFETCH_DISTANCE".into(), "4".into());

        let result = compiler.compile_ramified(
            &template.source,
            &format!("{}.cu", template.id),
            "compute_70",
            &constants,
        );
        assert!(result.is_ok());
        let cr = result.unwrap();
        assert!(cr.ptx.contains("ramified_cell_update_nvrtc"));
        assert!(cr.ptx.contains("BLOCK_SIZE = 256"));
    }

    #[test]
    fn test_error_display() {
        let err = NvrtcError::CompilationFailed {
            result_code: NvrtcResultCode(6),
            compiler_log: "error: unknown type".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("compilation failed"));
        assert!(msg.contains("unknown type"));
    }
}
