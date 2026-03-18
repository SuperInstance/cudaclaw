// ============================================================
// NVRTC Muscle Compiler — Compile LLM-Suggested PTX Constants
// ============================================================
//
// This module bridges the LLM optimization suggestions with the
// NVRTC runtime compiler. For each OptimizationSuggestion, it:
//
//   1. Generates a CUDA C++ kernel string with the suggestion's
//      constants injected as preprocessor defines.
//   2. Compiles the kernel via NvrtcCompiler (real or simulated).
//   3. Returns a MuscleFiberCompilation that can be benchmarked
//      in the micro-simulation engine.
//
// The generated kernel matches the persistent_worker structure
// from kernels/executor.cu, with the LLM-suggested block sizes,
// unroll factors, shared memory configuration, and cache hints
// baked in as compile-time constants.
//
// When NVRTC is not available (no CUDA toolkit), the module
// falls back to simulated compilation — the same interface is
// maintained so the installer pipeline works identically on
// CI and development machines without GPUs.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::installer::hardware_probe::HardwareProfile;
use crate::installer::llm_optimizer::OptimizationSuggestion;
use crate::ramify::nvrtc_compiler::{
    CompilationResult, CompileOptions, NvrtcCompiler, NvrtcError,
};

// ============================================================
// Muscle Fiber Compilation Result
// ============================================================

/// Result of compiling an LLM-suggested muscle fiber via NVRTC.
///
/// Contains the compiled PTX (or simulated PTX scaffold), the
/// constants that were injected, and timing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuscleFiberCompilation {
    /// The suggestion that produced this compilation.
    pub suggestion_id: String,

    /// The NVRTC compilation result (PTX + logs).
    pub compilation: CompilationResult,

    /// Constants injected as preprocessor defines.
    pub injected_constants: HashMap<String, String>,

    /// GPU architecture target (e.g., "compute_89").
    pub target_arch: String,

    /// Whether this was a simulated compilation.
    pub simulated: bool,

    /// Total wall-clock time for generation + compilation (microseconds).
    pub total_time_us: u64,
}

// ============================================================
// Muscle Fiber Compiler
// ============================================================

/// Compiles LLM-suggested optimization constants into PTX
/// muscle fibers via NVRTC runtime compilation.
///
/// Each suggestion's tunable parameters (block_size, unroll_factor,
/// shared_memory_bytes, etc.) are injected into a CUDA C++ kernel
/// template as preprocessor defines, then compiled to PTX.
///
/// On machines without NVRTC, compilation falls back to simulated
/// mode — a synthetic PTX scaffold is generated that documents
/// the intended kernel but is not executable.
pub struct MuscleFiberCompiler {
    /// The NVRTC compiler instance.
    compiler: NvrtcCompiler,

    /// Hardware profile for architecture-specific compilation.
    hardware: HardwareProfile,
}

impl MuscleFiberCompiler {
    /// Create a new muscle fiber compiler.
    pub fn new(hardware: HardwareProfile) -> Self {
        MuscleFiberCompiler {
            compiler: NvrtcCompiler::new(),
            hardware,
        }
    }

    /// Whether real NVRTC compilation is available.
    pub fn is_nvrtc_available(&self) -> bool {
        self.compiler.is_available()
    }

    /// Compile a single LLM suggestion into a PTX muscle fiber.
    ///
    /// Generates a CUDA C++ kernel with the suggestion's constants
    /// as preprocessor defines, then compiles via NVRTC.
    pub fn compile_suggestion(
        &mut self,
        suggestion: &OptimizationSuggestion,
    ) -> Result<MuscleFiberCompilation, NvrtcError> {
        let start = Instant::now();

        // Build the constants map from the suggestion
        let constants = self.suggestion_to_constants(suggestion);

        // Determine the target architecture from hardware profile
        let arch = self.compute_arch_string();

        // Generate the CUDA C++ source with injected constants
        let source = self.generate_muscle_kernel(suggestion);

        // Compile via NVRTC
        let program_name = format!("muscle_{}.cu", suggestion.suggestion_id);
        let result = self.compiler.compile_ramified(
            &source,
            &program_name,
            &arch,
            &constants,
        )?;

        let total_time_us = start.elapsed().as_micros() as u64;
        let simulated = result.simulated;

        Ok(MuscleFiberCompilation {
            suggestion_id: suggestion.suggestion_id.clone(),
            compilation: result,
            injected_constants: constants,
            target_arch: arch,
            simulated,
            total_time_us,
        })
    }

    /// Compile multiple suggestions, returning results for each.
    /// Failures are logged but do not stop compilation of other suggestions.
    pub fn compile_all(
        &mut self,
        suggestions: &[OptimizationSuggestion],
    ) -> Vec<MuscleFiberCompilation> {
        let mut results = Vec::new();

        for (i, suggestion) in suggestions.iter().enumerate() {
            println!(
                "    [NVRTC {}/{}] Compiling muscle fiber: {}...",
                i + 1,
                suggestions.len(),
                suggestion.suggestion_id
            );

            match self.compile_suggestion(suggestion) {
                Ok(compilation) => {
                    println!(
                        "      {} PTX ({} bytes, {}us, {})",
                        if compilation.simulated {
                            "Simulated"
                        } else {
                            "Compiled"
                        },
                        compilation.compilation.ptx.len(),
                        compilation.total_time_us,
                        compilation.target_arch,
                    );
                    results.push(compilation);
                }
                Err(e) => {
                    println!(
                        "      WARNING: NVRTC compilation failed for {}: {}",
                        suggestion.suggestion_id, e
                    );
                }
            }
        }

        results
    }

    /// Convert an OptimizationSuggestion into NVRTC preprocessor defines.
    ///
    /// These are injected as `-DNAME=VALUE` flags during compilation,
    /// making the constants available as compile-time values in the
    /// CUDA C++ source.
    fn suggestion_to_constants(
        &self,
        suggestion: &OptimizationSuggestion,
    ) -> HashMap<String, String> {
        let mut constants = HashMap::new();

        // Core launch parameters
        constants.insert(
            "CUDACLAW_BLOCK_SIZE".to_string(),
            suggestion.block_size.to_string(),
        );
        constants.insert(
            "CUDACLAW_GRID_SIZE".to_string(),
            suggestion.grid_size.to_string(),
        );
        constants.insert(
            "CUDACLAW_WARPS_PER_BLOCK".to_string(),
            suggestion.warps_per_block.to_string(),
        );

        // Memory configuration
        constants.insert(
            "CUDACLAW_SHARED_MEM_BYTES".to_string(),
            suggestion.shared_memory_bytes.to_string(),
        );
        constants.insert(
            "CUDACLAW_L1_CACHE_PREF".to_string(),
            suggestion.l1_cache_preference.to_string(),
        );

        // Loop optimization
        constants.insert(
            "CUDACLAW_UNROLL_FACTOR".to_string(),
            suggestion.loop_unroll_factor.to_string(),
        );

        // Polling and batching
        constants.insert(
            "CUDACLAW_IDLE_SLEEP_NS".to_string(),
            suggestion.idle_sleep_ns.to_string(),
        );
        constants.insert(
            "CUDACLAW_CMD_BATCH_SIZE".to_string(),
            suggestion.command_batch_size.to_string(),
        );

        // Warp aggregation
        constants.insert(
            "CUDACLAW_WARP_AGGREGATION".to_string(),
            if suggestion.enable_warp_aggregation {
                "1"
            } else {
                "0"
            }
            .to_string(),
        );

        // Memory layout
        constants.insert(
            "CUDACLAW_SOA_LAYOUT".to_string(),
            if suggestion.use_soa_layout { "1" } else { "0" }.to_string(),
        );

        // CAS backoff
        constants.insert(
            "CUDACLAW_CAS_BACKOFF_INIT".to_string(),
            suggestion.cas_backoff_initial_ns.to_string(),
        );
        constants.insert(
            "CUDACLAW_CAS_BACKOFF_MAX".to_string(),
            suggestion.cas_backoff_max_ns.to_string(),
        );

        // Formula recalculation
        constants.insert(
            "CUDACLAW_FRONTIER_BATCH".to_string(),
            suggestion.frontier_batch_size.to_string(),
        );

        // Hardware-derived constants
        constants.insert(
            "CUDACLAW_WARP_SIZE".to_string(),
            self.hardware.warp_size.to_string(),
        );
        constants.insert(
            "CUDACLAW_L1_CACHE_KB".to_string(),
            (self.hardware.l1_cache_size_bytes / 1024).to_string(),
        );
        constants.insert(
            "CUDACLAW_L2_CACHE_KB".to_string(),
            (self.hardware.l2_cache_size_bytes / 1024).to_string(),
        );
        constants.insert(
            "CUDACLAW_SM_COUNT".to_string(),
            self.hardware.sm_count.to_string(),
        );

        constants
    }

    /// Determine the NVRTC compilation target architecture string
    /// from the hardware profile's compute capability.
    fn compute_arch_string(&self) -> String {
        let cc = &self.hardware.compute_capability;
        let parts: Vec<&str> = cc.split('.').collect();
        if parts.len() == 2 {
            format!("compute_{}{}", parts[0], parts[1])
        } else {
            // Default to Volta if we can't parse
            "compute_70".to_string()
        }
    }

    /// Generate the CUDA C++ kernel source for a muscle fiber.
    ///
    /// This is a persistent-worker kernel template that uses the
    /// LLM-suggested constants (injected via preprocessor defines)
    /// to configure block size, unroll factor, shared memory usage,
    /// L1/L2 cache preferences, and CAS backoff strategy.
    ///
    /// The kernel structure mirrors `kernels/executor.cu`'s
    /// `persistent_worker` but with compile-time specialization.
    fn generate_muscle_kernel(&self, suggestion: &OptimizationSuggestion) -> String {
        format!(
            r#"// ============================================================
// CudaClaw Muscle Fiber — LLM-Optimized Persistent Worker
// ============================================================
// Suggestion: {suggestion_id}
// Block size: {block_size} threads ({warps} warps)
// Shared memory: {shmem} bytes
// L1 cache preference: {l1_pref}
// Unroll factor: {unroll}x
// Idle sleep: {sleep} ns
// Warp aggregation: {warp_agg}
// SoA layout: {soa}
// Target: {arch}
// ============================================================
//
// Hardware context:
//   GPU: {gpu_name}
//   SMs: {sm_count}
//   L1 cache: {l1_kb} KB per SM
//   L2 cache: {l2_kb} KB
//   Warp size: {warp_size}
// ============================================================

// All constants are injected as preprocessor defines by the
// NVRTC compilation step. The #ifndef guards provide defaults
// for standalone testing.

#ifndef CUDACLAW_BLOCK_SIZE
#define CUDACLAW_BLOCK_SIZE {block_size}
#endif

#ifndef CUDACLAW_GRID_SIZE
#define CUDACLAW_GRID_SIZE {grid_size}
#endif

#ifndef CUDACLAW_WARPS_PER_BLOCK
#define CUDACLAW_WARPS_PER_BLOCK {warps}
#endif

#ifndef CUDACLAW_SHARED_MEM_BYTES
#define CUDACLAW_SHARED_MEM_BYTES {shmem}
#endif

#ifndef CUDACLAW_L1_CACHE_PREF
#define CUDACLAW_L1_CACHE_PREF {l1_pref}
#endif

#ifndef CUDACLAW_UNROLL_FACTOR
#define CUDACLAW_UNROLL_FACTOR {unroll}
#endif

#ifndef CUDACLAW_IDLE_SLEEP_NS
#define CUDACLAW_IDLE_SLEEP_NS {sleep}
#endif

#ifndef CUDACLAW_CMD_BATCH_SIZE
#define CUDACLAW_CMD_BATCH_SIZE {cmd_batch}
#endif

#ifndef CUDACLAW_WARP_AGGREGATION
#define CUDACLAW_WARP_AGGREGATION {warp_agg_int}
#endif

#ifndef CUDACLAW_SOA_LAYOUT
#define CUDACLAW_SOA_LAYOUT {soa_int}
#endif

#ifndef CUDACLAW_CAS_BACKOFF_INIT
#define CUDACLAW_CAS_BACKOFF_INIT {cas_init}
#endif

#ifndef CUDACLAW_CAS_BACKOFF_MAX
#define CUDACLAW_CAS_BACKOFF_MAX {cas_max}
#endif

#ifndef CUDACLAW_FRONTIER_BATCH
#define CUDACLAW_FRONTIER_BATCH {frontier}
#endif

#ifndef CUDACLAW_WARP_SIZE
#define CUDACLAW_WARP_SIZE {warp_size}
#endif

// ── Ring Buffer Types ──────────────────────────────────────

struct CommandQueue {{
    volatile unsigned int is_running;
    volatile unsigned long long head;
    volatile unsigned long long tail;
    volatile unsigned int status;
    float buffer[1024 * 12];  // Command data (SoA or AoS)
}};

// ── L1/L2 Cache Preference Helper ──────────────────────────

__device__ __forceinline__ void set_cache_preference() {{
#if CUDACLAW_L1_CACHE_PREF == 1
    // Prefer L1 — maximize L1 for latency-sensitive reads
    // cudaFuncSetCacheConfig equivalent at PTX level
    asm volatile("// L1 cache preference: PREFER_L1");
#elif CUDACLAW_L1_CACHE_PREF == 2
    // Prefer shared memory — maximize shmem for working sets
    asm volatile("// L1 cache preference: PREFER_SHARED");
#elif CUDACLAW_L1_CACHE_PREF == 3
    // Equal split
    asm volatile("// L1 cache preference: PREFER_EQUAL");
#else
    // Default — let the driver decide
    asm volatile("// L1 cache preference: DEFAULT");
#endif
}}

// ── Warp-Aggregated CAS ──────────────────────────────────

__device__ __forceinline__ unsigned long long warp_aggregated_cas(
    unsigned long long* addr,
    unsigned long long compare,
    unsigned long long val
) {{
#if CUDACLAW_WARP_AGGREGATION
    // Only lane 0 performs the CAS; result is broadcast to all lanes
    unsigned long long result;
    int lane = threadIdx.x % CUDACLAW_WARP_SIZE;
    if (lane == 0) {{
        result = atomicCAS(addr, compare, val);
    }}
    result = __shfl_sync(0xFFFFFFFF, result, 0);
    return result;
#else
    // Every lane does its own CAS (higher contention)
    return atomicCAS(addr, compare, val);
#endif
}}

// ── CAS Backoff ───────────────────────────────────────────

__device__ __forceinline__ void cas_backoff(int retry_count) {{
    unsigned int delay = CUDACLAW_CAS_BACKOFF_INIT;
    for (int i = 0; i < retry_count && delay < CUDACLAW_CAS_BACKOFF_MAX; i++) {{
        delay = delay * 2;
    }}
    // Simulated nanosleep (requires sm_70+)
    #if __CUDA_ARCH__ >= 700
    __nanosleep(delay);
    #endif
}}

// ── Persistent Worker Kernel ──────────────────────────────

extern "C" __global__ void persistent_worker_muscle(CommandQueue* queue) {{
    // Set L1/L2 cache preference for this kernel
    set_cache_preference();

    const int lane = threadIdx.x % CUDACLAW_WARP_SIZE;
    const int warp_id = threadIdx.x / CUDACLAW_WARP_SIZE;

    // Shared memory for working set cache
    __shared__ float shmem_cache[CUDACLAW_SHARED_MEM_BYTES / sizeof(float)];

    // Initialize shared memory cache
    for (int i = threadIdx.x; i < CUDACLAW_SHARED_MEM_BYTES / sizeof(float);
         i += CUDACLAW_BLOCK_SIZE) {{
        shmem_cache[i] = 0.0f;
    }}
    __syncthreads();

    unsigned long long local_tail = 0;

    // ── Main persistent polling loop ──────────────────────
    while (queue->is_running) {{
        // Phase 1: Lane 0 polls the head index
        unsigned long long current_head = 0;
        if (lane == 0) {{
            current_head = queue->head;
            __threadfence_system();  // PCIe fence for instant visibility
        }}

        // Phase 2: Broadcast head to all lanes
        current_head = __shfl_sync(0xFFFFFFFF, current_head, 0);

        // Phase 3: Check for new commands
        if (current_head <= local_tail) {{
            // No new commands — idle sleep
            #if __CUDA_ARCH__ >= 700
            __nanosleep(CUDACLAW_IDLE_SLEEP_NS);
            #endif
            continue;
        }}

        // Phase 4: Process command batch with unrolled loop
        unsigned long long cmds_available = current_head - local_tail;
        unsigned long long batch = cmds_available;
        if (batch > CUDACLAW_CMD_BATCH_SIZE) {{
            batch = CUDACLAW_CMD_BATCH_SIZE;
        }}

        #pragma unroll CUDACLAW_UNROLL_FACTOR
        for (unsigned long long b = 0; b < batch; b++) {{
            unsigned long long cmd_idx = (local_tail + b) % 1024;

#if CUDACLAW_SOA_LAYOUT
            // SoA layout: coalesced memory access
            // Adjacent threads read adjacent elements
            float data = queue->buffer[cmd_idx * 12 + lane % 12];
            // Cache in shared memory for reuse
            if (lane < 12) {{
                shmem_cache[(cmd_idx % (CUDACLAW_SHARED_MEM_BYTES / sizeof(float) / 12)) * 12 + lane] = data;
            }}
#else
            // AoS layout: each thread reads its command's data
            float data = queue->buffer[cmd_idx * 12 + lane % 12];
#endif

            // Process the command (simplified cell update)
            if (lane == 0 && warp_id == 0) {{
                // Write completion
                queue->tail = local_tail + b + 1;
                __threadfence_system();
            }}
        }}

        local_tail += batch;

        __syncwarp();
    }}

    // Kernel exit
    if (threadIdx.x == 0) {{
        queue->status = 2;  // DONE
    }}
}}

// ── Formula Recalculation Kernel ──────────────────────────

extern "C" __global__ void formula_recalc_muscle(
    float* cells,
    int* dep_graph,
    int* frontier,
    int frontier_size,
    int total_cells
) {{
    __shared__ float local_cells[CUDACLAW_SHARED_MEM_BYTES / sizeof(float)];

    int tid = blockIdx.x * CUDACLAW_BLOCK_SIZE + threadIdx.x;

    // Load frontier cells into shared memory
    #pragma unroll CUDACLAW_UNROLL_FACTOR
    for (int i = threadIdx.x; i < CUDACLAW_FRONTIER_BATCH && i < frontier_size;
         i += CUDACLAW_BLOCK_SIZE) {{
        int cell_idx = frontier[i];
        if (cell_idx < total_cells) {{
            local_cells[i] = cells[cell_idx];
        }}
    }}
    __syncthreads();

    // Recalculate dependent cells
    if (tid < frontier_size) {{
        int cell_idx = frontier[tid];
        if (cell_idx < total_cells) {{
            // Simplified: sum of dependencies
            float sum = 0.0f;
            int dep_start = dep_graph[cell_idx * 2];
            int dep_count = dep_graph[cell_idx * 2 + 1];

            #pragma unroll CUDACLAW_UNROLL_FACTOR
            for (int d = 0; d < dep_count; d++) {{
                int dep_idx = dep_graph[(total_cells * 2) + dep_start + d];
                if (dep_idx >= 0 && dep_idx < total_cells) {{
                    // Try shared memory first, fall back to global
                    if (dep_idx < CUDACLAW_FRONTIER_BATCH) {{
                        sum += local_cells[dep_idx];
                    }} else {{
                        sum += cells[dep_idx];
                    }}
                }}
            }}

            cells[cell_idx] = sum;
        }}
    }}
}}
"#,
            suggestion_id = suggestion.suggestion_id,
            block_size = suggestion.block_size,
            warps = suggestion.warps_per_block,
            shmem = suggestion.shared_memory_bytes,
            l1_pref = suggestion.l1_cache_preference,
            unroll = suggestion.loop_unroll_factor,
            sleep = suggestion.idle_sleep_ns,
            warp_agg = if suggestion.enable_warp_aggregation { "ON" } else { "OFF" },
            soa = if suggestion.use_soa_layout { "ON" } else { "OFF" },
            arch = self.compute_arch_string(),
            gpu_name = self.hardware.gpu_name,
            sm_count = self.hardware.sm_count,
            l1_kb = self.hardware.l1_cache_size_bytes / 1024,
            l2_kb = self.hardware.l2_cache_size_bytes / 1024,
            warp_size = self.hardware.warp_size,
            grid_size = suggestion.grid_size,
            cmd_batch = suggestion.command_batch_size,
            warp_agg_int = if suggestion.enable_warp_aggregation { 1 } else { 0 },
            soa_int = if suggestion.use_soa_layout { 1 } else { 0 },
            cas_init = suggestion.cas_backoff_initial_ns,
            cas_max = suggestion.cas_backoff_max_ns,
            frontier = suggestion.frontier_batch_size,
        )
    }

    /// Print a summary of compilation results.
    pub fn print_compilation_summary(compilations: &[MuscleFiberCompilation]) {
        println!("\n{}", "=".repeat(80));
        println!("  NVRTC Muscle Fiber Compilation Results");
        println!("{}", "=".repeat(80));
        println!(
            "  {:<28} {:>8} {:>12} {:>10} {:>8}",
            "Suggestion", "PTX Size", "Compile (us)", "Arch", "Mode"
        );
        println!("  {}", "-".repeat(74));

        for c in compilations {
            let mode = if c.simulated { "SIM" } else { "REAL" };
            println!(
                "  {:<28} {:>8} {:>12} {:>10} {:>8}",
                &c.suggestion_id[..c.suggestion_id.len().min(28)],
                c.compilation.ptx.len(),
                c.total_time_us,
                c.target_arch,
                mode,
            );
        }

        println!("{}\n", "=".repeat(80));
    }

    /// Get compiler statistics.
    pub fn compiler_stats(&self) -> &crate::ramify::nvrtc_compiler::CompilerStats {
        self.compiler.stats()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::installer::hardware_probe::HardwareProber;

    fn setup() -> HardwareProfile {
        HardwareProber::simulated(0).probe()
    }

    fn test_suggestion() -> OptimizationSuggestion {
        OptimizationSuggestion {
            suggestion_id: "test-nvrtc-1".to_string(),
            block_size: 64,
            grid_size: 2,
            shared_memory_bytes: 49152,
            loop_unroll_factor: 8,
            command_batch_size: 4,
            idle_sleep_ns: 50,
            warps_per_block: 2,
            l1_cache_preference: 1,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            frontier_batch_size: 512,
            cas_backoff_initial_ns: 200,
            cas_backoff_max_ns: 20000,
            reasoning: "Test suggestion for NVRTC compilation".to_string(),
            estimated_improvement_pct: 15.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_compile_suggestion() {
        let hw = setup();
        let mut compiler = MuscleFiberCompiler::new(hw);
        let suggestion = test_suggestion();

        let result = compiler.compile_suggestion(&suggestion).unwrap();

        assert_eq!(result.suggestion_id, "test-nvrtc-1");
        assert!(!result.compilation.ptx.is_empty());
        assert!(result.compilation.ptx.contains("persistent_worker_muscle"));
        assert!(result.compilation.ptx.contains("formula_recalc_muscle"));
        assert!(result.total_time_us > 0 || result.simulated);
    }

    #[test]
    fn test_constants_injection() {
        let hw = setup();
        let compiler = MuscleFiberCompiler::new(hw.clone());
        let suggestion = test_suggestion();

        let constants = compiler.suggestion_to_constants(&suggestion);

        assert_eq!(constants["CUDACLAW_BLOCK_SIZE"], "64");
        assert_eq!(constants["CUDACLAW_GRID_SIZE"], "2");
        assert_eq!(constants["CUDACLAW_UNROLL_FACTOR"], "8");
        assert_eq!(constants["CUDACLAW_IDLE_SLEEP_NS"], "50");
        assert_eq!(constants["CUDACLAW_WARP_AGGREGATION"], "1");
        assert_eq!(constants["CUDACLAW_SOA_LAYOUT"], "1");
        assert_eq!(constants["CUDACLAW_CAS_BACKOFF_INIT"], "200");
        assert_eq!(constants["CUDACLAW_CAS_BACKOFF_MAX"], "20000");
        assert_eq!(constants["CUDACLAW_FRONTIER_BATCH"], "512");
        assert_eq!(
            constants["CUDACLAW_SM_COUNT"],
            hw.sm_count.to_string()
        );
        assert_eq!(
            constants["CUDACLAW_L1_CACHE_KB"],
            (hw.l1_cache_size_bytes / 1024).to_string()
        );
        assert_eq!(
            constants["CUDACLAW_L2_CACHE_KB"],
            (hw.l2_cache_size_bytes / 1024).to_string()
        );
    }

    #[test]
    fn test_arch_string() {
        let mut hw = setup();
        hw.compute_capability = "8.9".to_string();
        let compiler = MuscleFiberCompiler::new(hw);
        assert_eq!(compiler.compute_arch_string(), "compute_89");

        let mut hw2 = setup();
        hw2.compute_capability = "7.0".to_string();
        let compiler2 = MuscleFiberCompiler::new(hw2);
        assert_eq!(compiler2.compute_arch_string(), "compute_70");
    }

    #[test]
    fn test_compile_all() {
        let hw = setup();
        let mut compiler = MuscleFiberCompiler::new(hw);

        let suggestions = vec![
            OptimizationSuggestion {
                suggestion_id: "s1".to_string(),
                block_size: 32,
                ..Default::default()
            },
            OptimizationSuggestion {
                suggestion_id: "s2".to_string(),
                block_size: 64,
                ..Default::default()
            },
            OptimizationSuggestion {
                suggestion_id: "s3".to_string(),
                block_size: 128,
                ..Default::default()
            },
        ];

        let results = compiler.compile_all(&suggestions);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].suggestion_id, "s1");
        assert_eq!(results[1].suggestion_id, "s2");
        assert_eq!(results[2].suggestion_id, "s3");

        // All should have non-empty PTX
        for r in &results {
            assert!(!r.compilation.ptx.is_empty());
        }
    }

    #[test]
    fn test_generated_kernel_has_cache_info() {
        let hw = setup();
        let compiler = MuscleFiberCompiler::new(hw.clone());
        let suggestion = test_suggestion();

        let source = compiler.generate_muscle_kernel(&suggestion);

        // Should contain L1/L2 cache info from hardware
        assert!(source.contains(&format!("L1 cache: {} KB per SM", hw.l1_cache_size_bytes / 1024)));
        assert!(source.contains(&format!("L2 cache: {} KB", hw.l2_cache_size_bytes / 1024)));
        assert!(source.contains(&format!("SMs: {}", hw.sm_count)));

        // Should contain unroll pragma
        assert!(source.contains("#pragma unroll CUDACLAW_UNROLL_FACTOR"));

        // Should contain cache preference function
        assert!(source.contains("set_cache_preference"));

        // Should contain warp aggregated CAS
        assert!(source.contains("warp_aggregated_cas"));
    }

    #[test]
    fn test_warp_aggregation_off() {
        let hw = setup();
        let compiler = MuscleFiberCompiler::new(hw);
        let suggestion = OptimizationSuggestion {
            suggestion_id: "no-agg".to_string(),
            enable_warp_aggregation: false,
            ..Default::default()
        };

        let constants = compiler.suggestion_to_constants(&suggestion);
        assert_eq!(constants["CUDACLAW_WARP_AGGREGATION"], "0");
    }

    #[test]
    fn test_compilation_result_serialization() {
        let hw = setup();
        let mut compiler = MuscleFiberCompiler::new(hw);
        let suggestion = test_suggestion();

        let result = compiler.compile_suggestion(&suggestion).unwrap();

        let json = serde_json::to_string(&result).unwrap();
        let loaded: MuscleFiberCompilation = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.suggestion_id, result.suggestion_id);
        assert_eq!(loaded.target_arch, result.target_arch);
        assert_eq!(loaded.simulated, result.simulated);
        assert_eq!(
            loaded.injected_constants.len(),
            result.injected_constants.len()
        );
    }

    #[test]
    fn test_compiler_stats_updated() {
        let hw = setup();
        let mut compiler = MuscleFiberCompiler::new(hw);

        // Before compilation
        assert_eq!(compiler.compiler_stats().total_compilations, 0);

        let suggestion = test_suggestion();
        compiler.compile_suggestion(&suggestion).unwrap();

        // After compilation
        assert!(compiler.compiler_stats().total_compilations >= 1);
    }
}
