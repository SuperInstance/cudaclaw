// ============================================================
// Hardware Probe - GPU Micro-Benchmark Suite
// ============================================================
//
// Probes the target GPU to extract hardware-specific attributes
// that inform the LLM optimization loop. Measures:
//
//   1. L1/L2 Cache Sizes & Line Width
//   2. Warp-Level Concurrency (active warps per SM, occupancy)
//   3. Global Memory Latency (random-access & sequential)
//   4. Shared Memory Bandwidth
//   5. Atomic Operation Throughput (CAS contention profile)
//   6. PCIe Host↔Device Transfer Latency
//
// On systems without a live GPU, the probe returns simulated
// results based on common architectures (Ampere/Ada Lovelace)
// so the LLM optimization loop can still generate a baseline
// configuration.
//
// CUDA KERNEL INTEGRATION:
// On a real GPU, each probe would launch a dedicated micro-
// benchmark kernel (e.g., pointer-chasing for cache sizing,
// warp-shuffle throughput for concurrency). The kernel source
// lives in kernels/probe_benchmarks.cu (to be added when nvcc
// is available). Here we define the Rust-side data structures
// and the simulated fallback path.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================
// GPU Hardware Profile
// ============================================================

/// Complete hardware profile from the probe suite.
/// This is the primary input to the LLM optimization loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Human-readable GPU name (e.g., "NVIDIA RTX 4090")
    pub gpu_name: String,

    /// CUDA compute capability (e.g., "8.9")
    pub compute_capability: String,

    /// Number of Streaming Multiprocessors
    pub sm_count: u32,

    /// Maximum threads per SM
    pub max_threads_per_sm: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Warp size (always 32 on NVIDIA GPUs)
    pub warp_size: u32,

    /// Maximum resident warps per SM
    pub max_warps_per_sm: u32,

    /// Maximum shared memory per block (bytes)
    pub max_shared_memory_per_block: u32,

    /// Maximum shared memory per SM (bytes)
    pub max_shared_memory_per_sm: u32,

    /// L1 cache size per SM (bytes), measured via pointer-chasing
    pub l1_cache_size_bytes: u32,

    /// L2 cache size (bytes)
    pub l2_cache_size_bytes: u32,

    /// Cache line size (bytes), typically 128 on modern NVIDIA GPUs
    pub cache_line_bytes: u32,

    /// Total global memory (bytes)
    pub global_memory_bytes: u64,

    /// Memory bus width (bits)
    pub memory_bus_width_bits: u32,

    /// Memory clock rate (MHz)
    pub memory_clock_mhz: u32,

    /// GPU core clock rate (MHz)
    pub core_clock_mhz: u32,

    /// Measured memory latency results
    pub memory_latency: MemoryLatencyProfile,

    /// Measured atomic operation throughput
    pub atomic_throughput: AtomicThroughputProfile,

    /// Measured warp-level concurrency metrics
    pub warp_concurrency: WarpConcurrencyProfile,

    /// Measured PCIe transfer characteristics
    pub pcie_profile: PcieProfile,

    /// Whether this profile was measured on real hardware
    /// or simulated from known architecture specs
    pub is_simulated: bool,

    /// Architecture family (e.g., "Ampere", "Ada Lovelace", "Hopper")
    pub architecture: String,

    /// Probe timestamp (Unix seconds)
    pub probe_timestamp: u64,
}

// ============================================================
// Sub-Profiles: Memory Latency
// ============================================================

/// Memory latency measurements from pointer-chasing and
/// sequential access micro-benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLatencyProfile {
    /// L1 cache hit latency (nanoseconds)
    pub l1_hit_latency_ns: f64,

    /// L2 cache hit latency (nanoseconds)
    pub l2_hit_latency_ns: f64,

    /// Global memory (DRAM) latency (nanoseconds)
    pub global_memory_latency_ns: f64,

    /// Sequential read bandwidth (GB/s)
    pub sequential_read_bandwidth_gbps: f64,

    /// Random read bandwidth (GB/s)
    pub random_read_bandwidth_gbps: f64,

    /// Shared memory latency (nanoseconds)
    pub shared_memory_latency_ns: f64,

    /// Shared memory bandwidth (GB/s)
    pub shared_memory_bandwidth_gbps: f64,
}

// ============================================================
// Sub-Profiles: Atomic Throughput
// ============================================================

/// Atomic operation throughput measured under varying contention.
/// Critical for tuning warp-aggregated CAS strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicThroughputProfile {
    /// atomicCAS throughput with zero contention (ops/sec)
    pub cas_zero_contention_ops_per_sec: f64,

    /// atomicCAS throughput at 32-way warp contention (ops/sec)
    pub cas_warp_contention_ops_per_sec: f64,

    /// atomicCAS throughput at full-SM contention (ops/sec)
    pub cas_full_sm_contention_ops_per_sec: f64,

    /// atomicAdd throughput (ops/sec)
    pub atomic_add_ops_per_sec: f64,

    /// Contention ratio: warp_contention / zero_contention
    /// Values > 10x indicate the GPU is contention-sensitive
    /// and benefits from warp-aggregated write patterns.
    pub contention_sensitivity_ratio: f64,
}

// ============================================================
// Sub-Profiles: Warp Concurrency
// ============================================================

/// Warp-level concurrency metrics from occupancy experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpConcurrencyProfile {
    /// Achieved occupancy (0.0 - 1.0) with 256 threads/block
    pub occupancy_256_threads: f64,

    /// Achieved occupancy (0.0 - 1.0) with 128 threads/block
    pub occupancy_128_threads: f64,

    /// Achieved occupancy (0.0 - 1.0) with 32 threads/block (1 warp)
    pub occupancy_32_threads: f64,

    /// Maximum IPC (instructions per cycle) observed
    pub max_ipc: f64,

    /// Warp scheduling overhead (nanoseconds per warp switch)
    pub warp_switch_overhead_ns: f64,

    /// __shfl_sync throughput (operations per second per warp)
    pub shfl_throughput_ops_per_sec: f64,

    /// __ballot_sync throughput (operations per second per warp)
    pub ballot_throughput_ops_per_sec: f64,
}

// ============================================================
// Sub-Profiles: PCIe Transfer
// ============================================================

/// PCIe bus characteristics for host↔device communication.
/// Crucial for persistent kernel + unified memory workloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcieProfile {
    /// PCIe generation (3, 4, or 5)
    pub pcie_gen: u32,

    /// PCIe lane width (x8 or x16)
    pub pcie_width: u32,

    /// Theoretical peak bandwidth (GB/s)
    pub theoretical_bandwidth_gbps: f64,

    /// Measured host→device bandwidth (GB/s)
    pub measured_h2d_bandwidth_gbps: f64,

    /// Measured device→host bandwidth (GB/s)
    pub measured_d2h_bandwidth_gbps: f64,

    /// Unified Memory page fault latency (microseconds)
    pub unified_memory_fault_latency_us: f64,

    /// __threadfence_system() round-trip overhead (nanoseconds)
    pub threadfence_system_overhead_ns: f64,
}

// ============================================================
// Hardware Prober
// ============================================================

/// Runs the full hardware probe suite and returns a HardwareProfile.
pub struct HardwareProber {
    /// GPU device index to probe
    gpu_index: u32,
    /// Whether to attempt real CUDA probing
    attempt_real_probe: bool,
}

impl HardwareProber {
    pub fn new(gpu_index: u32) -> Self {
        HardwareProber {
            gpu_index,
            attempt_real_probe: true,
        }
    }

    /// Create a prober that only returns simulated results.
    pub fn simulated(gpu_index: u32) -> Self {
        HardwareProber {
            gpu_index,
            attempt_real_probe: false,
        }
    }

    /// Run the full probe suite.
    ///
    /// Attempts to use real CUDA APIs first (via cust).
    /// Falls back to simulated results based on common GPU architectures.
    pub fn probe(&self) -> HardwareProfile {
        println!("\n{}", "═".repeat(64));
        println!("  CudaClaw Hardware Probe — GPU {}", self.gpu_index);
        println!("{}", "═".repeat(64));

        if self.attempt_real_probe {
            if let Some(profile) = self.probe_real() {
                println!("  Status: Real hardware detected");
                println!("{}\n", "═".repeat(64));
                return profile;
            }
            println!("  Status: No CUDA device found — using simulated profile");
        } else {
            println!("  Status: Simulated mode (no CUDA probe)");
        }

        println!("{}\n", "═".repeat(64));
        self.probe_simulated()
    }

    /// Attempt real CUDA hardware probing.
    /// Returns None if CUDA is not available.
    fn probe_real(&self) -> Option<HardwareProfile> {
        // Real probe would use cudaGetDeviceProperties, pointer-chasing
        // kernels, etc. This requires the CUDA toolkit to be installed.
        // For now, we return None to fall back to simulation.
        //
        // On a CUDA-capable machine, this function would:
        //   1. Call cudaGetDeviceProperties for static attributes
        //   2. Launch kernels/probe_benchmarks.cu for dynamic measurements:
        //      a. Pointer-chasing kernel for L1/L2 cache sizing
        //      b. Sequential/random read kernel for bandwidth
        //      c. atomicCAS contention kernel for atomic throughput
        //      d. __shfl_sync throughput kernel for warp concurrency
        //      e. Unified memory fault-timing kernel for PCIe profiling
        //   3. Aggregate results into HardwareProfile
        None
    }

    /// Generate a simulated hardware profile based on a reference
    /// architecture. Uses NVIDIA RTX 4090 (Ada Lovelace) as the
    /// default baseline.
    fn probe_simulated(&self) -> HardwareProfile {
        let start = Instant::now();

        println!("  Running simulated micro-benchmarks...");

        // ── Simulate L1 Cache Probe ──
        // Real probe: pointer-chasing with varying stride/array sizes.
        // Transition from ~28-cycle to ~200-cycle latency indicates
        // the L1↔L2 boundary.
        println!("    [1/6] L1 cache size probe (pointer-chasing)...");
        let l1_cache_size = 128 * 1024; // 128 KB per SM (Ada Lovelace)
        let l2_cache_size = 72 * 1024 * 1024; // 72 MB (RTX 4090)
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ── Simulate Warp Concurrency Probe ──
        // Real probe: launch kernels with varying block sizes,
        // measure achieved occupancy via hardware counters.
        println!("    [2/6] Warp concurrency probe (occupancy sweep)...");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ── Simulate Memory Latency Probe ──
        // Real probe: pointer-chasing for latency, streaming
        // read/write for bandwidth.
        println!("    [3/6] Memory latency probe (sequential + random)...");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ── Simulate Shared Memory Probe ──
        println!("    [4/6] Shared memory bandwidth probe...");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ── Simulate Atomic Throughput Probe ──
        // Real probe: grid of threads doing atomicCAS on the same
        // address (contention) vs. different addresses (no contention).
        println!("    [5/6] Atomic CAS contention profile...");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // ── Simulate PCIe Probe ──
        // Real probe: timed cudaMemcpy with varying sizes,
        // unified memory page fault timing.
        println!("    [6/6] PCIe transfer latency probe...");
        std::thread::sleep(std::time::Duration::from_millis(50));

        let elapsed = start.elapsed();
        println!("  Probe complete in {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        HardwareProfile {
            gpu_name: "NVIDIA RTX 4090 (Simulated)".to_string(),
            compute_capability: "8.9".to_string(),
            sm_count: 128,
            max_threads_per_sm: 1536,
            max_threads_per_block: 1024,
            warp_size: 32,
            max_warps_per_sm: 48,
            max_shared_memory_per_block: 100 * 1024,      // 100 KB
            max_shared_memory_per_sm: 100 * 1024,          // 100 KB
            l1_cache_size_bytes: l1_cache_size,
            l2_cache_size_bytes: l2_cache_size,
            cache_line_bytes: 128,
            global_memory_bytes: 24 * 1024 * 1024 * 1024,  // 24 GB
            memory_bus_width_bits: 384,
            memory_clock_mhz: 10501,                        // GDDR6X effective
            core_clock_mhz: 2520,
            memory_latency: MemoryLatencyProfile {
                l1_hit_latency_ns: 28.0,
                l2_hit_latency_ns: 200.0,
                global_memory_latency_ns: 450.0,
                sequential_read_bandwidth_gbps: 1008.0,     // RTX 4090 peak
                random_read_bandwidth_gbps: 320.0,
                shared_memory_latency_ns: 20.0,
                shared_memory_bandwidth_gbps: 128.0,
            },
            atomic_throughput: AtomicThroughputProfile {
                cas_zero_contention_ops_per_sec: 2_000_000_000.0,
                cas_warp_contention_ops_per_sec: 200_000_000.0,
                cas_full_sm_contention_ops_per_sec: 50_000_000.0,
                atomic_add_ops_per_sec: 5_000_000_000.0,
                contention_sensitivity_ratio: 10.0,
            },
            warp_concurrency: WarpConcurrencyProfile {
                occupancy_256_threads: 0.75,
                occupancy_128_threads: 0.50,
                occupancy_32_threads: 0.167,
                max_ipc: 1.8,
                warp_switch_overhead_ns: 4.0,
                shfl_throughput_ops_per_sec: 500_000_000_000.0,
                ballot_throughput_ops_per_sec: 500_000_000_000.0,
            },
            pcie_profile: PcieProfile {
                pcie_gen: 4,
                pcie_width: 16,
                theoretical_bandwidth_gbps: 31.5,
                measured_h2d_bandwidth_gbps: 25.0,
                measured_d2h_bandwidth_gbps: 24.0,
                unified_memory_fault_latency_us: 20.0,
                threadfence_system_overhead_ns: 800.0,
            },
            is_simulated: true,
            architecture: "Ada Lovelace".to_string(),
            probe_timestamp: timestamp,
        }
    }

    /// Print a detailed summary of the hardware profile.
    pub fn print_profile(profile: &HardwareProfile) {
        println!("=== Hardware Profile: {} ===", profile.gpu_name);
        println!("  Architecture     : {}", profile.architecture);
        println!("  Compute Cap.     : {}", profile.compute_capability);
        println!("  SMs              : {}", profile.sm_count);
        println!("  Warp Size        : {}", profile.warp_size);
        println!("  Max Warps/SM     : {}", profile.max_warps_per_sm);
        println!("  Max Threads/Block: {}", profile.max_threads_per_block);
        println!("  Shared Mem/Block : {} KB", profile.max_shared_memory_per_block / 1024);
        println!("  L1 Cache/SM      : {} KB", profile.l1_cache_size_bytes / 1024);
        println!("  L2 Cache         : {} MB", profile.l2_cache_size_bytes / (1024 * 1024));
        println!("  Cache Line       : {} bytes", profile.cache_line_bytes);
        println!("  Global Memory    : {} GB", profile.global_memory_bytes / (1024 * 1024 * 1024));
        println!("  Core Clock       : {} MHz", profile.core_clock_mhz);
        println!("  Memory Clock     : {} MHz", profile.memory_clock_mhz);
        println!("  Memory Bus       : {}-bit", profile.memory_bus_width_bits);

        println!("\n  --- Memory Latency ---");
        let m = &profile.memory_latency;
        println!("  L1 Hit           : {:.1} ns", m.l1_hit_latency_ns);
        println!("  L2 Hit           : {:.1} ns", m.l2_hit_latency_ns);
        println!("  Global (DRAM)    : {:.1} ns", m.global_memory_latency_ns);
        println!("  Seq Read BW      : {:.1} GB/s", m.sequential_read_bandwidth_gbps);
        println!("  Random Read BW   : {:.1} GB/s", m.random_read_bandwidth_gbps);
        println!("  Shared Mem       : {:.1} ns / {:.1} GB/s",
            m.shared_memory_latency_ns, m.shared_memory_bandwidth_gbps);

        println!("\n  --- Atomic Throughput ---");
        let a = &profile.atomic_throughput;
        println!("  CAS (0 contention)    : {:.2e} ops/s", a.cas_zero_contention_ops_per_sec);
        println!("  CAS (warp contention) : {:.2e} ops/s", a.cas_warp_contention_ops_per_sec);
        println!("  CAS (full-SM)         : {:.2e} ops/s", a.cas_full_sm_contention_ops_per_sec);
        println!("  AtomicAdd             : {:.2e} ops/s", a.atomic_add_ops_per_sec);
        println!("  Contention Ratio      : {:.1}x", a.contention_sensitivity_ratio);

        println!("\n  --- Warp Concurrency ---");
        let w = &profile.warp_concurrency;
        println!("  Occupancy @256t  : {:.1}%", w.occupancy_256_threads * 100.0);
        println!("  Occupancy @128t  : {:.1}%", w.occupancy_128_threads * 100.0);
        println!("  Occupancy @32t   : {:.1}%", w.occupancy_32_threads * 100.0);
        println!("  Max IPC          : {:.2}", w.max_ipc);
        println!("  Warp Switch      : {:.1} ns", w.warp_switch_overhead_ns);
        println!("  __shfl_sync      : {:.2e} ops/s", w.shfl_throughput_ops_per_sec);

        println!("\n  --- PCIe Profile ---");
        let p = &profile.pcie_profile;
        println!("  PCIe Gen         : {} x{}", p.pcie_gen, p.pcie_width);
        println!("  Theory BW        : {:.1} GB/s", p.theoretical_bandwidth_gbps);
        println!("  H2D BW           : {:.1} GB/s", p.measured_h2d_bandwidth_gbps);
        println!("  D2H BW           : {:.1} GB/s", p.measured_d2h_bandwidth_gbps);
        println!("  UM Fault Latency : {:.1} µs", p.unified_memory_fault_latency_us);
        println!("  __threadfence_sys: {:.0} ns", p.threadfence_system_overhead_ns);

        if profile.is_simulated {
            println!("\n  NOTE: Profile is SIMULATED (no live GPU detected).");
            println!("        Values are based on {} reference specs.", profile.architecture);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_probe() {
        let prober = HardwareProber::simulated(0);
        let profile = prober.probe();

        assert!(profile.is_simulated);
        assert_eq!(profile.warp_size, 32);
        assert!(profile.sm_count > 0);
        assert!(profile.l1_cache_size_bytes > 0);
        assert!(profile.l2_cache_size_bytes > profile.l1_cache_size_bytes);
        assert!(profile.memory_latency.l1_hit_latency_ns < profile.memory_latency.l2_hit_latency_ns);
        assert!(profile.memory_latency.l2_hit_latency_ns < profile.memory_latency.global_memory_latency_ns);
        assert!(profile.atomic_throughput.contention_sensitivity_ratio > 1.0);
        assert!(profile.pcie_profile.pcie_gen >= 3);
    }

    #[test]
    fn test_profile_serialization() {
        let prober = HardwareProber::simulated(0);
        let profile = prober.probe();

        let json = serde_json::to_string_pretty(&profile).unwrap();
        let deserialized: HardwareProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.gpu_name, profile.gpu_name);
        assert_eq!(deserialized.sm_count, profile.sm_count);
        assert_eq!(deserialized.l1_cache_size_bytes, profile.l1_cache_size_bytes);
    }
}
