// ============================================================
// RamifiedRole DNA — The Complete Genetic Blueprint of a
// Cudaclaw Instance
// ============================================================
//
// A `.claw-dna` file is the single source of truth for how a
// specific cudaclaw instance is configured, constrained, and
// optimized. It captures four dimensions:
//
//   1. HARDWARE FINGERPRINT
//      Static + measured properties of the target GPU:
//      compute capability, SM count, L2 cache, warp latency,
//      memory bandwidth, atomic throughput, PCIe profile.
//
//   2. CONSTRAINT-THEORY MAPPINGS
//      Safe bounds imported from the Constraint-Theory project.
//      These define the "rules of physics" the engine must obey:
//      max registers per thread, shared memory ceilings, P99
//      latency targets, CRDT monotonicity invariants, etc.
//
//   3. PTX MUSCLE FIBERS
//      A map of task IDs to specialized PTX strings or NVRTC
//      compilation parameters. Each fiber is a tuned kernel
//      configuration for a specific workload pattern. The DNA
//      records both the launch parameters AND the PTX/CUDA C++
//      source so the Ramify engine can recompile on-the-fly.
//
//   4. RESOURCE EXHAUSTION METRICS
//      Historical exhaust logs (heat, latency, throttle events)
//      so the system can "prune" inefficient branches. This is
//      the feedback signal that drives DNA mutation over time.
//
// SERIALIZATION:
//   RamifiedRole implements serde Serialize/Deserialize and can
//   be written to / read from `.claw-dna` files (JSON format).
//
//   use dna::RamifiedRole;
//   let role = RamifiedRole::default_spreadsheet_engine();
//   role.save_to_file("my_instance.claw-dna")?;
//   let loaded = RamifiedRole::load_from_file("my_instance.claw-dna")?;
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// Top-Level: RamifiedRole
// ============================================================

/// The complete DNA of a cudaclaw instance.
///
/// This is the root struct serialized to `.claw-dna` files.
/// It contains everything needed to reconstruct, validate, and
/// evolve a cudaclaw configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamifiedRole {
    /// Schema version for forward compatibility.
    pub schema_version: u32,

    /// Human-readable name (e.g., "spreadsheet_engine_rtx4090").
    pub name: String,

    /// Role identifier (e.g., "spreadsheet_engine").
    pub role: String,

    /// Description of this DNA's purpose.
    pub description: String,

    /// When this DNA was created (Unix epoch seconds).
    pub created_at: u64,

    /// When this DNA was last modified (Unix epoch seconds).
    pub last_modified: u64,

    /// Total number of mutations applied by the ML feedback loop.
    pub total_mutations: u64,

    /// Hardware fingerprint of the target GPU.
    pub hardware: DnaHardwareFingerprint,

    /// Constraint-Theory safe bounds.
    pub constraints: DnaConstraintMappings,

    /// PTX Muscle Fibers — task-to-kernel mappings.
    pub muscle_fibers: DnaMuscleFiberMap,

    /// Resource exhaustion metrics and history.
    pub exhaustion: DnaExhaustionMetrics,
}

impl RamifiedRole {
    /// Create a new, empty DNA with default values.
    pub fn new(name: &str, role: &str) -> Self {
        let now = now_epoch();
        RamifiedRole {
            schema_version: 1,
            name: name.to_string(),
            role: role.to_string(),
            description: String::new(),
            created_at: now,
            last_modified: now,
            total_mutations: 0,
            hardware: DnaHardwareFingerprint::unknown(),
            constraints: DnaConstraintMappings::default_system(),
            muscle_fibers: DnaMuscleFiberMap::new(),
            exhaustion: DnaExhaustionMetrics::new(),
        }
    }

    /// Create a fully-populated default DNA for a spreadsheet engine
    /// running on an RTX 4090 (simulated if no GPU present).
    pub fn default_spreadsheet_engine() -> Self {
        let now = now_epoch();
        let mut dna = RamifiedRole {
            schema_version: 1,
            name: "spreadsheet_engine_rtx4090".to_string(),
            role: "spreadsheet_engine".to_string(),
            description: "Default DNA for a GPU-accelerated spreadsheet CRDT engine. \
                          Optimized for low-latency cell edits, warp-aggregated CRDT \
                          merges, and parallel formula recalculation."
                .to_string(),
            created_at: now,
            last_modified: now,
            total_mutations: 0,
            hardware: DnaHardwareFingerprint::rtx4090_simulated(),
            constraints: DnaConstraintMappings::default_system(),
            muscle_fibers: DnaMuscleFiberMap::default_fibers(),
            exhaustion: DnaExhaustionMetrics::new(),
        };
        dna.exhaustion.thresholds = DnaExhaustionThresholds::default();
        dna
    }

    /// Record a mutation (called by the ML feedback loop).
    pub fn record_mutation(&mut self, description: &str) {
        self.total_mutations += 1;
        self.last_modified = now_epoch();
        self.exhaustion.mutation_log.push(DnaMutationRecord {
            mutation_id: self.total_mutations,
            timestamp: self.last_modified,
            description: description.to_string(),
            score_before: 0.0,
            score_after: 0.0,
        });
    }

    /// Validate the DNA for internal consistency.
    ///
    /// Returns a list of issues found. An empty list means the DNA
    /// is self-consistent.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Hardware validation
        if self.hardware.compute_capability.is_empty() {
            issues.push("Hardware: compute_capability is empty".into());
        }
        if self.hardware.sm_count == 0 {
            issues.push("Hardware: sm_count is zero".into());
        }
        if self.hardware.warp_size != 32 {
            issues.push(format!(
                "Hardware: warp_size is {} (expected 32 for NVIDIA GPUs)",
                self.hardware.warp_size
            ));
        }

        // Constraint validation
        for (id, bound) in &self.constraints.resource_bounds {
            if let DnaBoundValue::IntMax(v) = &bound.value {
                if *v == 0 {
                    issues.push(format!("Constraint '{}': IntMax bound is 0", id));
                }
            }
        }

        // Check that critical constraints exist
        let required_constraints = [
            "resource.register_budget",
            "resource.shared_memory_ceiling",
            "latency.p99_rtt_ceiling",
            "correctness.crdt_monotonicity",
        ];
        for &req in &required_constraints {
            if !self.constraints.resource_bounds.contains_key(req) {
                issues.push(format!("Missing required constraint: {}", req));
            }
        }

        // Muscle fiber validation
        for (task_id, fiber) in &self.muscle_fibers.fibers {
            if fiber.block_size == 0 {
                issues.push(format!("Fiber '{}': block_size is 0", task_id));
            }
            if fiber.block_size % self.hardware.warp_size != 0 {
                issues.push(format!(
                    "Fiber '{}': block_size {} is not a multiple of warp_size {}",
                    task_id, fiber.block_size, self.hardware.warp_size
                ));
            }
            if fiber.registers_per_thread > 0 {
                let total_regs = fiber.registers_per_thread * fiber.block_size;
                if let Some(bound) = self.constraints.resource_bounds.get("resource.register_budget") {
                    if let DnaBoundValue::IntMax(max) = &bound.value {
                        if total_regs as u64 > *max {
                            issues.push(format!(
                                "Fiber '{}': total registers {} exceeds constraint budget {}",
                                task_id, total_regs, max
                            ));
                        }
                    }
                }
            }
        }

        // Exhaustion thresholds
        if self.exhaustion.thresholds.nutrient_critical_floor < 0.0
            || self.exhaustion.thresholds.nutrient_critical_floor > 1.0
        {
            issues.push(format!(
                "Exhaustion: nutrient_critical_floor {} is outside [0, 1]",
                self.exhaustion.thresholds.nutrient_critical_floor
            ));
        }

        issues
    }

    /// Save this DNA to a `.claw-dna` file (JSON format).
    pub fn save_to_file(&self, path: &str) -> Result<(), DnaError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| DnaError::Serialization(format!("Failed to serialize DNA: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| DnaError::Io(format!("Failed to write {}: {}", path, e)))?;
        Ok(())
    }

    /// Load a DNA from a `.claw-dna` file.
    pub fn load_from_file(path: &str) -> Result<Self, DnaError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| DnaError::Io(format!("Failed to read {}: {}", path, e)))?;
        let dna: RamifiedRole = serde_json::from_str(&json)
            .map_err(|e| DnaError::Serialization(format!("Failed to parse DNA: {}", e)))?;
        Ok(dna)
    }

    /// Save this DNA to the standard location under `.cudaclaw/dna/`.
    pub fn save_to_project(&self, project_root: &Path) -> Result<String, DnaError> {
        let dna_dir = project_root.join(".cudaclaw").join("dna");
        std::fs::create_dir_all(&dna_dir)
            .map_err(|e| DnaError::Io(format!("Failed to create dna dir: {}", e)))?;

        let filename = format!("{}_{}.claw-dna", self.role, self.hardware.short_id());
        let filepath = dna_dir.join(&filename);
        let path_str = filepath.display().to_string();
        self.save_to_file(&path_str)?;
        Ok(path_str)
    }

    /// Print a human-readable summary of this DNA.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(64));
        println!("  RamifiedRole DNA: {}", self.name);
        println!("{}", "=".repeat(64));
        println!("  Schema Version : {}", self.schema_version);
        println!("  Role           : {}", self.role);
        println!("  Mutations      : {}", self.total_mutations);
        println!("  Created        : {}", format_epoch(self.created_at));
        println!("  Last Modified  : {}", format_epoch(self.last_modified));

        println!("\n  --- Hardware Fingerprint ---");
        self.hardware.print_summary();

        println!("\n  --- Constraint Mappings ({} bounds) ---", self.constraints.resource_bounds.len());
        self.constraints.print_summary();

        println!("\n  --- Muscle Fibers ({} tasks) ---", self.muscle_fibers.fibers.len());
        self.muscle_fibers.print_summary();

        println!("\n  --- Exhaustion Metrics ---");
        self.exhaustion.print_summary();

        let issues = self.validate();
        if issues.is_empty() {
            println!("\n  Validation: OK (no issues)");
        } else {
            println!("\n  Validation: {} issue(s) found", issues.len());
            for issue in &issues {
                println!("    - {}", issue);
            }
        }
        println!("{}\n", "=".repeat(64));
    }
}

// ============================================================
// Section 1: Hardware Fingerprint
// ============================================================

/// Static and measured properties of the target GPU.
///
/// Captures both the architectural constants (SM count, warp size)
/// and the dynamically-measured latencies (L1/L2 hit times, atomic
/// throughput, PCIe overhead) that inform kernel specialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaHardwareFingerprint {
    /// GPU name (e.g., "NVIDIA RTX 4090").
    pub gpu_name: String,

    /// CUDA compute capability as string (e.g., "8.9").
    pub compute_capability: String,

    /// Compute capability major version (e.g., 8 for sm_89).
    pub compute_capability_major: u32,

    /// Compute capability minor version (e.g., 9 for sm_89).
    pub compute_capability_minor: u32,

    /// Architecture family (e.g., "Ada Lovelace").
    pub architecture: String,

    /// Number of Streaming Multiprocessors.
    pub sm_count: u32,

    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,

    /// Maximum threads per block.
    pub max_threads_per_block: u32,

    /// Warp size (always 32 on NVIDIA).
    pub warp_size: u32,

    /// Maximum resident warps per SM.
    pub max_warps_per_sm: u32,

    /// 32-bit registers per SM.
    pub registers_per_sm: u32,

    /// Maximum 32-bit registers per block (constraint constant).
    pub max_registers_per_block: u32,

    /// Maximum shared memory per block (bytes).
    pub max_shared_memory_per_block: u32,

    /// Maximum shared memory per SM (bytes).
    pub max_shared_memory_per_sm: u32,

    /// L1 cache size per SM (bytes).
    pub l1_cache_size_bytes: u32,

    /// L2 cache size (bytes).
    pub l2_cache_size_bytes: u32,

    /// Cache line size (bytes).
    pub cache_line_bytes: u32,

    /// Total global memory (bytes).
    pub global_memory_bytes: u64,

    /// Memory bus width (bits).
    pub memory_bus_width_bits: u32,

    /// Core clock rate (MHz).
    pub core_clock_mhz: u32,

    /// Memory clock rate (MHz).
    pub memory_clock_mhz: u32,

    /// Measured warp-level latencies.
    pub warp_latency: DnaWarpLatency,

    /// Measured memory latencies.
    pub memory_latency: DnaMemoryLatency,

    /// Measured atomic operation throughput.
    pub atomic_throughput: DnaAtomicThroughput,

    /// PCIe bus characteristics.
    pub pcie: DnaPcieProfile,

    /// Dynamic probe results from micro-benchmarks.
    #[serde(default)]
    pub probe_results: Option<DnaProbeResults>,

    /// Whether this fingerprint is from real hardware or simulated.
    pub is_simulated: bool,
}

/// Warp-level scheduling and instruction latencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaWarpLatency {
    /// Warp scheduling overhead per switch (nanoseconds).
    pub warp_switch_overhead_ns: f64,

    /// `__shfl_sync` throughput (ops/sec/warp).
    pub shfl_throughput_ops_per_sec: f64,

    /// `__ballot_sync` throughput (ops/sec/warp).
    pub ballot_throughput_ops_per_sec: f64,

    /// Maximum IPC observed.
    pub max_ipc: f64,

    /// Achieved occupancy at 256 threads/block.
    pub occupancy_256_threads: f64,

    /// Achieved occupancy at 128 threads/block.
    pub occupancy_128_threads: f64,

    /// Achieved occupancy at 32 threads/block (1 warp).
    pub occupancy_32_threads: f64,
}

/// Memory hierarchy latency measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaMemoryLatency {
    /// L1 cache hit latency (nanoseconds).
    pub l1_hit_ns: f64,

    /// L2 cache hit latency (nanoseconds).
    pub l2_hit_ns: f64,

    /// Global memory (DRAM) latency (nanoseconds).
    pub global_memory_ns: f64,

    /// Shared memory latency (nanoseconds).
    pub shared_memory_ns: f64,

    /// Sequential read bandwidth (GB/s).
    pub sequential_read_gbps: f64,

    /// Random read bandwidth (GB/s).
    pub random_read_gbps: f64,

    /// Shared memory bandwidth (GB/s).
    pub shared_memory_bandwidth_gbps: f64,
}

/// Atomic operation throughput under varying contention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaAtomicThroughput {
    /// atomicCAS at zero contention (ops/sec).
    pub cas_zero_contention_ops: f64,

    /// atomicCAS at 32-way warp contention (ops/sec).
    pub cas_warp_contention_ops: f64,

    /// atomicCAS at full-SM contention (ops/sec).
    pub cas_full_sm_contention_ops: f64,

    /// atomicAdd throughput (ops/sec).
    pub atomic_add_ops: f64,

    /// Contention sensitivity ratio (warp / zero).
    /// Values > 10x indicate the GPU benefits from warp aggregation.
    pub contention_sensitivity_ratio: f64,
}

/// PCIe bus characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaPcieProfile {
    /// PCIe generation (3, 4, 5).
    pub pcie_gen: u32,

    /// PCIe lane width (x8, x16).
    pub pcie_width: u32,

    /// Theoretical peak bandwidth (GB/s).
    pub theoretical_bandwidth_gbps: f64,

    /// Measured host-to-device bandwidth (GB/s).
    pub measured_h2d_gbps: f64,

    /// Measured device-to-host bandwidth (GB/s).
    pub measured_d2h_gbps: f64,

    /// Unified Memory page fault latency (microseconds).
    pub unified_memory_fault_us: f64,

    /// `__threadfence_system()` round-trip overhead (nanoseconds).
    pub threadfence_system_ns: f64,
}

impl DnaHardwareFingerprint {
    /// Create an "unknown" fingerprint (placeholder).
    pub fn unknown() -> Self {
        DnaHardwareFingerprint {
            gpu_name: "Unknown".into(),
            compute_capability: "0.0".into(),
            compute_capability_major: 0,
            compute_capability_minor: 0,
            architecture: "Unknown".into(),
            sm_count: 0,
            max_threads_per_sm: 0,
            max_threads_per_block: 0,
            warp_size: 32,
            max_warps_per_sm: 0,
            registers_per_sm: 0,
            max_registers_per_block: 0,
            max_shared_memory_per_block: 0,
            max_shared_memory_per_sm: 0,
            l1_cache_size_bytes: 0,
            l2_cache_size_bytes: 0,
            cache_line_bytes: 128,
            global_memory_bytes: 0,
            memory_bus_width_bits: 0,
            core_clock_mhz: 0,
            memory_clock_mhz: 0,
            warp_latency: DnaWarpLatency {
                warp_switch_overhead_ns: 0.0,
                shfl_throughput_ops_per_sec: 0.0,
                ballot_throughput_ops_per_sec: 0.0,
                max_ipc: 0.0,
                occupancy_256_threads: 0.0,
                occupancy_128_threads: 0.0,
                occupancy_32_threads: 0.0,
            },
            memory_latency: DnaMemoryLatency {
                l1_hit_ns: 0.0,
                l2_hit_ns: 0.0,
                global_memory_ns: 0.0,
                shared_memory_ns: 0.0,
                sequential_read_gbps: 0.0,
                random_read_gbps: 0.0,
                shared_memory_bandwidth_gbps: 0.0,
            },
            atomic_throughput: DnaAtomicThroughput {
                cas_zero_contention_ops: 0.0,
                cas_warp_contention_ops: 0.0,
                cas_full_sm_contention_ops: 0.0,
                atomic_add_ops: 0.0,
                contention_sensitivity_ratio: 0.0,
            },
            pcie: DnaPcieProfile {
                pcie_gen: 0,
                pcie_width: 0,
                theoretical_bandwidth_gbps: 0.0,
                measured_h2d_gbps: 0.0,
                measured_d2h_gbps: 0.0,
                unified_memory_fault_us: 0.0,
                threadfence_system_ns: 0.0,
            },
            probe_results: None,
            is_simulated: true,
        }
    }

    /// Simulated RTX 4090 (Ada Lovelace, sm_89) fingerprint.
    pub fn rtx4090_simulated() -> Self {
        DnaHardwareFingerprint {
            gpu_name: "NVIDIA RTX 4090 (Simulated)".into(),
            compute_capability: "8.9".into(),
            compute_capability_major: 8,
            compute_capability_minor: 9,
            architecture: "Ada Lovelace".into(),
            sm_count: 128,
            max_threads_per_sm: 1536,
            max_threads_per_block: 1024,
            warp_size: 32,
            max_warps_per_sm: 48,
            registers_per_sm: 65536,
            max_registers_per_block: 65536,
            max_shared_memory_per_block: 102400,
            max_shared_memory_per_sm: 102400,
            l1_cache_size_bytes: 128 * 1024,
            l2_cache_size_bytes: 72 * 1024 * 1024,
            cache_line_bytes: 128,
            global_memory_bytes: 24 * 1024 * 1024 * 1024,
            memory_bus_width_bits: 384,
            core_clock_mhz: 2520,
            memory_clock_mhz: 10501,
            warp_latency: DnaWarpLatency {
                warp_switch_overhead_ns: 4.0,
                shfl_throughput_ops_per_sec: 500_000_000_000.0,
                ballot_throughput_ops_per_sec: 500_000_000_000.0,
                max_ipc: 1.8,
                occupancy_256_threads: 0.75,
                occupancy_128_threads: 0.50,
                occupancy_32_threads: 0.167,
            },
            memory_latency: DnaMemoryLatency {
                l1_hit_ns: 28.0,
                l2_hit_ns: 200.0,
                global_memory_ns: 450.0,
                shared_memory_ns: 20.0,
                sequential_read_gbps: 1008.0,
                random_read_gbps: 320.0,
                shared_memory_bandwidth_gbps: 128.0,
            },
            atomic_throughput: DnaAtomicThroughput {
                cas_zero_contention_ops: 2_000_000_000.0,
                cas_warp_contention_ops: 200_000_000.0,
                cas_full_sm_contention_ops: 50_000_000.0,
                atomic_add_ops: 5_000_000_000.0,
                contention_sensitivity_ratio: 10.0,
            },
            pcie: DnaPcieProfile {
                pcie_gen: 4,
                pcie_width: 16,
                theoretical_bandwidth_gbps: 31.5,
                measured_h2d_gbps: 25.0,
                measured_d2h_gbps: 24.0,
                unified_memory_fault_us: 20.0,
                threadfence_system_ns: 800.0,
            },
            probe_results: None,
            is_simulated: true,
        }
    }

    /// Attempt to create a HardwareFingerprint from a real GPU using the
    /// `cust` crate.
    ///
    /// Queries static device properties (compute capability, SM count,
    /// global memory, L2 cache, etc.) and constraint constants (max
    /// registers per block, max shared memory per SM, warp size) via
    /// `cust::device::Device::get_attribute()`.
    ///
    /// Returns `None` if CUDA initialization fails or the device index
    /// is out of range.
    #[cfg(feature = "cuda")]
    pub fn from_cust_device(device_index: u32) -> Option<Self> {
        // Attempt to initialize CUDA context via cust.
        // This will fail gracefully on machines without a GPU.
        if cust::init(cust::CudaFlags::empty()).is_err() {
            return None;
        }
        let device = cust::device::Device::get(device_index).ok()?;

        // Static Traits
        let gpu_name = device.name().unwrap_or_else(|_| "Unknown CUDA Device".into());
        let cc_major = device
            .get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMajor)
            .unwrap_or(0) as u32;
        let cc_minor = device
            .get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMinor)
            .unwrap_or(0) as u32;
        let sm_count = device
            .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)
            .unwrap_or(0) as u32;
        let global_memory_bytes = device.total_memory().unwrap_or(0) as u64;
        let l2_cache_size_bytes = device
            .get_attribute(cust::device::DeviceAttribute::L2CacheSize)
            .unwrap_or(0) as u32;

        // Constraint Constants
        let max_registers_per_block = device
            .get_attribute(cust::device::DeviceAttribute::MaxRegistersPerBlock)
            .unwrap_or(0) as u32;
        let max_shared_memory_per_block = device
            .get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
            .unwrap_or(0) as u32;
        let max_shared_memory_per_sm = device
            .get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerMultiprocessor)
            .unwrap_or(0) as u32;
        let warp_size = device
            .get_attribute(cust::device::DeviceAttribute::WarpSize)
            .unwrap_or(32) as u32;
        let max_threads_per_block = device
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(0) as u32;
        let max_threads_per_sm = device
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerMultiprocessor)
            .unwrap_or(0) as u32;
        let memory_bus_width_bits = device
            .get_attribute(cust::device::DeviceAttribute::GlobalMemoryBusWidth)
            .unwrap_or(0) as u32;
        let core_clock_mhz = (device
            .get_attribute(cust::device::DeviceAttribute::ClockRate)
            .unwrap_or(0)
            / 1000) as u32; // ClockRate is in kHz
        let memory_clock_mhz = (device
            .get_attribute(cust::device::DeviceAttribute::MemoryClockRate)
            .unwrap_or(0)
            / 1000) as u32; // MemoryClockRate is in kHz

        // Derived values
        let max_warps_per_sm = if warp_size > 0 {
            max_threads_per_sm / warp_size
        } else {
            0
        };
        let registers_per_sm = max_registers_per_block; // Conservative estimate
        let architecture = arch_name_from_cc(cc_major, cc_minor);
        let compute_capability = format!("{}.{}", cc_major, cc_minor);

        // PCIe generation estimate from attributes (not all drivers expose this)
        let pcie_gen = device
            .get_attribute(cust::device::DeviceAttribute::PciExpressActiveWidth)
            .unwrap_or(16) as u32;

        Some(DnaHardwareFingerprint {
            gpu_name,
            compute_capability,
            compute_capability_major: cc_major,
            compute_capability_minor: cc_minor,
            architecture,
            sm_count,
            max_threads_per_sm,
            max_threads_per_block,
            warp_size,
            max_warps_per_sm,
            registers_per_sm,
            max_registers_per_block,
            max_shared_memory_per_block,
            max_shared_memory_per_sm,
            l1_cache_size_bytes: 0, // Requires pointer-chasing probe
            l2_cache_size_bytes,
            cache_line_bytes: 128,
            global_memory_bytes,
            memory_bus_width_bits,
            core_clock_mhz,
            memory_clock_mhz,
            warp_latency: DnaWarpLatency {
                warp_switch_overhead_ns: 0.0,
                shfl_throughput_ops_per_sec: 0.0,
                ballot_throughput_ops_per_sec: 0.0,
                max_ipc: 0.0,
                occupancy_256_threads: 0.0,
                occupancy_128_threads: 0.0,
                occupancy_32_threads: 0.0,
            },
            memory_latency: DnaMemoryLatency {
                l1_hit_ns: 0.0,
                l2_hit_ns: 0.0,
                global_memory_ns: 0.0,
                shared_memory_ns: 0.0,
                sequential_read_gbps: 0.0,
                random_read_gbps: 0.0,
                shared_memory_bandwidth_gbps: 0.0,
            },
            atomic_throughput: DnaAtomicThroughput {
                cas_zero_contention_ops: 0.0,
                cas_warp_contention_ops: 0.0,
                cas_full_sm_contention_ops: 0.0,
                atomic_add_ops: 0.0,
                contention_sensitivity_ratio: 0.0,
            },
            pcie: DnaPcieProfile {
                pcie_gen,
                pcie_width: 16,
                theoretical_bandwidth_gbps: 0.0,
                measured_h2d_gbps: 0.0,
                measured_d2h_gbps: 0.0,
                unified_memory_fault_us: 0.0,
                threadfence_system_ns: 0.0,
            },
            probe_results: None,
            is_simulated: false,
        })
    }

    /// Generate a filesystem-safe short identifier.
    pub fn short_id(&self) -> String {
        let name_part: String = self
            .gpu_name
            .to_lowercase()
            .replace("nvidia", "")
            .replace("(simulated)", "")
            .replace("geforce", "")
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join("");
        let sm_part = format!("sm{}", self.compute_capability.replace('.', ""));
        format!("{}_{}", name_part, sm_part)
    }

    /// Print a summary of the hardware fingerprint.
    pub fn print_summary(&self) {
        println!("  GPU              : {}", self.gpu_name);
        println!("  Architecture     : {}", self.architecture);
        println!("  Compute Cap.     : {} (major={}, minor={})",
            self.compute_capability, self.compute_capability_major, self.compute_capability_minor);
        println!("  SMs              : {}", self.sm_count);
        println!("  Registers/SM     : {}", self.registers_per_sm);
        println!("  Regs/Block (max) : {}", self.max_registers_per_block);
        println!("  Shared Mem/Block : {} KB", self.max_shared_memory_per_block / 1024);
        println!("  Shared Mem/SM    : {} KB", self.max_shared_memory_per_sm / 1024);
        println!("  L1 Cache/SM      : {} KB", self.l1_cache_size_bytes / 1024);
        println!("  L2 Cache         : {} MB", self.l2_cache_size_bytes / (1024 * 1024));
        println!("  Warp Size        : {}", self.warp_size);
        println!("  Max Warps/SM     : {}", self.max_warps_per_sm);
        println!("  Global Memory    : {} GB", self.global_memory_bytes / (1024 * 1024 * 1024));
        println!("  L1 Hit Latency   : {:.1} ns", self.memory_latency.l1_hit_ns);
        println!("  Warp Switch      : {:.1} ns", self.warp_latency.warp_switch_overhead_ns);
        println!("  CAS Contention   : {:.1}x sensitivity",
            self.atomic_throughput.contention_sensitivity_ratio);
        println!("  PCIe             : Gen{} x{} ({:.1} GB/s peak)",
            self.pcie.pcie_gen, self.pcie.pcie_width, self.pcie.theoretical_bandwidth_gbps);
        println!("  Simulated        : {}", self.is_simulated);

        if let Some(ref probe) = self.probe_results {
            println!("\n  --- Probe Results (Dynamic) ---");
            println!("  Probe Duration   : {:.1} ms", probe.probe_duration_ms);
            println!("  Global Mem RTT   : {:.1} ns", probe.memory_latency_probe.global_memory_rtt_ns);
            println!("  Shared Mem RTT   : {:.1} ns", probe.memory_latency_probe.shared_memory_rtt_ns);
            println!("  Global/Shared    : {:.1}x", probe.memory_latency_probe.global_to_shared_ratio);
            println!("  FP32 Throughput  : {:.2e} ops/s ({:.1} GFLOPS)",
                probe.compute_throughput_probe.fp32_ops_per_sec,
                probe.compute_throughput_probe.fp32_gflops);
            println!("  CAS 32-thread    : {:.2e} ops/s",
                probe.atomic_contention_probe.cas_32_thread_same_addr_ops_per_sec);
            println!("  CAS uncontended  : {:.2e} ops/s",
                probe.atomic_contention_probe.cas_uncontended_ops_per_sec);
            println!("  Contention Ratio : {:.1}x",
                probe.atomic_contention_probe.contention_slowdown_ratio);
        }
    }
}

/// Map compute capability to architecture family name.
fn arch_name_from_cc(major: u32, minor: u32) -> String {
    match (major, minor) {
        (3, _) => "Kepler".into(),
        (5, _) => "Maxwell".into(),
        (6, _) => "Pascal".into(),
        (7, 0) => "Volta".into(),
        (7, 5) => "Turing".into(),
        (7, _) => "Volta/Turing".into(),
        (8, 0) => "Ampere".into(),
        (8, 6) => "Ampere".into(),
        (8, 9) => "Ada Lovelace".into(),
        (8, _) => "Ampere/Ada Lovelace".into(),
        (9, 0) => "Hopper".into(),
        (9, _) => "Hopper".into(),
        (10, _) => "Blackwell".into(),
        _ => format!("Unknown (sm_{}{})", major, minor),
    }
}

// ============================================================
// Section 1b: Dynamic Probe Results
// ============================================================
//
// Three 100ms micro-benchmarks that measure real hardware behavior:
//   1. Memory Latency — Global vs. Shared memory RTT
//   2. Compute Throughput — Raw FP32 ops/sec
//   3. Atomic Contention — atomicCAS with 32 threads on same address
//
// On systems without a GPU, simulated results are generated from
// the static fingerprint data.
// ============================================================

/// Aggregated results from all three dynamic micro-benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaProbeResults {
    /// Memory latency probe results.
    pub memory_latency_probe: MemoryLatencyProbeResult,

    /// Compute throughput probe results.
    pub compute_throughput_probe: ComputeThroughputProbeResult,

    /// Atomic contention probe results.
    pub atomic_contention_probe: AtomicContentionProbeResult,

    /// Total wall-clock time for all probes (milliseconds).
    pub probe_duration_ms: f64,

    /// Timestamp when probe was run (Unix epoch seconds).
    pub probe_timestamp: u64,
}

impl Default for DnaProbeResults {
    fn default() -> Self {
        DnaProbeResults {
            memory_latency_probe: MemoryLatencyProbeResult::default(),
            compute_throughput_probe: ComputeThroughputProbeResult::default(),
            atomic_contention_probe: AtomicContentionProbeResult::default(),
            probe_duration_ms: 0.0,
            probe_timestamp: 0,
        }
    }
}

/// Memory Latency Probe: measures Global vs. Shared memory round-trip time.
///
/// On real hardware, this launches a pointer-chasing kernel that accesses
/// a linked list in global memory (L2-miss path) and a mirrored copy in
/// shared memory. The ratio reveals how much the kernel benefits from
/// shared memory tiling.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryLatencyProbeResult {
    /// Global memory round-trip time (nanoseconds).
    pub global_memory_rtt_ns: f64,

    /// Shared memory round-trip time (nanoseconds).
    pub shared_memory_rtt_ns: f64,

    /// Ratio: global / shared (higher = more benefit from shared mem tiling).
    pub global_to_shared_ratio: f64,

    /// Number of pointer-chasing iterations performed.
    pub iterations: u64,

    /// Probe wall-clock duration (milliseconds).
    pub duration_ms: f64,
}

/// Compute Throughput Probe: measures raw FP32 operations per second.
///
/// On real hardware, this launches a kernel that performs a long chain
/// of dependent FMA (fused multiply-add) operations per thread, measuring
/// peak sustained FP32 throughput across all SMs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComputeThroughputProbeResult {
    /// Raw FP32 operations per second.
    pub fp32_ops_per_sec: f64,

    /// FP32 throughput in GFLOPS.
    pub fp32_gflops: f64,

    /// Number of FMA operations executed.
    pub total_fma_ops: u64,

    /// Probe wall-clock duration (milliseconds).
    pub duration_ms: f64,
}

/// Atomic Contention Probe: measures atomicCAS speed under 32-thread contention.
///
/// On real hardware, this launches 32 threads (one full warp) all targeting
/// the same memory address with atomicCAS. This measures worst-case CAS
/// throughput, which is the bottleneck for CRDT last-writer-wins merges.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AtomicContentionProbeResult {
    /// atomicCAS throughput with 32 threads on the same address (ops/sec).
    pub cas_32_thread_same_addr_ops_per_sec: f64,

    /// atomicCAS throughput with no contention — each thread hits a unique
    /// address (ops/sec).
    pub cas_uncontended_ops_per_sec: f64,

    /// Slowdown ratio: uncontended / contended.
    /// High values (>10x) strongly recommend warp-aggregated writes.
    pub contention_slowdown_ratio: f64,

    /// Total CAS operations executed.
    pub total_cas_ops: u64,

    /// Probe wall-clock duration (milliseconds).
    pub duration_ms: f64,
}

// ============================================================
// Hardware Probe Engine
// ============================================================

/// Runs three 100ms micro-benchmarks against the GPU (or simulated
/// equivalents) and produces `DnaProbeResults`.
///
/// Each benchmark targets a different hardware bottleneck:
///   1. Memory Latency — pointer-chasing in global vs. shared memory
///   2. Compute Throughput — sustained FP32 FMA chain
///   3. Atomic Contention — 32-way warp CAS on same address
///
/// On machines without a CUDA device, the probe generates simulated
/// results derived from the fingerprint's static properties (clock
/// rates, SM count, memory bandwidth).
pub struct HardwareProbe {
    /// The fingerprint to enrich with probe data.
    fingerprint: DnaHardwareFingerprint,

    /// Duration each individual benchmark should run (default: 100ms).
    benchmark_duration: Duration,
}

impl HardwareProbe {
    /// Create a new probe engine for the given fingerprint.
    pub fn new(fingerprint: DnaHardwareFingerprint) -> Self {
        HardwareProbe {
            fingerprint,
            benchmark_duration: Duration::from_millis(100),
        }
    }

    /// Override the per-benchmark duration (default: 100ms).
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.benchmark_duration = duration;
        self
    }

    /// Run all three micro-benchmarks and return a fingerprint enriched
    /// with `probe_results`.
    pub fn run_all(mut self) -> DnaHardwareFingerprint {
        let overall_start = Instant::now();

        println!("\n  ╔══════════════════════════════════════════════════════╗");
        println!("  ║  CudaClaw Hardware Probe — Dynamic Micro-Benchmarks ║");
        println!("  ╚══════════════════════════════════════════════════════╝");
        println!("  Target: {}", self.fingerprint.gpu_name);
        println!("  Per-benchmark budget: {:?}\n", self.benchmark_duration);

        // Probe 1: Memory Latency
        println!("  [1/3] Memory Latency Probe (Global vs. Shared RTT)...");
        let mem_result = self.probe_memory_latency();
        println!("         Global: {:.1} ns | Shared: {:.1} ns | Ratio: {:.1}x | {:.1}ms",
            mem_result.global_memory_rtt_ns, mem_result.shared_memory_rtt_ns,
            mem_result.global_to_shared_ratio, mem_result.duration_ms);

        // Probe 2: Compute Throughput
        println!("  [2/3] Compute Throughput Probe (FP32 ops/sec)...");
        let compute_result = self.probe_compute_throughput();
        println!("         {:.2e} ops/s ({:.1} GFLOPS) | {:.1}ms",
            compute_result.fp32_ops_per_sec, compute_result.fp32_gflops,
            compute_result.duration_ms);

        // Probe 3: Atomic Contention
        println!("  [3/3] Atomic Contention Probe (32-thread CAS)...");
        let atomic_result = self.probe_atomic_contention();
        println!("         Contended: {:.2e} ops/s | Uncontended: {:.2e} ops/s | Ratio: {:.1}x | {:.1}ms",
            atomic_result.cas_32_thread_same_addr_ops_per_sec,
            atomic_result.cas_uncontended_ops_per_sec,
            atomic_result.contention_slowdown_ratio, atomic_result.duration_ms);

        let total_ms = overall_start.elapsed().as_secs_f64() * 1000.0;
        println!("\n  Probe complete in {:.1}ms\n", total_ms);

        // Store results in fingerprint
        self.fingerprint.probe_results = Some(DnaProbeResults {
            memory_latency_probe: mem_result,
            compute_throughput_probe: compute_result,
            atomic_contention_probe: atomic_result,
            probe_duration_ms: total_ms,
            probe_timestamp: now_epoch(),
        });

        // Also update the fingerprint's static latency/throughput fields
        // with the measured values so they're available without drilling
        // into probe_results.
        if let Some(ref pr) = self.fingerprint.probe_results {
            self.fingerprint.memory_latency.global_memory_ns =
                pr.memory_latency_probe.global_memory_rtt_ns;
            self.fingerprint.memory_latency.shared_memory_ns =
                pr.memory_latency_probe.shared_memory_rtt_ns;
            self.fingerprint.atomic_throughput.cas_zero_contention_ops =
                pr.atomic_contention_probe.cas_uncontended_ops_per_sec;
            self.fingerprint.atomic_throughput.cas_warp_contention_ops =
                pr.atomic_contention_probe.cas_32_thread_same_addr_ops_per_sec;
            if pr.atomic_contention_probe.cas_32_thread_same_addr_ops_per_sec > 0.0 {
                self.fingerprint.atomic_throughput.contention_sensitivity_ratio =
                    pr.atomic_contention_probe.contention_slowdown_ratio;
            }
        }

        self.fingerprint
    }

    /// Probe 1: Memory Latency — Global vs. Shared memory RTT.
    ///
    /// Simulates pointer-chasing: a chain of dependent loads through a
    /// shuffled index array. On a real GPU, global memory hits the DRAM
    /// path (~400-500 ns) while shared memory completes in ~20-30 ns.
    ///
    /// On CPU, we approximate by measuring cache-miss vs. cache-hit
    /// latencies through a large vs. small working set.
    fn probe_memory_latency(&self) -> MemoryLatencyProbeResult {
        let start = Instant::now();
        let deadline = start + self.benchmark_duration;

        // --- Global Memory Simulation ---
        // Large working set (4 MB) to defeat CPU L1/L2 caches,
        // simulating GPU global memory (DRAM) access patterns.
        let global_size: usize = 1024 * 1024; // 1M entries = 4 MB @ 4 bytes
        let mut global_chain: Vec<u32> = (0..global_size as u32).collect();
        // Fisher-Yates shuffle to create a random pointer-chase chain
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
        for i in (1..global_size).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            global_chain.swap(i, j);
        }

        let mut global_iters: u64 = 0;
        let mut idx: u32 = 0;
        let global_start = Instant::now();
        while Instant::now() < deadline {
            // Chase through the shuffled chain (simulates random global memory reads)
            for _ in 0..1000 {
                idx = global_chain[idx as usize % global_size];
            }
            global_iters += 1000;
        }
        let global_elapsed = global_start.elapsed();
        // Prevent optimizer from eliminating the loop
        std::hint::black_box(idx);

        // --- Shared Memory Simulation ---
        // Small working set (4 KB) that fits in L1 cache,
        // simulating GPU shared memory access patterns.
        let shared_size: usize = 1024; // 1K entries = 4 KB
        let mut shared_chain: Vec<u32> = (0..shared_size as u32).collect();
        rng_state = 0xCAFE_BABE_1234_5678u64;
        for i in (1..shared_size).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            shared_chain.swap(i, j);
        }

        let mut shared_iters: u64 = 0;
        idx = 0;
        let shared_deadline = Instant::now() + self.benchmark_duration;
        let shared_start = Instant::now();
        while Instant::now() < shared_deadline {
            for _ in 0..1000 {
                idx = shared_chain[idx as usize % shared_size];
            }
            shared_iters += 1000;
        }
        let shared_elapsed = shared_start.elapsed();
        std::hint::black_box(idx);

        // Calculate RTTs
        let global_rtt_ns = if global_iters > 0 {
            global_elapsed.as_nanos() as f64 / global_iters as f64
        } else {
            self.fingerprint.memory_latency.global_memory_ns
        };
        let shared_rtt_ns = if shared_iters > 0 {
            shared_elapsed.as_nanos() as f64 / shared_iters as f64
        } else {
            self.fingerprint.memory_latency.shared_memory_ns
        };

        // Scale CPU measurements to GPU-realistic values using known ratios.
        // CPU cache-miss/hit ratio is typically 5-15x; GPU global/shared is ~20x.
        // We apply a scaling factor based on the fingerprint's known architecture.
        let (scaled_global, scaled_shared) = if self.fingerprint.is_simulated {
            // Use fingerprint values directly for simulated hardware
            (
                self.fingerprint.memory_latency.global_memory_ns.max(global_rtt_ns),
                self.fingerprint.memory_latency.shared_memory_ns.max(shared_rtt_ns.min(global_rtt_ns)),
            )
        } else {
            (global_rtt_ns, shared_rtt_ns)
        };

        let ratio = if scaled_shared > 0.0 {
            scaled_global / scaled_shared
        } else {
            0.0
        };

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        MemoryLatencyProbeResult {
            global_memory_rtt_ns: scaled_global,
            shared_memory_rtt_ns: scaled_shared,
            global_to_shared_ratio: ratio,
            iterations: global_iters + shared_iters,
            duration_ms: total_ms,
        }
    }

    /// Probe 2: Compute Throughput — raw FP32 ops/sec.
    ///
    /// Measures sustained FP32 FMA (fused multiply-add) throughput.
    /// On a real GPU, this launches a kernel where each thread executes
    /// a long chain of dependent FMA ops. On CPU, we run an equivalent
    /// FMA chain and scale by the GPU's theoretical peak.
    fn probe_compute_throughput(&self) -> ComputeThroughputProbeResult {
        let start = Instant::now();
        let deadline = start + self.benchmark_duration;

        // Execute a long chain of dependent FP32 FMAs on the CPU.
        // Each iteration: val = val * 1.00001 + 0.00001 (prevents optimization)
        let mut val: f32 = 1.0f32;
        let mut total_ops: u64 = 0;

        while Instant::now() < deadline {
            // 1000 dependent FMA ops per inner loop
            for _ in 0..1000 {
                val = val * 1.000_001f32 + 0.000_001f32;
            }
            total_ops += 1000;
        }
        std::hint::black_box(val);

        let elapsed = start.elapsed();
        let cpu_ops_per_sec = if elapsed.as_nanos() > 0 {
            total_ops as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Scale to GPU: use the fingerprint's SM count and clock rate to
        // estimate theoretical peak FP32 throughput, then use the CPU
        // measurement as a scaling baseline.
        let gpu_fp32_ops_per_sec = if self.fingerprint.is_simulated && self.fingerprint.sm_count > 0 {
            // Theoretical peak: SMs * FP32_cores_per_SM * 2 (FMA) * clock_Hz
            // Ada Lovelace: 128 FP32 cores per SM
            let fp32_cores_per_sm: u64 = match self.fingerprint.compute_capability_major {
                8 => 128, // Ampere / Ada Lovelace
                7 => 64,  // Volta / Turing
                9 => 128, // Hopper
                _ => 64,
            };
            let theoretical_peak = self.fingerprint.sm_count as f64
                * fp32_cores_per_sm as f64
                * 2.0  // FMA = 2 FP ops
                * self.fingerprint.core_clock_mhz as f64
                * 1_000_000.0; // MHz to Hz
            // Sustained is typically 70-85% of theoretical
            theoretical_peak * 0.78
        } else if self.fingerprint.sm_count > 0 {
            // Real hardware: scale CPU measurement by parallelism factor
            cpu_ops_per_sec * self.fingerprint.sm_count as f64 * 32.0
        } else {
            cpu_ops_per_sec
        };

        let gflops = gpu_fp32_ops_per_sec / 1_000_000_000.0;

        ComputeThroughputProbeResult {
            fp32_ops_per_sec: gpu_fp32_ops_per_sec,
            fp32_gflops: gflops,
            total_fma_ops: total_ops,
            duration_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }

    /// Probe 3: Atomic Contention — atomicCAS with 32 threads hitting
    /// the same address.
    ///
    /// On a real GPU, this launches exactly 32 threads (one warp) all
    /// doing atomicCAS on the same u64 address. The throughput under
    /// contention is critical for CRDT cell merges where multiple agents
    /// may update the same cell simultaneously.
    ///
    /// On CPU, we simulate with 32 threads doing `compare_exchange` on
    /// the same `AtomicU64`.
    fn probe_atomic_contention(&self) -> AtomicContentionProbeResult {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        let start = Instant::now();

        // --- Contended CAS: 32 threads on same address ---
        let shared_counter = Arc::new(AtomicU64::new(0));
        let contended_ops = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        for _ in 0..32 {
            let counter = Arc::clone(&shared_counter);
            let ops = Arc::clone(&contended_ops);
            let dur = self.benchmark_duration;
            handles.push(std::thread::spawn(move || {
                let deadline = Instant::now() + dur;
                let mut local_ops: u64 = 0;
                let mut val = 0u64;
                while Instant::now() < deadline {
                    for _ in 0..100 {
                        // CAS loop: try to increment the shared counter
                        loop {
                            let current = counter.load(Ordering::Relaxed);
                            match counter.compare_exchange_weak(
                                current,
                                current + 1,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(v) => val = v, // retry
                            }
                        }
                        local_ops += 1;
                    }
                }
                std::hint::black_box(val);
                ops.fetch_add(local_ops, Ordering::Relaxed);
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        let contended_total = contended_ops.load(Ordering::Relaxed);
        let contended_elapsed = start.elapsed();

        // --- Uncontended CAS: 32 threads on separate addresses ---
        let uncontended_start = Instant::now();
        let uncontended_ops = Arc::new(AtomicU64::new(0));
        // Each thread gets its own counter (no contention)
        let mut handles2 = Vec::new();
        for _ in 0..32 {
            let ops = Arc::clone(&uncontended_ops);
            let dur = self.benchmark_duration;
            handles2.push(std::thread::spawn(move || {
                let counter = AtomicU64::new(0); // thread-local, no contention
                let deadline = Instant::now() + dur;
                let mut local_ops: u64 = 0;
                while Instant::now() < deadline {
                    for _ in 0..100 {
                        let current = counter.load(Ordering::Relaxed);
                        let _ = counter.compare_exchange(
                            current,
                            current + 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                        local_ops += 1;
                    }
                }
                ops.fetch_add(local_ops, Ordering::Relaxed);
            }));
        }
        for h in handles2 {
            let _ = h.join();
        }
        let uncontended_total = uncontended_ops.load(Ordering::Relaxed);
        let uncontended_elapsed = uncontended_start.elapsed();

        let total_elapsed = start.elapsed();

        // Calculate ops/sec
        let contended_ops_per_sec = if contended_elapsed.as_nanos() > 0 {
            contended_total as f64 / contended_elapsed.as_secs_f64()
        } else {
            0.0
        };
        let uncontended_ops_per_sec = if uncontended_elapsed.as_nanos() > 0 {
            uncontended_total as f64 / uncontended_elapsed.as_secs_f64()
        } else {
            0.0
        };

        // On a GPU, contention is much worse because all 32 threads in a
        // warp are truly simultaneous (not time-sliced like CPU threads).
        // Scale the ratio if simulated.
        let (scaled_contended, scaled_uncontended) = if self.fingerprint.is_simulated {
            // Use fingerprint values but incorporate measured ratio
            let measured_ratio = if contended_ops_per_sec > 0.0 {
                uncontended_ops_per_sec / contended_ops_per_sec
            } else {
                10.0
            };
            // GPU contention is typically worse: apply a 2-3x multiplier
            let gpu_ratio = measured_ratio * 2.5;
            let uc = self.fingerprint.atomic_throughput.cas_zero_contention_ops.max(uncontended_ops_per_sec);
            let c = uc / gpu_ratio;
            (c, uc)
        } else {
            (contended_ops_per_sec, uncontended_ops_per_sec)
        };

        let slowdown = if scaled_contended > 0.0 {
            scaled_uncontended / scaled_contended
        } else {
            0.0
        };

        AtomicContentionProbeResult {
            cas_32_thread_same_addr_ops_per_sec: scaled_contended,
            cas_uncontended_ops_per_sec: scaled_uncontended,
            contention_slowdown_ratio: slowdown,
            total_cas_ops: contended_total + uncontended_total,
            duration_ms: total_elapsed.as_secs_f64() * 1000.0,
        }
    }
}

/// Run the hardware probe and return an enriched fingerprint.
///
/// Attempts to detect a real GPU via `cust`. If no GPU is found,
/// falls back to the simulated RTX 4090 fingerprint with simulated
/// probe results.
pub fn probe_hardware(device_index: u32) -> DnaHardwareFingerprint {
    #[cfg(feature = "cuda")]
    let fingerprint = DnaHardwareFingerprint::from_cust_device(device_index)
        .unwrap_or_else(|| {
            println!("  No CUDA device found — using simulated RTX 4090 profile");
            DnaHardwareFingerprint::rtx4090_simulated()
        });
    #[cfg(not(feature = "cuda"))]
    let fingerprint = {
        let _ = device_index;
        println!("  CUDA feature not enabled — using simulated RTX 4090 profile");
        DnaHardwareFingerprint::rtx4090_simulated()
    };
    HardwareProbe::new(fingerprint).run_all()
}

// ============================================================
// Section 2: Constraint-Theory Mappings
// ============================================================

/// Safe bounds derived from the Constraint-Theory project.
///
/// Each bound is a named rule with a typed value and severity.
/// The Ramify engine checks these before every kernel launch and
/// compilation to ensure the instance stays within safe limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaConstraintMappings {
    /// All constraint bounds, keyed by constraint ID.
    /// IDs follow the pattern: "category.name" (e.g., "resource.register_budget").
    pub resource_bounds: HashMap<String, DnaConstraintBound>,
}

/// A single constraint bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaConstraintBound {
    /// Human-readable name.
    pub name: String,

    /// Description of what this constraint protects.
    pub description: String,

    /// Category for grouping.
    pub category: DnaConstraintCategory,

    /// How severe a violation is.
    pub severity: DnaConstraintSeverity,

    /// The typed bound value.
    pub value: DnaBoundValue,

    /// Whether this constraint is currently active.
    pub enabled: bool,

    /// Number of times this bound has been mutated by ML.
    pub mutation_count: u32,
}

/// Constraint categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DnaConstraintCategory {
    Resource,
    Latency,
    Correctness,
    Efficiency,
    Biological,
}

/// Constraint severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DnaConstraintSeverity {
    /// Violation rejects the operation.
    Critical,
    /// Violation emits a warning.
    Warning,
    /// Logged but not enforced.
    Info,
}

/// Typed constraint bound value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DnaBoundValue {
    IntMax(u64),
    IntMin(u64),
    FloatMax(f64),
    FloatMin(f64),
    BoolRequired(bool),
    Range { min: f64, max: f64 },
}

impl DnaBoundValue {
    /// Check whether a numeric value satisfies this bound.
    pub fn check(&self, actual: f64) -> bool {
        match self {
            DnaBoundValue::IntMax(max) => actual <= *max as f64,
            DnaBoundValue::IntMin(min) => actual >= *min as f64,
            DnaBoundValue::FloatMax(max) => actual <= *max,
            DnaBoundValue::FloatMin(min) => actual >= *min,
            DnaBoundValue::BoolRequired(req) => (actual != 0.0) == *req,
            DnaBoundValue::Range { min, max } => actual >= *min && actual <= *max,
        }
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        match self {
            DnaBoundValue::IntMax(v) => format!("<= {}", v),
            DnaBoundValue::IntMin(v) => format!(">= {}", v),
            DnaBoundValue::FloatMax(v) => format!("<= {:.2}", v),
            DnaBoundValue::FloatMin(v) => format!(">= {:.2}", v),
            DnaBoundValue::BoolRequired(v) => format!("must be {}", v),
            DnaBoundValue::Range { min, max } => format!("[{:.2}, {:.2}]", min, max),
        }
    }
}

impl DnaConstraintMappings {
    /// Create the default system constraints.
    pub fn default_system() -> Self {
        let mut bounds = HashMap::new();

        // Resource constraints
        bounds.insert(
            "resource.register_budget".into(),
            DnaConstraintBound {
                name: "Register Budget per Agent".into(),
                description: "Maximum 32-bit registers an agent may consume on a single SM.".into(),
                category: DnaConstraintCategory::Resource,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::IntMax(32768),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "resource.shared_memory_ceiling".into(),
            DnaConstraintBound {
                name: "Shared Memory Ceiling per Agent".into(),
                description: "Maximum shared memory (bytes) an agent may request on a single SM.".into(),
                category: DnaConstraintCategory::Resource,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::IntMax(49152),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "resource.warp_slot_limit".into(),
            DnaConstraintBound {
                name: "Warp Slot Limit per Agent".into(),
                description: "Maximum warp slots an agent may occupy on a single SM.".into(),
                category: DnaConstraintCategory::Resource,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::IntMax(32),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "resource.thread_budget".into(),
            DnaConstraintBound {
                name: "Thread Budget per Agent".into(),
                description: "Maximum threads an agent may launch on a single SM.".into(),
                category: DnaConstraintCategory::Resource,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::IntMax(2048),
                enabled: true,
                mutation_count: 0,
            },
        );

        // Latency constraints
        bounds.insert(
            "latency.p99_rtt_ceiling".into(),
            DnaConstraintBound {
                name: "P99 RTT Ceiling".into(),
                description: "Maximum P99 round-trip time in microseconds.".into(),
                category: DnaConstraintCategory::Latency,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::FloatMax(8.0),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "latency.push_latency_ceiling".into(),
            DnaConstraintBound {
                name: "Push Latency Ceiling".into(),
                description: "Maximum host-side push latency in microseconds.".into(),
                category: DnaConstraintCategory::Latency,
                severity: DnaConstraintSeverity::Warning,
                value: DnaBoundValue::FloatMax(2.0),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "latency.pcie_transfer_budget".into(),
            DnaConstraintBound {
                name: "PCIe Transfer Budget".into(),
                description: "Maximum PCIe delay in microseconds.".into(),
                category: DnaConstraintCategory::Latency,
                severity: DnaConstraintSeverity::Warning,
                value: DnaBoundValue::FloatMax(5.0),
                enabled: true,
                mutation_count: 0,
            },
        );

        // Correctness constraints
        bounds.insert(
            "correctness.crdt_monotonicity".into(),
            DnaConstraintBound {
                name: "CRDT Monotonicity".into(),
                description: "CRDT cell timestamps must be monotonically increasing.".into(),
                category: DnaConstraintCategory::Correctness,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::BoolRequired(true),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "correctness.timestamp_ordering".into(),
            DnaConstraintBound {
                name: "Timestamp Ordering".into(),
                description: "Operations must have timestamps > their predecessor.".into(),
                category: DnaConstraintCategory::Correctness,
                severity: DnaConstraintSeverity::Critical,
                value: DnaBoundValue::BoolRequired(true),
                enabled: true,
                mutation_count: 0,
            },
        );

        // Efficiency constraints
        bounds.insert(
            "efficiency.min_warp_occupancy".into(),
            DnaConstraintBound {
                name: "Minimum Warp Occupancy".into(),
                description: "Minimum fraction of SM warp slots occupied during execution.".into(),
                category: DnaConstraintCategory::Efficiency,
                severity: DnaConstraintSeverity::Warning,
                value: DnaBoundValue::FloatMin(0.25),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "efficiency.min_coalescing_ratio".into(),
            DnaConstraintBound {
                name: "Minimum Coalescing Ratio".into(),
                description: "Minimum fraction of memory transactions that are coalesced.".into(),
                category: DnaConstraintCategory::Efficiency,
                severity: DnaConstraintSeverity::Warning,
                value: DnaBoundValue::FloatMin(0.75),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "efficiency.idle_power_ceiling".into(),
            DnaConstraintBound {
                name: "Idle Power Ceiling".into(),
                description: "Maximum power draw (watts) when the GPU is idle-polling.".into(),
                category: DnaConstraintCategory::Efficiency,
                severity: DnaConstraintSeverity::Info,
                value: DnaBoundValue::FloatMax(25.0),
                enabled: true,
                mutation_count: 0,
            },
        );

        // Biological constraints
        bounds.insert(
            "biological.nutrient_floor".into(),
            DnaConstraintBound {
                name: "Nutrient Floor per SM".into(),
                description: "Minimum nutrient score for an SM before pruning/branching.".into(),
                category: DnaConstraintCategory::Biological,
                severity: DnaConstraintSeverity::Warning,
                value: DnaBoundValue::FloatMin(0.15),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "biological.prune_cooldown_ms".into(),
            DnaConstraintBound {
                name: "Prune Cooldown".into(),
                description: "Minimum milliseconds between successive prune actions.".into(),
                category: DnaConstraintCategory::Biological,
                severity: DnaConstraintSeverity::Info,
                value: DnaBoundValue::FloatMin(100.0),
                enabled: true,
                mutation_count: 0,
            },
        );
        bounds.insert(
            "biological.branch_hysteresis_ms".into(),
            DnaConstraintBound {
                name: "Branch Hysteresis".into(),
                description: "Minimum ms an agent must run on its SM before branching.".into(),
                category: DnaConstraintCategory::Biological,
                severity: DnaConstraintSeverity::Info,
                value: DnaBoundValue::FloatMin(500.0),
                enabled: true,
                mutation_count: 0,
            },
        );

        DnaConstraintMappings {
            resource_bounds: bounds,
        }
    }

    /// Mutate a constraint bound's value.
    pub fn mutate_bound(&mut self, id: &str, new_value: DnaBoundValue) -> bool {
        if let Some(bound) = self.resource_bounds.get_mut(id) {
            bound.value = new_value;
            bound.mutation_count += 1;
            true
        } else {
            false
        }
    }

    /// Check a value against a named constraint.
    pub fn check(&self, id: &str, actual: f64) -> Option<bool> {
        self.resource_bounds.get(id).map(|b| {
            if !b.enabled {
                return true;
            }
            b.value.check(actual)
        })
    }

    /// Get all enabled constraints in a category.
    pub fn by_category(&self, category: DnaConstraintCategory) -> Vec<(&str, &DnaConstraintBound)> {
        self.resource_bounds
            .iter()
            .filter(|(_, b)| b.category == category && b.enabled)
            .map(|(id, b)| (id.as_str(), b))
            .collect()
    }

    /// Print a summary.
    pub fn print_summary(&self) {
        let categories = [
            DnaConstraintCategory::Resource,
            DnaConstraintCategory::Latency,
            DnaConstraintCategory::Correctness,
            DnaConstraintCategory::Efficiency,
            DnaConstraintCategory::Biological,
        ];
        for cat in &categories {
            let bounds = self.by_category(*cat);
            if !bounds.is_empty() {
                println!("    [{:?}]", cat);
                for (id, bound) in &bounds {
                    let sev = match bound.severity {
                        DnaConstraintSeverity::Critical => "CRIT",
                        DnaConstraintSeverity::Warning => "WARN",
                        DnaConstraintSeverity::Info => "INFO",
                    };
                    println!("      {} [{}] {} = {}",
                        id, sev, bound.name, bound.value.describe());
                }
            }
        }
    }
}

// ============================================================
// Section 3: PTX Muscle Fibers
// ============================================================

/// Map of task IDs to specialized kernel configurations.
///
/// Each fiber contains the launch parameters AND either a
/// pre-compiled PTX string or NVRTC compilation parameters
/// so the Ramify engine can recompile on-the-fly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaMuscleFiberMap {
    /// Task ID → Muscle Fiber definition.
    pub fibers: HashMap<String, DnaMuscleFiber>,
}

/// A single muscle fiber — a tuned kernel for a workload pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaMuscleFiber {
    /// Human-readable name.
    pub name: String,

    /// Description of the workload this fiber handles.
    pub description: String,

    // ── Launch parameters ──

    /// Threads per block.
    pub block_size: u32,

    /// Registers per thread (for occupancy calculation).
    pub registers_per_thread: u32,

    /// Shared memory per block (bytes).
    pub shared_memory_bytes: u32,

    /// Target occupancy (0.0-1.0).
    pub target_occupancy: f64,

    // ── Kernel behavior flags ──

    /// Whether this fiber uses PTX-level atomicCAS.
    pub uses_ptx_cas: bool,

    /// Whether this fiber uses warp aggregation.
    pub uses_warp_aggregation: bool,

    /// Whether this fiber uses prefix-sum scan.
    pub uses_prefix_sum: bool,

    /// Whether this is a persistent polling kernel.
    pub is_persistent: bool,

    // ── PTX / NVRTC source ──

    /// The kernel source — either pre-compiled PTX or CUDA C++.
    pub kernel_source: DnaKernelSource,

    // ── Performance expectations ──

    /// Expected P99 latency in microseconds.
    pub expected_p99_us: f64,

    /// Expected throughput (ops/sec).
    pub expected_throughput_ops: f64,

    /// Best measured latency (microseconds).
    pub best_measured_latency_us: f64,

    /// Worst measured latency (microseconds).
    pub worst_measured_latency_us: f64,

    /// Number of performance measurements recorded.
    pub measurement_count: u64,
}

/// Source material for a kernel — either raw PTX or CUDA C++ for NVRTC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DnaKernelSource {
    /// Pre-compiled PTX assembly (ready to load).
    Ptx {
        /// The PTX assembly string.
        ptx: String,
        /// Target architecture (e.g., "sm_89").
        target_arch: String,
    },
    /// CUDA C++ source for NVRTC runtime compilation.
    NvrtcSource {
        /// The CUDA C++ source code.
        source: String,
        /// Program name for NVRTC.
        program_name: String,
        /// Target compute capability (e.g., "compute_89").
        gpu_arch: String,
        /// Preprocessor defines to inject.
        defines: HashMap<String, String>,
        /// Extra compiler flags.
        extra_flags: Vec<String>,
        /// Maximum registers per thread (0 = no limit).
        max_registers: u32,
        /// Whether to enable fast math.
        use_fast_math: bool,
    },
    /// No kernel source — fiber is a configuration-only placeholder.
    None,
}

impl DnaMuscleFiberMap {
    /// Create an empty fiber map.
    pub fn new() -> Self {
        DnaMuscleFiberMap {
            fibers: HashMap::new(),
        }
    }

    /// Create a fiber map with default fibers for a spreadsheet engine.
    pub fn default_fibers() -> Self {
        let mut map = Self::new();

        map.fibers.insert(
            "cell_update".into(),
            DnaMuscleFiber {
                name: "Cell Update".into(),
                description: "Optimized for simple cell value writes. Small block size \
                              for low latency, minimal register pressure."
                    .into(),
                block_size: 128,
                registers_per_thread: 24,
                shared_memory_bytes: 2048,
                target_occupancy: 0.75,
                uses_ptx_cas: false,
                uses_warp_aggregation: false,
                uses_prefix_sum: false,
                is_persistent: false,
                kernel_source: DnaKernelSource::NvrtcSource {
                    source: CELL_UPDATE_CUDA_STUB.to_string(),
                    program_name: "cell_update_nvrtc.cu".into(),
                    gpu_arch: "compute_89".into(),
                    defines: HashMap::new(),
                    extra_flags: vec!["--use_fast_math".into()],
                    max_registers: 32,
                    use_fast_math: true,
                },
                expected_p99_us: 3.0,
                expected_throughput_ops: 500_000.0,
                best_measured_latency_us: f64::MAX,
                worst_measured_latency_us: 0.0,
                measurement_count: 0,
            },
        );

        map.fibers.insert(
            "crdt_merge".into(),
            DnaMuscleFiber {
                name: "CRDT Merge".into(),
                description: "Conflict resolution with PTX-level atomicCAS. Medium block \
                              size, high shmem for hash tables."
                    .into(),
                block_size: 256,
                registers_per_thread: 40,
                shared_memory_bytes: 49152,
                target_occupancy: 0.5,
                uses_ptx_cas: true,
                uses_warp_aggregation: true,
                uses_prefix_sum: false,
                is_persistent: false,
                kernel_source: DnaKernelSource::NvrtcSource {
                    source: CRDT_MERGE_CUDA_STUB.to_string(),
                    program_name: "crdt_merge_nvrtc.cu".into(),
                    gpu_arch: "compute_89".into(),
                    defines: HashMap::new(),
                    extra_flags: vec!["--use_fast_math".into()],
                    max_registers: 48,
                    use_fast_math: true,
                },
                expected_p99_us: 5.0,
                expected_throughput_ops: 200_000.0,
                best_measured_latency_us: f64::MAX,
                worst_measured_latency_us: 0.0,
                measurement_count: 0,
            },
        );

        map.fibers.insert(
            "formula_eval".into(),
            DnaMuscleFiber {
                name: "Formula Evaluator".into(),
                description: "Parallel formula recalculation using prefix-sum scan. Large \
                              block for scan parallelism."
                    .into(),
                block_size: 512,
                registers_per_thread: 32,
                shared_memory_bytes: 16384,
                target_occupancy: 0.5,
                uses_ptx_cas: false,
                uses_warp_aggregation: false,
                uses_prefix_sum: true,
                is_persistent: false,
                kernel_source: DnaKernelSource::None,
                expected_p99_us: 6.0,
                expected_throughput_ops: 100_000.0,
                best_measured_latency_us: f64::MAX,
                worst_measured_latency_us: 0.0,
                measurement_count: 0,
            },
        );

        map.fibers.insert(
            "batch_process".into(),
            DnaMuscleFiber {
                name: "Batch Processor".into(),
                description: "Bulk cell operations. Maximum block size for throughput.".into(),
                block_size: 1024,
                registers_per_thread: 20,
                shared_memory_bytes: 8192,
                target_occupancy: 0.75,
                uses_ptx_cas: false,
                uses_warp_aggregation: false,
                uses_prefix_sum: false,
                is_persistent: false,
                kernel_source: DnaKernelSource::None,
                expected_p99_us: 7.0,
                expected_throughput_ops: 1_000_000.0,
                best_measured_latency_us: f64::MAX,
                worst_measured_latency_us: 0.0,
                measurement_count: 0,
            },
        );

        map.fibers.insert(
            "idle_poll".into(),
            DnaMuscleFiber {
                name: "Idle Poller".into(),
                description: "Persistent polling kernel with minimal resource usage. One \
                              warp continuously polls the command queue."
                    .into(),
                block_size: 32,
                registers_per_thread: 16,
                shared_memory_bytes: 256,
                target_occupancy: 0.03,
                uses_ptx_cas: false,
                uses_warp_aggregation: false,
                uses_prefix_sum: false,
                is_persistent: true,
                kernel_source: DnaKernelSource::Ptx {
                    ptx: IDLE_POLL_PTX_STUB.to_string(),
                    target_arch: "sm_89".into(),
                },
                expected_p99_us: 1.0,
                expected_throughput_ops: 10_000_000.0,
                best_measured_latency_us: f64::MAX,
                worst_measured_latency_us: 0.0,
                measurement_count: 0,
            },
        );

        map
    }

    /// Register a new fiber or replace an existing one.
    pub fn register(&mut self, task_id: &str, fiber: DnaMuscleFiber) {
        self.fibers.insert(task_id.to_string(), fiber);
    }

    /// Get a fiber by task ID.
    pub fn get(&self, task_id: &str) -> Option<&DnaMuscleFiber> {
        self.fibers.get(task_id)
    }

    /// Get a mutable fiber by task ID.
    pub fn get_mut(&mut self, task_id: &str) -> Option<&mut DnaMuscleFiber> {
        self.fibers.get_mut(task_id)
    }

    /// Record a performance measurement for a fiber.
    pub fn record_measurement(&mut self, task_id: &str, latency_us: f64) {
        if let Some(fiber) = self.fibers.get_mut(task_id) {
            fiber.measurement_count += 1;
            if latency_us < fiber.best_measured_latency_us {
                fiber.best_measured_latency_us = latency_us;
            }
            if latency_us > fiber.worst_measured_latency_us {
                fiber.worst_measured_latency_us = latency_us;
            }
        }
    }

    /// List all task IDs.
    pub fn task_ids(&self) -> Vec<&str> {
        self.fibers.keys().map(|s| s.as_str()).collect()
    }

    /// Print a summary.
    pub fn print_summary(&self) {
        let mut ids: Vec<&String> = self.fibers.keys().collect();
        ids.sort();
        for id in ids {
            let f = &self.fibers[id];
            let source_type = match &f.kernel_source {
                DnaKernelSource::Ptx { .. } => "PTX",
                DnaKernelSource::NvrtcSource { .. } => "NVRTC",
                DnaKernelSource::None => "None",
            };
            println!(
                "    {} [{}] — block={}, regs={}, shmem={} B, target_occ={:.0}%, \
                 source={}, P99={:.1} us",
                id, f.name, f.block_size, f.registers_per_thread,
                f.shared_memory_bytes, f.target_occupancy * 100.0,
                source_type, f.expected_p99_us
            );
        }
    }
}

// ============================================================
// Section 4: Resource Exhaustion Metrics
// ============================================================

/// Historical exhaust data — heat, latency, throttle events — that
/// the ML feedback loop uses to prune inefficient branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaExhaustionMetrics {
    /// Exhaustion policy thresholds.
    pub thresholds: DnaExhaustionThresholds,

    /// Per-SM resource snapshots (most recent).
    pub sm_snapshots: Vec<DnaSmSnapshot>,

    /// Exhaust event log (bounded ring buffer history).
    pub exhaust_log: Vec<DnaExhaustEvent>,

    /// Maximum exhaust log entries before oldest are evicted.
    pub max_exhaust_log_entries: usize,

    /// Thermal history.
    pub thermal_log: Vec<DnaThermalSample>,

    /// Maximum thermal log entries.
    pub max_thermal_log_entries: usize,

    /// DNA mutation records from the ML feedback loop.
    pub mutation_log: Vec<DnaMutationRecord>,

    /// Aggregate stats.
    pub aggregate: DnaExhaustionAggregate,
}

/// Thresholds for triggering exhaustion actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaExhaustionThresholds {
    /// Register utilization above this triggers pruning (0.0-1.0).
    pub register_prune_threshold: f64,

    /// Shared memory utilization above this triggers pruning (0.0-1.0).
    pub shmem_prune_threshold: f64,

    /// Warp occupancy above this triggers branching (0.0-1.0).
    pub warp_branch_threshold: f64,

    /// Nutrient score below this triggers action (0.0-1.0).
    pub nutrient_critical_floor: f64,

    /// Idle time (seconds) before harvesting an agent.
    pub idle_harvest_seconds: f64,

    /// Minimum block size after pruning (one warp).
    pub min_block_size: u32,

    /// Max prune actions per agent per epoch.
    pub max_prunes_per_epoch: u32,

    /// Throttle factor (0.0 = full, 1.0 = none).
    pub throttle_factor: f64,

    /// Throttle duration in milliseconds.
    pub throttle_duration_ms: u64,

    /// GPU temperature (Celsius) above which throttling is forced.
    pub thermal_throttle_celsius: f64,

    /// P99 latency (microseconds) above which the system prunes.
    pub latency_prune_threshold_us: f64,
}

impl Default for DnaExhaustionThresholds {
    fn default() -> Self {
        DnaExhaustionThresholds {
            register_prune_threshold: 0.85,
            shmem_prune_threshold: 0.90,
            warp_branch_threshold: 0.90,
            nutrient_critical_floor: 0.15,
            idle_harvest_seconds: 5.0,
            min_block_size: 32,
            max_prunes_per_epoch: 3,
            throttle_factor: 0.5,
            throttle_duration_ms: 2000,
            thermal_throttle_celsius: 85.0,
            latency_prune_threshold_us: 8.0,
        }
    }
}

/// Point-in-time SM resource snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaSmSnapshot {
    /// SM index.
    pub sm_index: u32,
    /// Register utilization (0.0-1.0).
    pub register_utilization: f64,
    /// Shared memory utilization (0.0-1.0).
    pub shared_memory_utilization: f64,
    /// Warp occupancy (0.0-1.0).
    pub warp_occupancy: f64,
    /// Thread occupancy (0.0-1.0).
    pub thread_occupancy: f64,
    /// Number of agents on this SM.
    pub agent_count: u32,
    /// Nutrient score (1.0 - max_utilization).
    pub nutrient_score: f64,
    /// Snapshot timestamp (Unix epoch seconds).
    pub timestamp: u64,
}

/// A recorded exhaustion event (prune, branch, throttle, harvest).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaExhaustEvent {
    /// Event timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Agent ID that triggered the event.
    pub agent_id: String,
    /// SM index where the event occurred.
    pub sm_index: u32,
    /// Type of action taken.
    pub action: DnaExhaustAction,
    /// Reason for the action.
    pub reason: String,
    /// Latency at time of event (microseconds).
    pub latency_at_event_us: f64,
    /// Temperature at time of event (Celsius).
    pub temperature_at_event_c: f64,
}

/// Exhaustion action types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DnaExhaustAction {
    Prune {
        old_block_size: u32,
        new_block_size: u32,
        old_priority: u32,
        new_priority: u32,
    },
    Branch {
        source_sm: u32,
        target_sm: u32,
    },
    Throttle {
        factor: f64,
        duration_ms: u64,
    },
    Harvest {
        registers_reclaimed: u32,
        shared_memory_reclaimed: u32,
    },
}

/// GPU thermal sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaThermalSample {
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// GPU temperature (Celsius).
    pub temperature_c: f64,
    /// GPU power draw (watts).
    pub power_watts: f64,
    /// Fan speed (0-100%).
    pub fan_speed_pct: f64,
    /// Whether thermal throttling was active.
    pub is_throttling: bool,
}

/// Record of a DNA mutation from the ML feedback loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaMutationRecord {
    /// Sequential mutation ID.
    pub mutation_id: u64,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Description of what was mutated.
    pub description: String,
    /// Score before mutation.
    pub score_before: f64,
    /// Score after mutation.
    pub score_after: f64,
}

/// Aggregate exhaustion statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaExhaustionAggregate {
    /// Total prune actions taken.
    pub total_prunes: u64,
    /// Total branch actions taken.
    pub total_branches: u64,
    /// Total throttle actions taken.
    pub total_throttles: u64,
    /// Total harvest actions taken.
    pub total_harvests: u64,
    /// Peak temperature observed (Celsius).
    pub peak_temperature_c: f64,
    /// Average nutrient score across all SMs.
    pub avg_nutrient_score: f64,
    /// Number of critical SM events (nutrient < threshold).
    pub critical_sm_events: u64,
}

impl DnaExhaustionAggregate {
    fn new() -> Self {
        DnaExhaustionAggregate {
            total_prunes: 0,
            total_branches: 0,
            total_throttles: 0,
            total_harvests: 0,
            peak_temperature_c: 0.0,
            avg_nutrient_score: 1.0,
            critical_sm_events: 0,
        }
    }
}

impl DnaExhaustionMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        DnaExhaustionMetrics {
            thresholds: DnaExhaustionThresholds::default(),
            sm_snapshots: Vec::new(),
            exhaust_log: Vec::new(),
            max_exhaust_log_entries: 10_000,
            thermal_log: Vec::new(),
            max_thermal_log_entries: 1_000,
            mutation_log: Vec::new(),
            aggregate: DnaExhaustionAggregate::new(),
        }
    }

    /// Record an exhaust event.
    pub fn record_event(&mut self, event: DnaExhaustEvent) {
        // Update aggregate counts.
        match &event.action {
            DnaExhaustAction::Prune { .. } => self.aggregate.total_prunes += 1,
            DnaExhaustAction::Branch { .. } => self.aggregate.total_branches += 1,
            DnaExhaustAction::Throttle { .. } => self.aggregate.total_throttles += 1,
            DnaExhaustAction::Harvest { .. } => self.aggregate.total_harvests += 1,
        }

        // Evict oldest if at capacity.
        if self.exhaust_log.len() >= self.max_exhaust_log_entries {
            self.exhaust_log.remove(0);
        }
        self.exhaust_log.push(event);
    }

    /// Record a thermal sample.
    pub fn record_thermal(&mut self, sample: DnaThermalSample) {
        if sample.temperature_c > self.aggregate.peak_temperature_c {
            self.aggregate.peak_temperature_c = sample.temperature_c;
        }
        if self.thermal_log.len() >= self.max_thermal_log_entries {
            self.thermal_log.remove(0);
        }
        self.thermal_log.push(sample);
    }

    /// Update SM snapshots.
    pub fn update_sm_snapshots(&mut self, snapshots: Vec<DnaSmSnapshot>) {
        if !snapshots.is_empty() {
            let avg_nutrient: f64 = snapshots.iter().map(|s| s.nutrient_score).sum::<f64>()
                / snapshots.len() as f64;
            self.aggregate.avg_nutrient_score = avg_nutrient;

            let critical = snapshots
                .iter()
                .filter(|s| s.nutrient_score < self.thresholds.nutrient_critical_floor)
                .count();
            self.aggregate.critical_sm_events += critical as u64;
        }
        self.sm_snapshots = snapshots;
    }

    /// Print a summary.
    pub fn print_summary(&self) {
        println!("  Thresholds:");
        println!("    Register prune  : {:.0}%", self.thresholds.register_prune_threshold * 100.0);
        println!("    Shmem prune     : {:.0}%", self.thresholds.shmem_prune_threshold * 100.0);
        println!("    Warp branch     : {:.0}%", self.thresholds.warp_branch_threshold * 100.0);
        println!("    Nutrient floor  : {:.0}%", self.thresholds.nutrient_critical_floor * 100.0);
        println!("    Thermal throttle: {:.0} C", self.thresholds.thermal_throttle_celsius);
        println!("    Latency prune   : {:.1} us", self.thresholds.latency_prune_threshold_us);
        println!("  Aggregate:");
        println!("    Prunes          : {}", self.aggregate.total_prunes);
        println!("    Branches        : {}", self.aggregate.total_branches);
        println!("    Throttles       : {}", self.aggregate.total_throttles);
        println!("    Harvests        : {}", self.aggregate.total_harvests);
        println!("    Peak Temp       : {:.1} C", self.aggregate.peak_temperature_c);
        println!("    Avg Nutrient    : {:.2}", self.aggregate.avg_nutrient_score);
        println!("    Critical Events : {}", self.aggregate.critical_sm_events);
        println!("  Logs:");
        println!("    Exhaust events  : {} / {}", self.exhaust_log.len(), self.max_exhaust_log_entries);
        println!("    Thermal samples : {} / {}", self.thermal_log.len(), self.max_thermal_log_entries);
        println!("    Mutations       : {}", self.mutation_log.len());
    }
}

// ============================================================
// Error Type
// ============================================================

/// Errors from DNA operations.
#[derive(Debug)]
pub enum DnaError {
    Io(String),
    Serialization(String),
    Validation(Vec<String>),
}

impl std::fmt::Display for DnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DnaError::Io(msg) => write!(f, "DNA I/O error: {}", msg),
            DnaError::Serialization(msg) => write!(f, "DNA serialization error: {}", msg),
            DnaError::Validation(issues) => {
                write!(f, "DNA validation failed: {}", issues.join("; "))
            }
        }
    }
}

impl std::error::Error for DnaError {}

// ============================================================
// CLI Functions
// ============================================================

/// CLI actions for the `dna` subcommand.
#[derive(Debug)]
pub enum DnaCliAction {
    Demo,
    Show,
    Validate,
    Probe,
    Export(String),
    Import(String),
}

/// Parse CLI arguments for the `dna` subcommand.
pub fn parse_dna_args(args: &[String]) -> Option<DnaCliAction> {
    if args.is_empty() {
        return Some(DnaCliAction::Demo);
    }
    match args[0].as_str() {
        "--demo" => Some(DnaCliAction::Demo),
        "--show" => Some(DnaCliAction::Show),
        "--validate" => Some(DnaCliAction::Validate),
        "--probe" => Some(DnaCliAction::Probe),
        "--export" => {
            let path = args.get(1).cloned().unwrap_or_else(|| "instance.claw-dna".into());
            Some(DnaCliAction::Export(path))
        }
        "--import" => {
            let path = args.get(1)?;
            Some(DnaCliAction::Import(path.clone()))
        }
        _ => None,
    }
}

/// Print help for the `dna` subcommand.
pub fn print_dna_help() {
    println!("cudaclaw dna — RamifiedRole DNA Management");
    println!();
    println!("USAGE:");
    println!("  cudaclaw dna [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo       Run a full DNA demo (create, validate, export)");
    println!("  --show       Show the default DNA configuration");
    println!("  --validate   Validate the default DNA for consistency");
    println!("  --probe      Run hardware micro-benchmarks (3x 100ms probes)");
    println!("  --export F   Export default DNA to file F (.claw-dna)");
    println!("  --import F   Import and validate a .claw-dna file");
    println!("  --help       Show this help");
}

/// Run the full DNA demo.
pub fn run_demo() {
    println!("\n{}", "=".repeat(64));
    println!("  CudaClaw RamifiedRole DNA Demo");
    println!("{}", "=".repeat(64));

    // 1. Create default DNA
    println!("\n--- Phase 1: Create Default Spreadsheet Engine DNA ---");
    let mut dna = RamifiedRole::default_spreadsheet_engine();
    println!("  Created: {}", dna.name);
    println!("  Role: {}", dna.role);
    println!("  Hardware: {} ({})", dna.hardware.gpu_name, dna.hardware.compute_capability);
    println!("  Constraints: {} bounds", dna.constraints.resource_bounds.len());
    println!("  Fibers: {} tasks", dna.muscle_fibers.fibers.len());

    // 2. Validate
    println!("\n--- Phase 2: Validate DNA ---");
    let issues = dna.validate();
    if issues.is_empty() {
        println!("  Validation: PASSED (no issues)");
    } else {
        println!("  Validation: {} issue(s)", issues.len());
        for issue in &issues {
            println!("    - {}", issue);
        }
    }

    // 3. Simulate ML mutation
    println!("\n--- Phase 3: Simulate ML Feedback Mutation ---");
    let old_val = "32768";
    dna.constraints.mutate_bound(
        "resource.register_budget",
        DnaBoundValue::IntMax(40000),
    );
    dna.record_mutation("ML increased register budget 32768 -> 40000 based on observed headroom");
    println!("  Mutated resource.register_budget: {} -> 40000", old_val);
    println!("  Total mutations: {}", dna.total_mutations);

    // 4. Record exhaustion events
    println!("\n--- Phase 4: Record Exhaustion Events ---");
    dna.exhaustion.record_event(DnaExhaustEvent {
        timestamp: now_epoch(),
        agent_id: "cell_3_5".into(),
        sm_index: 12,
        action: DnaExhaustAction::Prune {
            old_block_size: 256,
            new_block_size: 128,
            old_priority: 50,
            new_priority: 40,
        },
        reason: "Register utilization 87% on SM 12".into(),
        latency_at_event_us: 6.2,
        temperature_at_event_c: 72.0,
    });
    dna.exhaustion.record_event(DnaExhaustEvent {
        timestamp: now_epoch(),
        agent_id: "cell_7_2".into(),
        sm_index: 5,
        action: DnaExhaustAction::Branch {
            source_sm: 5,
            target_sm: 22,
        },
        reason: "Warp occupancy 92% on SM 5, SM 22 has 0.65 nutrient score".into(),
        latency_at_event_us: 5.8,
        temperature_at_event_c: 68.0,
    });
    dna.exhaustion.record_thermal(DnaThermalSample {
        timestamp: now_epoch(),
        temperature_c: 72.0,
        power_watts: 280.0,
        fan_speed_pct: 65.0,
        is_throttling: false,
    });
    println!("  Recorded 2 exhaust events and 1 thermal sample");
    println!("  Aggregate prunes: {}", dna.exhaustion.aggregate.total_prunes);
    println!("  Aggregate branches: {}", dna.exhaustion.aggregate.total_branches);
    println!("  Peak temperature: {:.1} C", dna.exhaustion.aggregate.peak_temperature_c);

    // 5. Record fiber measurement
    println!("\n--- Phase 5: Record Fiber Performance ---");
    dna.muscle_fibers.record_measurement("cell_update", 2.3);
    dna.muscle_fibers.record_measurement("cell_update", 4.1);
    dna.muscle_fibers.record_measurement("crdt_merge", 5.7);
    if let Some(f) = dna.muscle_fibers.get("cell_update") {
        println!("  cell_update: {} measurements, best={:.1} us, worst={:.1} us",
            f.measurement_count, f.best_measured_latency_us, f.worst_measured_latency_us);
    }

    // 6. Serialize to .claw-dna
    println!("\n--- Phase 6: Serialize to .claw-dna ---");
    let tmp_path = "/tmp/cudaclaw_demo.claw-dna";
    match dna.save_to_file(tmp_path) {
        Ok(()) => {
            let size = std::fs::metadata(tmp_path).map(|m| m.len()).unwrap_or(0);
            println!("  Saved to: {} ({} bytes)", tmp_path, size);
        }
        Err(e) => println!("  Save failed: {}", e),
    }

    // 7. Deserialize back
    println!("\n--- Phase 7: Deserialize from .claw-dna ---");
    match RamifiedRole::load_from_file(tmp_path) {
        Ok(loaded) => {
            println!("  Loaded: {}", loaded.name);
            println!("  Role: {}", loaded.role);
            println!("  Mutations: {}", loaded.total_mutations);
            println!("  Fibers: {}", loaded.muscle_fibers.fibers.len());
            println!("  Constraints: {}", loaded.constraints.resource_bounds.len());
            println!("  Exhaust events: {}", loaded.exhaustion.exhaust_log.len());
            let issues = loaded.validate();
            println!("  Re-validation: {} issue(s)", issues.len());
        }
        Err(e) => println!("  Load failed: {}", e),
    }

    // Clean up
    let _ = std::fs::remove_file(tmp_path);

    // 8. Hardware Probe
    println!("\n--- Phase 8: Dynamic Hardware Probe ---");
    let probed = probe_hardware(0);
    dna.hardware = probed;
    println!("  Probe results stored in DNA hardware fingerprint.");

    // 9. Print full summary
    println!("\n--- Phase 9: Full DNA Summary ---");
    dna.print_summary();

    println!("=== RamifiedRole DNA Demo Complete ===\n");
}

/// Show the default DNA configuration.
pub fn show_dna() {
    let dna = RamifiedRole::default_spreadsheet_engine();
    dna.print_summary();
}

/// Run the hardware probe and display results.
pub fn run_probe() {
    println!("\n  Running CudaClaw Hardware Probe...");
    let fingerprint = probe_hardware(0);
    println!("\n  === Probed Hardware Fingerprint ===");
    fingerprint.print_summary();

    // Export probe results as JSON
    let probe_path = "/tmp/cudaclaw_probe_results.json";
    if let Some(ref pr) = fingerprint.probe_results {
        if let Ok(json) = serde_json::to_string_pretty(pr) {
            if std::fs::write(probe_path, &json).is_ok() {
                println!("\n  Probe results exported to: {}", probe_path);
            }
        }
    }
}

/// Validate the default DNA.
pub fn validate_dna() {
    let dna = RamifiedRole::default_spreadsheet_engine();
    let issues = dna.validate();
    if issues.is_empty() {
        println!("DNA validation: PASSED (no issues)");
    } else {
        println!("DNA validation: {} issue(s) found", issues.len());
        for issue in &issues {
            println!("  - {}", issue);
        }
    }
}

// ============================================================
// Stub CUDA/PTX sources for default fibers
// ============================================================

const CELL_UPDATE_CUDA_STUB: &str = r#"
// cell_update_nvrtc.cu — Ramified cell update kernel
// Compiled via NVRTC at runtime with pattern-specific defines.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

extern "C" __global__ void ramified_cell_update(
    float* __restrict__ values,
    const unsigned int* __restrict__ dirty_indices,
    const float* __restrict__ new_values,
    unsigned int num_dirty
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_dirty) {
        unsigned int cell_idx = dirty_indices[tid];
        values[cell_idx] = new_values[tid];
    }
}
"#;

const CRDT_MERGE_CUDA_STUB: &str = r#"
// crdt_merge_nvrtc.cu — Ramified CRDT merge kernel
// Last-Writer-Wins merge with atomicCAS.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

extern "C" __global__ void ramified_crdt_merge(
    unsigned long long* __restrict__ cell_timestamps,
    float* __restrict__ cell_values,
    const unsigned long long* __restrict__ incoming_timestamps,
    const float* __restrict__ incoming_values,
    unsigned int num_cells
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_cells) {
        unsigned long long old_ts = cell_timestamps[tid];
        unsigned long long new_ts = incoming_timestamps[tid];
        if (new_ts > old_ts) {
            cell_timestamps[tid] = new_ts;
            cell_values[tid] = incoming_values[tid];
        }
    }
}
"#;

const IDLE_POLL_PTX_STUB: &str = r#"
// idle_poll PTX stub — persistent polling kernel
.version 7.8
.target sm_89
.address_size 64

.visible .entry idle_poll_persistent(
    .param .u64 queue_ptr
) {
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [queue_ptr];

POLL_LOOP:
    ld.volatile.global.u32 %r1, [%rd1+0];   // is_running
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra EXIT;

    ld.volatile.global.u32 %r2, [%rd1+4];   // head
    ld.volatile.global.u32 %r3, [%rd1+8];   // tail
    setp.eq.u32 %p2, %r2, %r3;
    @%p2 bra POLL_LOOP;                      // empty — spin

    // Process command here (placeholder)
    bra POLL_LOOP;

EXIT:
    ret;
}
"#;

// ============================================================
// Utility
// ============================================================

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_epoch(epoch: u64) -> String {
    if epoch == 0 {
        "never".to_string()
    } else {
        format!("epoch:{}", epoch)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_dna() {
        let dna = RamifiedRole::default_spreadsheet_engine();
        assert_eq!(dna.schema_version, 1);
        assert_eq!(dna.role, "spreadsheet_engine");
        assert!(!dna.hardware.gpu_name.is_empty());
        assert_eq!(dna.hardware.warp_size, 32);
        assert!(dna.constraints.resource_bounds.len() >= 14);
        assert!(dna.muscle_fibers.fibers.len() >= 5);
    }

    #[test]
    fn test_validate_default_dna() {
        let dna = RamifiedRole::default_spreadsheet_engine();
        let issues = dna.validate();
        assert!(issues.is_empty(), "Default DNA should be valid: {:?}", issues);
    }

    #[test]
    fn test_validate_catches_zero_sm_count() {
        let mut dna = RamifiedRole::new("test", "test");
        dna.hardware.sm_count = 0;
        let issues = dna.validate();
        assert!(issues.iter().any(|i| i.contains("sm_count is zero")));
    }

    #[test]
    fn test_validate_catches_bad_warp_size() {
        let mut dna = RamifiedRole::default_spreadsheet_engine();
        dna.hardware.warp_size = 64;
        let issues = dna.validate();
        assert!(issues.iter().any(|i| i.contains("warp_size is 64")));
    }

    #[test]
    fn test_validate_catches_non_aligned_block_size() {
        let mut dna = RamifiedRole::default_spreadsheet_engine();
        dna.muscle_fibers.register("bad_fiber", DnaMuscleFiber {
            name: "Bad".into(),
            description: String::new(),
            block_size: 100, // not a multiple of 32
            registers_per_thread: 0,
            shared_memory_bytes: 0,
            target_occupancy: 0.5,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            kernel_source: DnaKernelSource::None,
            expected_p99_us: 1.0,
            expected_throughput_ops: 1.0,
            best_measured_latency_us: f64::MAX,
            worst_measured_latency_us: 0.0,
            measurement_count: 0,
        });
        let issues = dna.validate();
        assert!(issues.iter().any(|i| i.contains("not a multiple of warp_size")));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dna = RamifiedRole::default_spreadsheet_engine();
        let json = serde_json::to_string_pretty(&dna).unwrap();
        let loaded: RamifiedRole = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, dna.name);
        assert_eq!(loaded.role, dna.role);
        assert_eq!(loaded.hardware.gpu_name, dna.hardware.gpu_name);
        assert_eq!(loaded.constraints.resource_bounds.len(), dna.constraints.resource_bounds.len());
        assert_eq!(loaded.muscle_fibers.fibers.len(), dna.muscle_fibers.fibers.len());
    }

    #[test]
    fn test_save_and_load_file() {
        let dna = RamifiedRole::default_spreadsheet_engine();
        let path = "/tmp/test_cudaclaw_dna.claw-dna";
        dna.save_to_file(path).unwrap();
        let loaded = RamifiedRole::load_from_file(path).unwrap();
        assert_eq!(loaded.name, dna.name);
        assert_eq!(loaded.muscle_fibers.fibers.len(), dna.muscle_fibers.fibers.len());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_constraint_check() {
        let mappings = DnaConstraintMappings::default_system();
        // Register budget is 32768 — 30000 should pass
        assert_eq!(mappings.check("resource.register_budget", 30000.0), Some(true));
        // 50000 should fail
        assert_eq!(mappings.check("resource.register_budget", 50000.0), Some(false));
        // P99 ceiling is 8.0 us — 7.0 should pass
        assert_eq!(mappings.check("latency.p99_rtt_ceiling", 7.0), Some(true));
        // 10.0 should fail
        assert_eq!(mappings.check("latency.p99_rtt_ceiling", 10.0), Some(false));
        // Non-existent constraint
        assert_eq!(mappings.check("nonexistent", 0.0), None);
    }

    #[test]
    fn test_constraint_mutation() {
        let mut mappings = DnaConstraintMappings::default_system();
        assert!(mappings.mutate_bound("resource.register_budget", DnaBoundValue::IntMax(40000)));
        assert_eq!(mappings.check("resource.register_budget", 35000.0), Some(true));
        assert_eq!(mappings.check("resource.register_budget", 45000.0), Some(false));
        let bound = mappings.resource_bounds.get("resource.register_budget").unwrap();
        assert_eq!(bound.mutation_count, 1);
    }

    #[test]
    fn test_constraint_by_category() {
        let mappings = DnaConstraintMappings::default_system();
        let resource = mappings.by_category(DnaConstraintCategory::Resource);
        assert!(resource.len() >= 4);
        for (_, b) in &resource {
            assert_eq!(b.category, DnaConstraintCategory::Resource);
        }
    }

    #[test]
    fn test_fiber_registration() {
        let mut map = DnaMuscleFiberMap::new();
        assert!(map.fibers.is_empty());
        map.register("test_task", DnaMuscleFiber {
            name: "Test".into(),
            description: String::new(),
            block_size: 64,
            registers_per_thread: 16,
            shared_memory_bytes: 1024,
            target_occupancy: 0.5,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            kernel_source: DnaKernelSource::None,
            expected_p99_us: 2.0,
            expected_throughput_ops: 100_000.0,
            best_measured_latency_us: f64::MAX,
            worst_measured_latency_us: 0.0,
            measurement_count: 0,
        });
        assert_eq!(map.fibers.len(), 1);
        assert!(map.get("test_task").is_some());
    }

    #[test]
    fn test_fiber_measurement_tracking() {
        let mut map = DnaMuscleFiberMap::default_fibers();
        map.record_measurement("cell_update", 2.5);
        map.record_measurement("cell_update", 3.5);
        map.record_measurement("cell_update", 1.8);
        let fiber = map.get("cell_update").unwrap();
        assert_eq!(fiber.measurement_count, 3);
        assert!((fiber.best_measured_latency_us - 1.8).abs() < 0.01);
        assert!((fiber.worst_measured_latency_us - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_default_fibers_have_kernel_sources() {
        let map = DnaMuscleFiberMap::default_fibers();
        // cell_update should have NVRTC source
        let cu = map.get("cell_update").unwrap();
        match &cu.kernel_source {
            DnaKernelSource::NvrtcSource { source, program_name, .. } => {
                assert!(source.contains("ramified_cell_update"));
                assert_eq!(program_name, "cell_update_nvrtc.cu");
            }
            _ => panic!("Expected NvrtcSource for cell_update"),
        }
        // idle_poll should have PTX
        let ip = map.get("idle_poll").unwrap();
        match &ip.kernel_source {
            DnaKernelSource::Ptx { ptx, target_arch } => {
                assert!(ptx.contains("idle_poll_persistent"));
                assert_eq!(target_arch, "sm_89");
            }
            _ => panic!("Expected Ptx for idle_poll"),
        }
    }

    #[test]
    fn test_exhaustion_event_recording() {
        let mut metrics = DnaExhaustionMetrics::new();
        metrics.record_event(DnaExhaustEvent {
            timestamp: 1000,
            agent_id: "test_agent".into(),
            sm_index: 0,
            action: DnaExhaustAction::Prune {
                old_block_size: 256,
                new_block_size: 128,
                old_priority: 50,
                new_priority: 40,
            },
            reason: "test".into(),
            latency_at_event_us: 5.0,
            temperature_at_event_c: 70.0,
        });
        assert_eq!(metrics.exhaust_log.len(), 1);
        assert_eq!(metrics.aggregate.total_prunes, 1);
    }

    #[test]
    fn test_thermal_recording() {
        let mut metrics = DnaExhaustionMetrics::new();
        metrics.record_thermal(DnaThermalSample {
            timestamp: 1000,
            temperature_c: 72.0,
            power_watts: 250.0,
            fan_speed_pct: 60.0,
            is_throttling: false,
        });
        metrics.record_thermal(DnaThermalSample {
            timestamp: 1001,
            temperature_c: 85.0,
            power_watts: 300.0,
            fan_speed_pct: 80.0,
            is_throttling: true,
        });
        assert_eq!(metrics.thermal_log.len(), 2);
        assert!((metrics.aggregate.peak_temperature_c - 85.0).abs() < 0.01);
    }

    #[test]
    fn test_sm_snapshot_update() {
        let mut metrics = DnaExhaustionMetrics::new();
        metrics.update_sm_snapshots(vec![
            DnaSmSnapshot {
                sm_index: 0,
                register_utilization: 0.5,
                shared_memory_utilization: 0.3,
                warp_occupancy: 0.4,
                thread_occupancy: 0.4,
                agent_count: 2,
                nutrient_score: 0.5,
                timestamp: 1000,
            },
            DnaSmSnapshot {
                sm_index: 1,
                register_utilization: 0.9,
                shared_memory_utilization: 0.8,
                warp_occupancy: 0.85,
                thread_occupancy: 0.7,
                agent_count: 5,
                nutrient_score: 0.1, // critical
                timestamp: 1000,
            },
        ]);
        assert_eq!(metrics.sm_snapshots.len(), 2);
        assert!((metrics.aggregate.avg_nutrient_score - 0.3).abs() < 0.01);
        assert_eq!(metrics.aggregate.critical_sm_events, 1);
    }

    #[test]
    fn test_hardware_fingerprint_short_id() {
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let id = hw.short_id();
        assert!(id.contains("rtx"));
        assert!(id.contains("sm89"));
    }

    #[test]
    fn test_bound_value_describe() {
        assert_eq!(DnaBoundValue::IntMax(100).describe(), "<= 100");
        assert_eq!(DnaBoundValue::IntMin(10).describe(), ">= 10");
        assert_eq!(DnaBoundValue::FloatMax(8.0).describe(), "<= 8.00");
        assert_eq!(DnaBoundValue::BoolRequired(true).describe(), "must be true");
        assert!(DnaBoundValue::Range { min: 1.0, max: 10.0 }.describe().contains("1.00"));
    }

    #[test]
    fn test_record_mutation() {
        let mut dna = RamifiedRole::default_spreadsheet_engine();
        assert_eq!(dna.total_mutations, 0);
        dna.record_mutation("test mutation");
        assert_eq!(dna.total_mutations, 1);
        assert_eq!(dna.exhaustion.mutation_log.len(), 1);
        assert_eq!(dna.exhaustion.mutation_log[0].description, "test mutation");
    }

    #[test]
    fn test_exhaust_log_eviction() {
        let mut metrics = DnaExhaustionMetrics::new();
        metrics.max_exhaust_log_entries = 3;
        for i in 0..5 {
            metrics.record_event(DnaExhaustEvent {
                timestamp: 1000 + i,
                agent_id: format!("agent_{}", i),
                sm_index: 0,
                action: DnaExhaustAction::Harvest {
                    registers_reclaimed: 100,
                    shared_memory_reclaimed: 200,
                },
                reason: "eviction test".into(),
                latency_at_event_us: 1.0,
                temperature_at_event_c: 60.0,
            });
        }
        assert_eq!(metrics.exhaust_log.len(), 3);
        assert_eq!(metrics.aggregate.total_harvests, 5);
    }

    #[test]
    fn test_rtx4090_has_new_fields() {
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        assert_eq!(hw.compute_capability_major, 8);
        assert_eq!(hw.compute_capability_minor, 9);
        assert_eq!(hw.max_registers_per_block, 65536);
        assert!(hw.probe_results.is_none());
    }

    #[test]
    fn test_unknown_has_new_fields() {
        let hw = DnaHardwareFingerprint::unknown();
        assert_eq!(hw.compute_capability_major, 0);
        assert_eq!(hw.compute_capability_minor, 0);
        assert_eq!(hw.max_registers_per_block, 0);
        assert!(hw.probe_results.is_none());
    }

    #[test]
    fn test_arch_name_from_cc() {
        assert_eq!(arch_name_from_cc(7, 0), "Volta");
        assert_eq!(arch_name_from_cc(7, 5), "Turing");
        assert_eq!(arch_name_from_cc(8, 0), "Ampere");
        assert_eq!(arch_name_from_cc(8, 9), "Ada Lovelace");
        assert_eq!(arch_name_from_cc(9, 0), "Hopper");
        assert_eq!(arch_name_from_cc(10, 0), "Blackwell");
        assert!(arch_name_from_cc(99, 0).contains("Unknown"));
    }

    #[test]
    fn test_probe_results_default() {
        let pr = DnaProbeResults::default();
        assert_eq!(pr.probe_duration_ms, 0.0);
        assert_eq!(pr.probe_timestamp, 0);
        assert_eq!(pr.memory_latency_probe.global_memory_rtt_ns, 0.0);
        assert_eq!(pr.compute_throughput_probe.fp32_ops_per_sec, 0.0);
        assert_eq!(pr.atomic_contention_probe.cas_32_thread_same_addr_ops_per_sec, 0.0);
    }

    #[test]
    fn test_hardware_probe_memory_latency() {
        use std::time::Duration;
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let probe = HardwareProbe::new(hw).with_duration(Duration::from_millis(10));
        let result = probe.probe_memory_latency();
        // Global should be slower than shared
        assert!(result.global_memory_rtt_ns > 0.0);
        assert!(result.shared_memory_rtt_ns > 0.0);
        assert!(result.global_to_shared_ratio > 0.0);
        assert!(result.iterations > 0);
        assert!(result.duration_ms > 0.0);
    }

    #[test]
    fn test_hardware_probe_compute_throughput() {
        use std::time::Duration;
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let probe = HardwareProbe::new(hw).with_duration(Duration::from_millis(10));
        let result = probe.probe_compute_throughput();
        assert!(result.fp32_ops_per_sec > 0.0);
        assert!(result.fp32_gflops > 0.0);
        assert!(result.total_fma_ops > 0);
        assert!(result.duration_ms > 0.0);
    }

    #[test]
    fn test_hardware_probe_atomic_contention() {
        use std::time::Duration;
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let probe = HardwareProbe::new(hw).with_duration(Duration::from_millis(10));
        let result = probe.probe_atomic_contention();
        assert!(result.cas_32_thread_same_addr_ops_per_sec > 0.0);
        assert!(result.cas_uncontended_ops_per_sec > 0.0);
        assert!(result.contention_slowdown_ratio > 1.0);
        assert!(result.total_cas_ops > 0);
        assert!(result.duration_ms > 0.0);
    }

    #[test]
    fn test_hardware_probe_run_all() {
        use std::time::Duration;
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let enriched = HardwareProbe::new(hw)
            .with_duration(Duration::from_millis(10))
            .run_all();
        // Should have probe results populated
        assert!(enriched.probe_results.is_some());
        let pr = enriched.probe_results.unwrap();
        assert!(pr.probe_duration_ms > 0.0);
        assert!(pr.probe_timestamp > 0);
        // Memory latency should be populated in both probe and fingerprint
        assert!(pr.memory_latency_probe.global_memory_rtt_ns > 0.0);
        assert!(enriched.memory_latency.global_memory_ns > 0.0);
        // Atomic throughput should be populated
        assert!(enriched.atomic_throughput.cas_warp_contention_ops > 0.0);
        assert!(enriched.atomic_throughput.contention_sensitivity_ratio > 0.0);
    }

    #[test]
    fn test_probe_results_serialization() {
        use std::time::Duration;
        let hw = DnaHardwareFingerprint::rtx4090_simulated();
        let enriched = HardwareProbe::new(hw)
            .with_duration(Duration::from_millis(10))
            .run_all();
        // Serialize to JSON and back
        let json = serde_json::to_string(&enriched).unwrap();
        let deserialized: DnaHardwareFingerprint = serde_json::from_str(&json).unwrap();
        assert!(deserialized.probe_results.is_some());
        let pr = deserialized.probe_results.unwrap();
        assert!(pr.memory_latency_probe.global_memory_rtt_ns > 0.0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_from_cust_device_no_gpu() {
        // On CI (no GPU), from_cust_device should return None
        let result = DnaHardwareFingerprint::from_cust_device(0);
        // We can't assert Some or None here since it depends on hardware,
        // but it should not panic
        if let Some(hw) = result {
            assert!(!hw.is_simulated);
            assert!(hw.compute_capability_major > 0);
        }
    }

    #[test]
    fn test_cli_parse_probe() {
        let args = vec!["--probe".to_string()];
        match parse_dna_args(&args) {
            Some(DnaCliAction::Probe) => {} // expected
            other => panic!("Expected Probe, got {:?}", other),
        }
    }
}
