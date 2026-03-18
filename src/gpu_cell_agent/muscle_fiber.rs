// ============================================================
// Muscle Fiber — Optimized Kernel Configurations
// ============================================================
//
// A "Muscle Fiber" is a named, tuned kernel configuration for a
// specific workload pattern. Each cell agent is assigned to a
// fiber that best matches its observed access pattern. The ML
// feedback loop can reassign agents to different fibers as
// patterns change.
//
// FIBER TYPES:
//   cell_update   — Simple value writes. Small block, low regs.
//   crdt_merge    — Conflict-resolution with CAS. Medium block,
//                   high shmem for hash tables.
//   formula_eval  — Prefix-sum DAG evaluation. Large block for
//                   scan parallelism.
//   batch_process — Bulk operations. Maximum block size for
//                   throughput.
//   idle_poll     — Persistent polling. Minimal resources.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Fiber Type
// ============================================================

/// Named kernel configuration type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FiberType {
    CellUpdate,
    CrdtMerge,
    FormulaEval,
    BatchProcess,
    IdlePoll,
    Custom(String),
}

impl FiberType {
    pub fn as_str(&self) -> &str {
        match self {
            FiberType::CellUpdate => "cell_update",
            FiberType::CrdtMerge => "crdt_merge",
            FiberType::FormulaEval => "formula_eval",
            FiberType::BatchProcess => "batch_process",
            FiberType::IdlePoll => "idle_poll",
            FiberType::Custom(s) => s.as_str(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "cell_update" => FiberType::CellUpdate,
            "crdt_merge" => FiberType::CrdtMerge,
            "formula_eval" => FiberType::FormulaEval,
            "batch_process" => FiberType::BatchProcess,
            "idle_poll" => FiberType::IdlePoll,
            other => FiberType::Custom(other.to_string()),
        }
    }
}

// ============================================================
// Performance Profile
// ============================================================

/// Expected performance characteristics for a fiber.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberPerformanceProfile {
    /// Expected P99 latency in microseconds.
    pub expected_p99_us: f64,
    /// Expected throughput in operations per second.
    pub expected_throughput_ops: f64,
    /// Expected power efficiency (ops per watt).
    pub expected_efficiency: f64,
    /// Number of times this profile has been measured.
    pub measurement_count: u64,
    /// Cumulative measured latency for averaging.
    pub total_measured_latency_us: f64,
    /// Best measured latency.
    pub best_latency_us: f64,
    /// Worst measured latency.
    pub worst_latency_us: f64,
}

impl FiberPerformanceProfile {
    fn new(expected_p99: f64, expected_throughput: f64) -> Self {
        FiberPerformanceProfile {
            expected_p99_us: expected_p99,
            expected_throughput_ops: expected_throughput,
            expected_efficiency: expected_throughput / 200.0, // Assume 200W TDP.
            measurement_count: 0,
            total_measured_latency_us: 0.0,
            best_latency_us: f64::MAX,
            worst_latency_us: 0.0,
        }
    }

    /// Record a measured latency sample.
    pub fn record_measurement(&mut self, latency_us: f64) {
        self.measurement_count += 1;
        self.total_measured_latency_us += latency_us;
        if latency_us < self.best_latency_us {
            self.best_latency_us = latency_us;
        }
        if latency_us > self.worst_latency_us {
            self.worst_latency_us = latency_us;
        }
    }

    /// Average measured latency.
    pub fn avg_measured_latency_us(&self) -> f64 {
        if self.measurement_count == 0 {
            self.expected_p99_us
        } else {
            self.total_measured_latency_us / self.measurement_count as f64
        }
    }
}

// ============================================================
// Muscle Fiber
// ============================================================

/// A tuned GPU kernel configuration for a specific workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuscleFiber {
    /// Fiber type identifier.
    pub fiber_type: FiberType,
    /// Human-readable name.
    pub name: String,
    /// Description of the workload this fiber is optimized for.
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
    /// Whether this fiber is a persistent polling kernel.
    pub is_persistent: bool,

    // ── Performance profile ──
    pub performance: FiberPerformanceProfile,
}

impl MuscleFiber {
    /// Create a cell_update fiber (simple value writes).
    pub fn cell_update() -> Self {
        MuscleFiber {
            fiber_type: FiberType::CellUpdate,
            name: "Cell Update".into(),
            description: "Optimized for simple cell value writes. Small block \
                          size for low latency, minimal register pressure."
                .into(),
            block_size: 128,
            registers_per_thread: 24,
            shared_memory_bytes: 2048,
            target_occupancy: 0.75,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            performance: FiberPerformanceProfile::new(3.0, 500_000.0),
        }
    }

    /// Create a crdt_merge fiber (conflict resolution).
    pub fn crdt_merge() -> Self {
        MuscleFiber {
            fiber_type: FiberType::CrdtMerge,
            name: "CRDT Merge".into(),
            description: "Optimized for CRDT conflict resolution with PTX-level \
                          atomicCAS. Medium block size, high shmem for hash tables."
                .into(),
            block_size: 256,
            registers_per_thread: 40,
            shared_memory_bytes: 49152, // 48 KB for hash map.
            target_occupancy: 0.5,
            uses_ptx_cas: true,
            uses_warp_aggregation: true,
            uses_prefix_sum: false,
            is_persistent: false,
            performance: FiberPerformanceProfile::new(5.0, 200_000.0),
        }
    }

    /// Create a formula_eval fiber (prefix-sum DAG).
    pub fn formula_eval() -> Self {
        MuscleFiber {
            fiber_type: FiberType::FormulaEval,
            name: "Formula Evaluator".into(),
            description: "Optimized for parallel formula recalculation using \
                          prefix-sum scan. Large block for scan parallelism."
                .into(),
            block_size: 512,
            registers_per_thread: 32,
            shared_memory_bytes: 16384, // 16 KB for scan workspace.
            target_occupancy: 0.5,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: true,
            is_persistent: false,
            performance: FiberPerformanceProfile::new(6.0, 100_000.0),
        }
    }

    /// Create a batch_process fiber (bulk ops).
    pub fn batch_process() -> Self {
        MuscleFiber {
            fiber_type: FiberType::BatchProcess,
            name: "Batch Processor".into(),
            description: "Optimized for bulk cell operations. Maximum block size \
                          for throughput-oriented workloads."
                .into(),
            block_size: 1024,
            registers_per_thread: 20,
            shared_memory_bytes: 8192,
            target_occupancy: 0.75,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            performance: FiberPerformanceProfile::new(7.0, 1_000_000.0),
        }
    }

    /// Create an idle_poll fiber (persistent kernel).
    pub fn idle_poll() -> Self {
        MuscleFiber {
            fiber_type: FiberType::IdlePoll,
            name: "Idle Poller".into(),
            description: "Persistent polling kernel with minimal resource usage. \
                          One warp continuously polls the command queue."
                .into(),
            block_size: 32,
            registers_per_thread: 16,
            shared_memory_bytes: 256,
            target_occupancy: 0.03, // Minimal — just one warp.
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: true,
            performance: FiberPerformanceProfile::new(1.0, 10_000_000.0),
        }
    }

    /// Calculate the theoretical occupancy on a given SM.
    pub fn theoretical_occupancy(&self, sm_registers: u32, sm_shared_mem: u32, sm_max_threads: u32) -> f64 {
        // Threads limited by registers.
        let threads_by_regs = if self.registers_per_thread > 0 {
            sm_registers / self.registers_per_thread
        } else {
            sm_max_threads
        };

        // Threads limited by shared memory.
        let blocks_by_shmem = if self.shared_memory_bytes > 0 {
            sm_shared_mem / self.shared_memory_bytes
        } else {
            sm_max_threads / self.block_size
        };
        let threads_by_shmem = blocks_by_shmem * self.block_size;

        // Threads limited by hardware.
        let max_threads = threads_by_regs.min(threads_by_shmem).min(sm_max_threads);

        max_threads as f64 / sm_max_threads as f64
    }
}

// ============================================================
// Fiber Registry
// ============================================================

/// Registry of available Muscle Fibers.
pub struct FiberRegistry {
    fibers: HashMap<String, MuscleFiber>,
}

impl FiberRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        FiberRegistry {
            fibers: HashMap::new(),
        }
    }

    /// Create a registry with all default fibers.
    pub fn default_fibers() -> Self {
        let mut registry = Self::new();
        registry.register(MuscleFiber::cell_update());
        registry.register(MuscleFiber::crdt_merge());
        registry.register(MuscleFiber::formula_eval());
        registry.register(MuscleFiber::batch_process());
        registry.register(MuscleFiber::idle_poll());
        registry
    }

    /// Register a fiber.
    pub fn register(&mut self, fiber: MuscleFiber) {
        self.fibers.insert(fiber.fiber_type.as_str().to_string(), fiber);
    }

    /// Get a fiber by type name.
    pub fn get_fiber_by_name(&self, name: &str) -> Option<&MuscleFiber> {
        self.fibers.get(name)
    }

    /// Get a mutable fiber by type name.
    pub fn get_fiber_mut(&mut self, name: &str) -> Option<&mut MuscleFiber> {
        self.fibers.get_mut(name)
    }

    /// Get all fibers.
    pub fn all_fibers(&self) -> Vec<&MuscleFiber> {
        self.fibers.values().collect()
    }

    /// Number of registered fibers.
    pub fn fiber_count(&self) -> usize {
        self.fibers.len()
    }

    /// Recommend a fiber based on workload characteristics.
    pub fn recommend_fiber(
        &self,
        is_crdt: bool,
        has_formulas: bool,
        is_batch: bool,
        cell_count: u32,
    ) -> &str {
        if is_batch && cell_count > 1000 {
            "batch_process"
        } else if has_formulas {
            "formula_eval"
        } else if is_crdt {
            "crdt_merge"
        } else {
            "cell_update"
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fiber_creation() {
        let fiber = MuscleFiber::cell_update();
        assert_eq!(fiber.fiber_type, FiberType::CellUpdate);
        assert_eq!(fiber.block_size, 128);
        assert!(!fiber.uses_ptx_cas);
    }

    #[test]
    fn test_crdt_merge_fiber() {
        let fiber = MuscleFiber::crdt_merge();
        assert!(fiber.uses_ptx_cas);
        assert!(fiber.uses_warp_aggregation);
        assert_eq!(fiber.shared_memory_bytes, 49152);
    }

    #[test]
    fn test_default_registry() {
        let registry = FiberRegistry::default_fibers();
        assert_eq!(registry.fiber_count(), 5);
        assert!(registry.get_fiber_by_name("cell_update").is_some());
        assert!(registry.get_fiber_by_name("crdt_merge").is_some());
        assert!(registry.get_fiber_by_name("formula_eval").is_some());
        assert!(registry.get_fiber_by_name("batch_process").is_some());
        assert!(registry.get_fiber_by_name("idle_poll").is_some());
    }

    #[test]
    fn test_fiber_recommendation() {
        let registry = FiberRegistry::default_fibers();
        assert_eq!(registry.recommend_fiber(false, false, false, 1), "cell_update");
        assert_eq!(registry.recommend_fiber(true, false, false, 1), "crdt_merge");
        assert_eq!(registry.recommend_fiber(false, true, false, 1), "formula_eval");
        assert_eq!(registry.recommend_fiber(false, false, true, 2000), "batch_process");
    }

    #[test]
    fn test_performance_profile() {
        let mut fiber = MuscleFiber::cell_update();
        fiber.performance.record_measurement(2.5);
        fiber.performance.record_measurement(3.5);
        assert_eq!(fiber.performance.measurement_count, 2);
        assert!((fiber.performance.avg_measured_latency_us() - 3.0).abs() < 0.01);
        assert!((fiber.performance.best_latency_us - 2.5).abs() < 0.01);
        assert!((fiber.performance.worst_latency_us - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_occupancy_calculation() {
        let fiber = MuscleFiber::cell_update();
        // RTX 4090: 65536 regs, 100KB shmem, 2048 max threads per SM.
        let occupancy = fiber.theoretical_occupancy(65536, 102400, 2048);
        assert!(occupancy > 0.0 && occupancy <= 1.0);
    }

    #[test]
    fn test_fiber_type_roundtrip() {
        let ft = FiberType::CellUpdate;
        let s = ft.as_str();
        let ft2 = FiberType::from_str(s);
        assert_eq!(ft, ft2);

        let custom = FiberType::Custom("my_fiber".into());
        assert_eq!(custom.as_str(), "my_fiber");
    }
}
