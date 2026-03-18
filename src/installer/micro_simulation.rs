// ============================================================
// Micro-Simulation Engine — 5-Second Kernel Benchmark
// ============================================================
//
// Runs a timed "mini-kernel launch" for each candidate
// configuration. Instead of relying solely on a performance
// model, this module executes a simulated GPU workload that
// mimics the persistent-worker kernel's hot loop:
//
//   1. Lock-free CAS push into a ring buffer
//   2. Volatile head-index poll (GPU side)
//   3. Warp-broadcast of command fields
//   4. Cell update with optional atomicCAS
//   5. Completion flag writeback
//
// Each configuration is benchmarked for exactly 5 seconds
// (wall-clock). The engine collects per-command latencies,
// computes P50/P99/max, and measures sustained throughput.
//
// USAGE (called by Installer):
//   let baseline = MicroSimConfig::baseline();
//   let candidate = MicroSimConfig::from_suggestion(&suggestion, &hardware);
//   let engine = MicroSimEngine::new(hardware);
//   let baseline_result = engine.run(&baseline, Duration::from_secs(5));
//   let candidate_result = engine.run(&candidate, Duration::from_secs(5));
//   if candidate_result.beats(&baseline_result) { ... }
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::installer::hardware_probe::HardwareProfile;
use crate::installer::llm_optimizer::OptimizationSuggestion;

// ============================================================
// Micro-Simulation Configuration
// ============================================================

/// Parameters for a single micro-simulation run.
/// Maps 1:1 with the tunable constants the LLM can suggest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroSimConfig {
    /// Identifier for this configuration.
    pub config_id: String,

    /// Threads per block (determines warp count).
    pub block_size: u32,

    /// Grid size (number of blocks).
    pub grid_size: u32,

    /// Idle sleep between polls (nanoseconds).
    pub idle_sleep_ns: u32,

    /// Command batch size before processing.
    pub command_batch_size: u32,

    /// Loop unrolling factor (affects ILP simulation).
    pub loop_unroll_factor: u32,

    /// Whether to use warp-aggregated CAS.
    pub enable_warp_aggregation: bool,

    /// Whether to use SoA memory layout.
    pub use_soa_layout: bool,

    /// Shared memory per block (bytes).
    pub shared_memory_bytes: u32,

    /// L1 cache preference (0=default, 1=prefer L1, 2=prefer shmem).
    pub l1_cache_preference: u32,

    /// CAS backoff initial delay (nanoseconds).
    pub cas_backoff_initial_ns: u32,

    /// CAS backoff maximum delay (nanoseconds).
    pub cas_backoff_max_ns: u32,
}

impl MicroSimConfig {
    /// Create a conservative baseline configuration.
    /// This represents the "stock" settings before any LLM tuning.
    pub fn baseline() -> Self {
        MicroSimConfig {
            config_id: "baseline".to_string(),
            block_size: 32,
            grid_size: 1,
            idle_sleep_ns: 100,
            command_batch_size: 1,
            loop_unroll_factor: 4,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            shared_memory_bytes: 49152,
            l1_cache_preference: 1,
            cas_backoff_initial_ns: 100,
            cas_backoff_max_ns: 10000,
        }
    }

    /// Convert an LLM-generated OptimizationSuggestion into a
    /// MicroSimConfig for benchmarking.
    pub fn from_suggestion(s: &OptimizationSuggestion) -> Self {
        MicroSimConfig {
            config_id: s.suggestion_id.clone(),
            block_size: s.block_size,
            grid_size: s.grid_size,
            idle_sleep_ns: s.idle_sleep_ns,
            command_batch_size: s.command_batch_size,
            loop_unroll_factor: s.loop_unroll_factor,
            enable_warp_aggregation: s.enable_warp_aggregation,
            use_soa_layout: s.use_soa_layout,
            shared_memory_bytes: s.shared_memory_bytes,
            l1_cache_preference: s.l1_cache_preference,
            cas_backoff_initial_ns: s.cas_backoff_initial_ns,
            cas_backoff_max_ns: s.cas_backoff_max_ns,
        }
    }
}

// ============================================================
// Micro-Simulation Result
// ============================================================

/// Result of a single 5-second micro-simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroSimResult {
    /// Which config produced this result.
    pub config_id: String,

    /// Total commands processed in the benchmark window.
    pub total_commands: u64,

    /// Benchmark wall-clock duration (seconds).
    pub duration_secs: f64,

    /// Sustained throughput (commands/sec).
    pub throughput_cmds_per_sec: f64,

    /// Median latency per command (microseconds).
    pub p50_latency_us: f64,

    /// 99th percentile latency (microseconds).
    pub p99_latency_us: f64,

    /// Maximum latency observed (microseconds).
    pub max_latency_us: f64,

    /// Minimum latency observed (microseconds).
    pub min_latency_us: f64,

    /// Mean latency (microseconds).
    pub mean_latency_us: f64,

    /// Standard deviation of latency (microseconds).
    pub stddev_latency_us: f64,

    /// Whether P99 meets the 8 microsecond target.
    pub meets_p99_target: bool,

    /// Simulated GPU temperature at end (Celsius).
    pub estimated_temperature_c: f64,

    /// Simulated power draw (watts, relative).
    pub estimated_power_w: f64,

    /// Number of simulated warp stalls.
    pub warp_stall_count: u64,

    /// Number of simulated CAS retries.
    pub cas_retry_count: u64,
}

impl MicroSimResult {
    /// Returns true if this result strictly beats `other` on
    /// overall quality: lower P99 latency AND higher throughput,
    /// OR same latency tier with >10% throughput advantage.
    pub fn beats(&self, other: &MicroSimResult) -> bool {
        // Primary criterion: P99 latency
        let latency_better = self.p99_latency_us < other.p99_latency_us;
        let throughput_better = self.throughput_cmds_per_sec > other.throughput_cmds_per_sec;

        if latency_better && throughput_better {
            return true;
        }

        // If latency is within 5%, compare throughput with >10% margin
        let latency_ratio = self.p99_latency_us / other.p99_latency_us.max(0.001);
        let throughput_ratio = self.throughput_cmds_per_sec
            / other.throughput_cmds_per_sec.max(1.0);

        if latency_ratio <= 1.05 && throughput_ratio > 1.10 {
            return true;
        }

        // If throughput is within 5%, pure latency win
        if throughput_ratio >= 0.95 && latency_better {
            return true;
        }

        false
    }

    /// Compute a weighted score (same formula as SimulationEngine).
    /// Higher is better.
    pub fn score(&self, latency_weight: f64, throughput_weight: f64, power_weight: f64) -> f64 {
        let latency_score = if self.p99_latency_us > 0.0 {
            (8.0 / self.p99_latency_us * 100.0).min(100.0)
        } else {
            100.0
        };

        let throughput_score = (self.throughput_cmds_per_sec / 10_000_000.0 * 100.0).min(100.0);

        // Power efficiency: lower temp = more efficient
        let power_score = if self.estimated_temperature_c < 70.0 {
            80.0
        } else if self.estimated_temperature_c < 85.0 {
            60.0
        } else {
            40.0
        };

        latency_score * latency_weight
            + throughput_score * throughput_weight
            + power_score * power_weight
    }

    /// Print a one-line summary.
    pub fn print_summary(&self) {
        let p99_ok = if self.meets_p99_target { "YES" } else { "NO " };
        println!(
            "  {:<24} P99: {:>7.2} us  Throughput: {:>10.0} cmd/s  \
             Temp: {:>5.1} C  P99 OK: {}",
            self.config_id,
            self.p99_latency_us,
            self.throughput_cmds_per_sec,
            self.estimated_temperature_c,
            p99_ok,
        );
    }
}

// ============================================================
// Micro-Simulation Engine
// ============================================================

/// Executes timed micro-simulations of the cudaclaw hot loop.
///
/// The engine models the persistent-worker kernel by simulating:
///   - Lock-free ring buffer pushes (CAS on head index)
///   - Polling delay (idle_sleep_ns)
///   - Warp-broadcast overhead
///   - Cell update with optional atomicCAS contention
///   - Completion flag writeback
///
/// On a machine without a real GPU, the simulation uses hardware
/// profile data to compute realistic timing. On a CUDA machine,
/// this would launch the actual persistent_worker kernel with the
/// candidate's parameters.
pub struct MicroSimEngine {
    hardware: HardwareProfile,
}

impl MicroSimEngine {
    pub fn new(hardware: HardwareProfile) -> Self {
        MicroSimEngine { hardware }
    }

    /// Run a micro-simulation for the given duration (typically 5 seconds).
    ///
    /// Returns detailed performance metrics.
    pub fn run(&self, config: &MicroSimConfig, duration: Duration) -> MicroSimResult {
        println!(
            "    Running micro-simulation: {} ({}s)...",
            config.config_id,
            duration.as_secs()
        );

        // Shared state: simulates unified-memory ring buffer
        let head = Arc::new(AtomicU64::new(0));
        let tail = Arc::new(AtomicU64::new(0));
        let is_running = Arc::new(AtomicU32::new(1));

        let mut latencies_ns: Vec<u64> = Vec::with_capacity(2_000_000);

        let start = Instant::now();
        let mut commands_processed: u64 = 0;
        let mut warp_stalls: u64 = 0;
        let mut cas_retries: u64 = 0;

        // Pre-compute per-command cost model from hardware + config
        let cost = self.compute_cost_model(config);

        // Main simulation loop — runs for exactly `duration` wall-clock time
        while start.elapsed() < duration {
            let batch_start = Instant::now();

            // Simulate a batch of commands
            let batch_size = config.command_batch_size.max(1) as u64;

            for _ in 0..batch_size {
                let cmd_start = Instant::now();

                // Phase 1: Host push (CAS on head)
                let old_head = head.fetch_add(1, Ordering::Relaxed);
                // Simulate push latency
                self.spin_ns(cost.host_push_ns);

                // Phase 2: GPU poll detection
                // Simulate the idle sleep + poll cycle
                self.spin_ns(cost.poll_detect_ns);

                // Phase 3: Warp broadcast
                self.spin_ns(cost.warp_broadcast_ns);

                // Phase 4: GPU processing (cell update or CRDT merge)
                self.spin_ns(cost.gpu_process_ns);

                // Phase 5: Optional CAS contention
                if config.enable_warp_aggregation {
                    // Warp-aggregated: single-lane CAS, rarely contended
                    self.spin_ns(cost.cas_single_ns);
                    if old_head % 64 == 0 {
                        // Occasional contention from adjacent warps
                        self.spin_ns(cost.cas_retry_ns);
                        cas_retries += 1;
                    }
                } else {
                    // All lanes CAS: higher contention
                    self.spin_ns(cost.cas_contended_ns);
                    cas_retries += config.block_size as u64 / 32;
                }

                // Phase 6: Completion writeback
                tail.fetch_add(1, Ordering::Release);
                self.spin_ns(cost.writeback_ns);

                let cmd_elapsed = cmd_start.elapsed().as_nanos() as u64;
                latencies_ns.push(cmd_elapsed);
                commands_processed += 1;

                // Check for warp stalls (simulated: happens when occupancy is low)
                if commands_processed % (config.block_size as u64 * 100) == 0 {
                    warp_stalls += 1;
                }
            }

            // Simulate inter-batch idle if using batching
            if config.command_batch_size > 1 {
                self.spin_ns(config.idle_sleep_ns as u64 / 2);
            }
        }

        is_running.store(0, Ordering::Release);
        let total_duration = start.elapsed();

        // Compute statistics
        self.compute_result(
            config,
            &mut latencies_ns,
            commands_processed,
            total_duration,
            warp_stalls,
            cas_retries,
        )
    }

    /// Run the baseline configuration for 5 seconds.
    pub fn run_baseline(&self) -> MicroSimResult {
        let baseline = MicroSimConfig::baseline();
        self.run(&baseline, Duration::from_secs(5))
    }

    /// Run a candidate configuration for 5 seconds and compare
    /// against the baseline result.
    pub fn run_and_compare(
        &self,
        candidate: &MicroSimConfig,
        baseline_result: &MicroSimResult,
    ) -> (MicroSimResult, bool) {
        let result = self.run(candidate, Duration::from_secs(5));
        let is_better = result.beats(baseline_result);
        (result, is_better)
    }

    /// Run the full narrowing loop:
    ///   1. Benchmark baseline (5s)
    ///   2. Benchmark each candidate (5s each)
    ///   3. Return the best configuration
    pub fn narrow(
        &self,
        candidates: &[MicroSimConfig],
    ) -> NarrowingResult {
        println!("\n{}", "=".repeat(72));
        println!("  Micro-Simulation Narrowing (5s per configuration)");
        println!("{}", "=".repeat(72));

        // Step 1: Baseline
        println!("\n  [1/{}] Benchmarking baseline...", candidates.len() + 1);
        let baseline_result = self.run_baseline();
        baseline_result.print_summary();

        // Step 2: Candidates
        let mut all_results = vec![baseline_result.clone()];
        let mut best_result = baseline_result.clone();
        let mut best_config_id = "baseline".to_string();

        for (i, candidate) in candidates.iter().enumerate() {
            println!(
                "\n  [{}/{}] Benchmarking {}...",
                i + 2,
                candidates.len() + 1,
                candidate.config_id
            );
            let (result, is_better) = self.run_and_compare(candidate, &baseline_result);
            result.print_summary();

            if is_better {
                println!("    ^ BEATS BASELINE");
            }

            if result.beats(&best_result) {
                best_result = result.clone();
                best_config_id = candidate.config_id.clone();
            }

            all_results.push(result);
        }

        // Summary
        println!("\n{}", "=".repeat(72));
        println!("  Narrowing complete.");
        println!(
            "  Best: {} (P99: {:.2} us, Throughput: {:.0} cmd/s)",
            best_config_id, best_result.p99_latency_us, best_result.throughput_cmds_per_sec
        );
        let beats_baseline = best_config_id != "baseline";
        if beats_baseline {
            let improvement_pct = (1.0
                - best_result.p99_latency_us / baseline_result.p99_latency_us.max(0.001))
                * 100.0;
            println!("  Improvement over baseline: {:.1}% lower P99 latency", improvement_pct);
        } else {
            println!("  Baseline was the best configuration.");
        }
        println!("{}\n", "=".repeat(72));

        NarrowingResult {
            baseline: baseline_result,
            candidates: all_results,
            best_config_id,
            best_result,
            beats_baseline,
        }
    }

    // ── Cost Model ──────────────────────────────────────────

    /// Build a per-command cost model from hardware profile + config.
    /// All values in nanoseconds.
    fn compute_cost_model(&self, config: &MicroSimConfig) -> CommandCostModel {
        let hw = &self.hardware;

        // Host push: CAS on head index (~50-80 ns on modern CPU)
        let host_push_ns: u64 = 60;

        // Poll detection: idle_sleep + warp switch overhead
        let poll_detect_ns: u64 = config.idle_sleep_ns as u64
            + hw.warp_concurrency.warp_switch_overhead_ns as u64;

        // Warp broadcast: 9 fields * ~1 cycle per __shfl_sync
        let clock_period_ns =
            1_000_000_000.0 / (hw.core_clock_mhz as f64 * 1_000_000.0);
        let warp_broadcast_ns: u64 = (9.0 * clock_period_ns) as u64;

        // GPU processing: memory access + computation
        let mem_ns = if config.use_soa_layout {
            hw.memory_latency.l1_hit_latency_ns * 2.0 // Coalesced: 2 cache lines
        } else {
            hw.memory_latency.l2_hit_latency_ns * 1.2 // AoS: L2 fallback
        };

        // Unrolling benefit
        let unroll_factor = match config.loop_unroll_factor {
            1 => 1.0,
            2 => 0.85,
            4 => 0.75,
            8 => 0.70,
            _ => 0.80,
        };

        let gpu_process_ns: u64 = (150.0 + mem_ns * unroll_factor) as u64;

        // CAS costs
        let cas_single_ns: u64 =
            (1_000_000_000.0 / hw.atomic_throughput.cas_zero_contention_ops_per_sec) as u64;
        let cas_contended_ns: u64 =
            (1_000_000_000.0 / hw.atomic_throughput.cas_warp_contention_ops_per_sec) as u64;
        let cas_retry_ns: u64 = config.cas_backoff_initial_ns as u64;

        // Writeback: small fence + store
        let writeback_ns: u64 =
            30 + (hw.pcie_profile.threadfence_system_overhead_ns * 0.2) as u64;

        CommandCostModel {
            host_push_ns,
            poll_detect_ns,
            warp_broadcast_ns,
            gpu_process_ns,
            cas_single_ns,
            cas_contended_ns,
            cas_retry_ns,
            writeback_ns,
        }
    }

    /// Spin-wait for approximately `ns` nanoseconds.
    /// Uses a busy loop for sub-microsecond precision.
    #[inline(always)]
    fn spin_ns(&self, ns: u64) {
        if ns == 0 {
            return;
        }
        let target = Instant::now() + Duration::from_nanos(ns);
        while Instant::now() < target {
            std::hint::spin_loop();
        }
    }

    /// Compute the final MicroSimResult from raw latency samples.
    fn compute_result(
        &self,
        config: &MicroSimConfig,
        latencies_ns: &mut Vec<u64>,
        total_commands: u64,
        duration: Duration,
        warp_stalls: u64,
        cas_retries: u64,
    ) -> MicroSimResult {
        let duration_secs = duration.as_secs_f64();

        if latencies_ns.is_empty() {
            return MicroSimResult {
                config_id: config.config_id.clone(),
                total_commands: 0,
                duration_secs,
                throughput_cmds_per_sec: 0.0,
                p50_latency_us: 0.0,
                p99_latency_us: 0.0,
                max_latency_us: 0.0,
                min_latency_us: 0.0,
                mean_latency_us: 0.0,
                stddev_latency_us: 0.0,
                meets_p99_target: false,
                estimated_temperature_c: 40.0,
                estimated_power_w: 50.0,
                warp_stall_count: 0,
                cas_retry_count: 0,
            };
        }

        // Sort for percentile computation
        latencies_ns.sort_unstable();

        let n = latencies_ns.len();
        let p50_idx = n / 2;
        let p99_idx = (n as f64 * 0.99) as usize;

        let p50_ns = latencies_ns[p50_idx];
        let p99_ns = latencies_ns[p99_idx.min(n - 1)];
        let max_ns = *latencies_ns.last().unwrap();
        let min_ns = *latencies_ns.first().unwrap();

        let sum: u64 = latencies_ns.iter().sum();
        let mean_ns = sum as f64 / n as f64;

        let variance: f64 = latencies_ns
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let stddev_ns = variance.sqrt();

        let throughput = total_commands as f64 / duration_secs;

        // Estimate temperature from workload intensity
        let base_temp = 45.0; // Idle GPU temp
        let load_factor = (throughput / 5_000_000.0).min(1.0); // Normalize to 5M cmd/s
        let sleep_cooling = (config.idle_sleep_ns as f64 / 1000.0).min(1.0) * 5.0;
        let estimated_temp = base_temp + load_factor * 40.0 - sleep_cooling;

        // Estimate power from config
        let base_power = 50.0; // Idle watts
        let active_power = load_factor * 250.0; // Up to 300W under load
        let sleep_savings = sleep_cooling * 10.0;
        let estimated_power = base_power + active_power - sleep_savings;

        let p99_us = p99_ns as f64 / 1000.0;

        MicroSimResult {
            config_id: config.config_id.clone(),
            total_commands,
            duration_secs,
            throughput_cmds_per_sec: throughput,
            p50_latency_us: p50_ns as f64 / 1000.0,
            p99_latency_us: p99_us,
            max_latency_us: max_ns as f64 / 1000.0,
            min_latency_us: min_ns as f64 / 1000.0,
            mean_latency_us: mean_ns / 1000.0,
            stddev_latency_us: stddev_ns / 1000.0,
            meets_p99_target: p99_us < 8.0,
            estimated_temperature_c: estimated_temp.clamp(35.0, 105.0),
            estimated_power_w: estimated_power.clamp(30.0, 450.0),
            warp_stall_count: warp_stalls,
            cas_retry_count: cas_retries,
        }
    }

    /// Print a comparison table of all results.
    pub fn print_comparison(results: &[MicroSimResult]) {
        println!("\n{}", "=".repeat(100));
        println!("  Micro-Simulation Results (5s per config)");
        println!("{}", "=".repeat(100));
        println!(
            "  {:<24} {:>10} {:>14} {:>10} {:>10} {:>8} {:>8}",
            "Config", "P99 (us)", "Throughput", "P50 (us)", "Max (us)", "Temp C", "P99 OK"
        );
        println!("  {}", "-".repeat(94));

        for r in results {
            let p99_ok = if r.meets_p99_target { "YES" } else { "NO " };
            println!(
                "  {:<24} {:>10.2} {:>11.0} cmd/s {:>10.2} {:>10.2} {:>8.1} {:>8}",
                &r.config_id[..r.config_id.len().min(24)],
                r.p99_latency_us,
                r.throughput_cmds_per_sec,
                r.p50_latency_us,
                r.max_latency_us,
                r.estimated_temperature_c,
                p99_ok,
            );
        }
        println!("{}\n", "=".repeat(100));
    }
}

// ============================================================
// Narrowing Result
// ============================================================

/// Complete result of the narrowing process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrowingResult {
    /// Baseline benchmark result.
    pub baseline: MicroSimResult,

    /// All results (baseline + candidates).
    pub candidates: Vec<MicroSimResult>,

    /// ID of the best configuration.
    pub best_config_id: String,

    /// The best result.
    pub best_result: MicroSimResult,

    /// Whether the best config beats the baseline.
    pub beats_baseline: bool,
}

// ============================================================
// Internal: Per-Command Cost Model
// ============================================================

/// Pre-computed timing for each phase of a command's lifecycle.
/// All values in nanoseconds.
struct CommandCostModel {
    host_push_ns: u64,
    poll_detect_ns: u64,
    warp_broadcast_ns: u64,
    gpu_process_ns: u64,
    cas_single_ns: u64,
    cas_contended_ns: u64,
    cas_retry_ns: u64,
    writeback_ns: u64,
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

    #[test]
    fn test_baseline_config() {
        let config = MicroSimConfig::baseline();
        assert_eq!(config.config_id, "baseline");
        assert_eq!(config.block_size, 32);
        assert!(config.enable_warp_aggregation);
    }

    #[test]
    fn test_from_suggestion() {
        let s = OptimizationSuggestion {
            suggestion_id: "llm-round-1".to_string(),
            block_size: 64,
            grid_size: 2,
            idle_sleep_ns: 50,
            command_batch_size: 4,
            ..Default::default()
        };

        let config = MicroSimConfig::from_suggestion(&s);
        assert_eq!(config.config_id, "llm-round-1");
        assert_eq!(config.block_size, 64);
        assert_eq!(config.grid_size, 2);
        assert_eq!(config.command_batch_size, 4);
    }

    #[test]
    fn test_micro_sim_short_run() {
        let hw = setup();
        let engine = MicroSimEngine::new(hw);
        let config = MicroSimConfig::baseline();

        // Run for only 100ms (not full 5s) to keep tests fast
        let result = engine.run(&config, Duration::from_millis(100));

        assert!(result.total_commands > 0);
        assert!(result.throughput_cmds_per_sec > 0.0);
        assert!(result.p50_latency_us > 0.0);
        assert!(result.p99_latency_us > 0.0);
        assert!(result.p99_latency_us >= result.p50_latency_us);
        assert!(result.max_latency_us >= result.p99_latency_us);
        assert!(result.min_latency_us <= result.p50_latency_us);
        assert!(result.duration_secs >= 0.09); // ~100ms
    }

    #[test]
    fn test_beats_logic() {
        let better = MicroSimResult {
            config_id: "better".into(),
            total_commands: 100_000,
            duration_secs: 5.0,
            throughput_cmds_per_sec: 500_000.0,
            p50_latency_us: 2.0,
            p99_latency_us: 4.0,
            max_latency_us: 10.0,
            min_latency_us: 1.0,
            mean_latency_us: 3.0,
            stddev_latency_us: 1.0,
            meets_p99_target: true,
            estimated_temperature_c: 60.0,
            estimated_power_w: 200.0,
            warp_stall_count: 10,
            cas_retry_count: 50,
        };

        let worse = MicroSimResult {
            config_id: "worse".into(),
            p99_latency_us: 6.0,
            throughput_cmds_per_sec: 400_000.0,
            ..better.clone()
        };

        assert!(better.beats(&worse));
        assert!(!worse.beats(&better));
    }

    #[test]
    fn test_beats_marginal_improvement() {
        let base = MicroSimResult {
            config_id: "base".into(),
            total_commands: 100_000,
            duration_secs: 5.0,
            throughput_cmds_per_sec: 500_000.0,
            p50_latency_us: 3.0,
            p99_latency_us: 5.0,
            max_latency_us: 12.0,
            min_latency_us: 1.0,
            mean_latency_us: 3.5,
            stddev_latency_us: 1.0,
            meets_p99_target: true,
            estimated_temperature_c: 65.0,
            estimated_power_w: 200.0,
            warp_stall_count: 10,
            cas_retry_count: 50,
        };

        // Same latency tier, >10% throughput improvement
        let throughput_win = MicroSimResult {
            config_id: "throughput_win".into(),
            p99_latency_us: 5.1, // Within 5% of base
            throughput_cmds_per_sec: 600_000.0, // >10% better
            ..base.clone()
        };

        assert!(throughput_win.beats(&base));
    }

    #[test]
    fn test_score_computation() {
        let result = MicroSimResult {
            config_id: "test".into(),
            total_commands: 100_000,
            duration_secs: 5.0,
            throughput_cmds_per_sec: 5_000_000.0,
            p50_latency_us: 2.0,
            p99_latency_us: 4.0,
            max_latency_us: 10.0,
            min_latency_us: 1.0,
            mean_latency_us: 3.0,
            stddev_latency_us: 1.0,
            meets_p99_target: true,
            estimated_temperature_c: 60.0,
            estimated_power_w: 200.0,
            warp_stall_count: 10,
            cas_retry_count: 50,
        };

        let score = result.score(0.7, 0.2, 0.1);
        assert!(score > 0.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_narrowing_short() {
        let hw = setup();
        let engine = MicroSimEngine::new(hw);

        // Create a single candidate — use very short runs for test speed
        let candidate = MicroSimConfig {
            config_id: "low-sleep".into(),
            idle_sleep_ns: 10,
            ..MicroSimConfig::baseline()
        };

        // We test the comparison logic, not the full 5s run
        let baseline = engine.run(&MicroSimConfig::baseline(), Duration::from_millis(50));
        let (cand_result, _is_better) =
            engine.run_and_compare(&candidate, &baseline);

        assert!(cand_result.total_commands > 0);
    }

    #[test]
    fn test_empty_latencies() {
        let hw = setup();
        let engine = MicroSimEngine::new(hw);
        let config = MicroSimConfig::baseline();
        let result = engine.compute_result(
            &config,
            &mut Vec::new(),
            0,
            Duration::from_millis(100),
            0,
            0,
        );
        assert_eq!(result.total_commands, 0);
        assert_eq!(result.throughput_cmds_per_sec, 0.0);
    }

    #[test]
    fn test_result_serialization() {
        let result = MicroSimResult {
            config_id: "test".into(),
            total_commands: 42,
            duration_secs: 5.0,
            throughput_cmds_per_sec: 8.4,
            p50_latency_us: 1.0,
            p99_latency_us: 2.0,
            max_latency_us: 5.0,
            min_latency_us: 0.5,
            mean_latency_us: 1.5,
            stddev_latency_us: 0.5,
            meets_p99_target: true,
            estimated_temperature_c: 55.0,
            estimated_power_w: 150.0,
            warp_stall_count: 3,
            cas_retry_count: 7,
        };

        let json = serde_json::to_string(&result).unwrap();
        let loaded: MicroSimResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.config_id, "test");
        assert_eq!(loaded.total_commands, 42);
    }
}
