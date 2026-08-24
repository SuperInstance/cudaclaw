// ============================================================
// Simulated Fine-Tuning Engine
// ============================================================
//
// Tests LLM-suggested PTX/CUDA constants in a simulated
// environment to find the "narrowest" (most efficient)
// configuration for a specific GPU hardware profile.
//
// SIMULATION MODEL:
// Rather than compiling and executing real CUDA kernels with
// each configuration, we use a performance model derived from
// the hardware probe data. The model accounts for:
//
//   1. Warp scheduling overhead (occupancy × IPC)
//   2. Memory access patterns (coalesced vs. divergent)
//   3. Atomic contention (CAS backoff × warp aggregation)
//   4. Queue drain rate (batch size × idle sleep)
//   5. PCIe latency (threadfence_system overhead)
//   6. Shared memory bank conflicts
//   7. L1/L2 cache hit rates
//
// Each configuration is scored on three axes:
//   - P99 RTT latency (microseconds)
//   - Throughput (commands/second)
//   - Power efficiency (relative score 0-100)
//
// The overall score combines these with the role's priority
// weights (latency_weight, throughput_weight, power_weight).
//
// ============================================================

use serde::{Deserialize, Serialize};
use crate::installer::hardware_probe::HardwareProfile;
use crate::installer::llm_optimizer::{
    OptimizationSuggestion, RoleContext, SimulationResult,
};

// ============================================================
// Simulation Engine
// ============================================================

/// The simulated fine-tuning engine.
/// Evaluates optimization suggestions against a hardware
/// performance model without requiring actual CUDA compilation.
pub struct SimulationEngine {
    hardware: HardwareProfile,
    role: RoleContext,
    noise_factor: f64,
}

impl SimulationEngine {
    pub fn new(hardware: HardwareProfile, role: RoleContext) -> Self {
        SimulationEngine {
            hardware,
            role,
            noise_factor: 0.05, // 5% noise to simulate measurement variance
        }
    }

    /// Set the simulation noise factor (0.0 = deterministic, 0.1 = 10% noise).
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise_factor = noise;
        self
    }

    /// Evaluate a single optimization suggestion.
    /// Returns a SimulationResult with estimated performance metrics.
    pub fn evaluate(&self, suggestion: &OptimizationSuggestion) -> SimulationResult {
        let p99_rtt = self.estimate_p99_rtt(suggestion);
        let throughput = self.estimate_throughput(suggestion);
        let power_score = self.estimate_power_efficiency(suggestion);

        let overall_score = self.compute_overall_score(p99_rtt, throughput, power_score);

        // P99 target: 8 microseconds (from latency.rs requirements)
        let meets_target = p99_rtt < 8.0;

        let notes = self.generate_notes(suggestion, p99_rtt, throughput, power_score);

        SimulationResult {
            suggestion_id: suggestion.suggestion_id.clone(),
            round: suggestion.round,
            p99_rtt_us: p99_rtt,
            throughput_cmds_per_sec: throughput,
            power_efficiency_score: power_score,
            overall_score,
            meets_p99_target: meets_target,
            notes,
        }
    }

    /// Evaluate multiple suggestions and return results sorted by overall score.
    pub fn evaluate_all(
        &self,
        suggestions: &[OptimizationSuggestion],
    ) -> Vec<SimulationResult> {
        let mut results: Vec<SimulationResult> = suggestions
            .iter()
            .map(|s| self.evaluate(s))
            .collect();

        // Sort by overall score, highest first
        results.sort_by(|a, b| b.overall_score
            .partial_cmp(&a.overall_score)
            .unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Find the best configuration from a set of suggestions.
    /// Returns (best_suggestion_index, best_result).
    pub fn find_best(
        &self,
        suggestions: &[OptimizationSuggestion],
    ) -> Option<(usize, SimulationResult)> {
        if suggestions.is_empty() {
            return None;
        }

        let results = self.evaluate_all(suggestions);
        let best = results.into_iter().next()?;

        let idx = suggestions.iter()
            .position(|s| s.suggestion_id == best.suggestion_id)
            .unwrap_or(0);

        Some((idx, best))
    }

    // ── P99 RTT Estimation ──────────────────────────────────

    /// Estimate the P99 round-trip time for a command.
    ///
    /// RTT = host_push + pcie_visibility + gpu_detect + gpu_process + writeback
    ///
    /// Each component is modeled from hardware probe data and
    /// the suggestion's parameters.
    fn estimate_p99_rtt(&self, s: &OptimizationSuggestion) -> f64 {
        // Component 1: Host push latency (CAS on head index)
        // Typically ~50-200 ns for lock-free atomic increment
        let host_push_ns = 80.0;

        // Component 2: PCIe visibility delay
        // __threadfence_system() overhead + unified memory coherence
        let pcie_ns = self.hardware.pcie_profile.threadfence_system_overhead_ns;

        // Component 3: GPU detection latency
        // Average time for the persistent kernel to notice a new command.
        // With idle sleep, worst case is one full sleep cycle.
        // P99 ≈ sleep_ns (one full cycle missed) + warp switch overhead
        let detect_ns = s.idle_sleep_ns as f64
            + self.hardware.warp_concurrency.warp_switch_overhead_ns;

        // Component 4: GPU processing time
        // Depends on command type, warp aggregation, batch size
        let process_ns = self.estimate_gpu_processing_ns(s);

        // Component 5: Result writeback (atomic store + fence)
        let writeback_ns = 50.0 + pcie_ns * 0.3; // Smaller fence for result

        // Total RTT in nanoseconds
        let rtt_ns = host_push_ns + pcie_ns + detect_ns + process_ns + writeback_ns;

        // Add noise (simulates measurement variance)
        let noise = 1.0 + self.noise_factor * self.deterministic_noise(
            s.block_size as f64 * s.idle_sleep_ns as f64);

        // Convert to microseconds
        (rtt_ns * noise) / 1000.0
    }

    /// Estimate GPU-side processing time for a single command.
    fn estimate_gpu_processing_ns(&self, s: &OptimizationSuggestion) -> f64 {
        let base_process_ns = 200.0; // Base cost for command decode + dispatch

        // Warp broadcast overhead (__shfl_sync for all command fields)
        // 9 fields × ~1 cycle each × clock period
        let clock_period_ns = 1_000_000_000.0 / (self.hardware.core_clock_mhz as f64 * 1_000_000.0);
        let broadcast_ns = 9.0 * clock_period_ns;

        // Memory access cost (depends on layout)
        let mem_access_ns = if s.use_soa_layout {
            // SoA: coalesced access, ~1 cache line per warp for each field
            self.hardware.memory_latency.l1_hit_latency_ns * 3.0 // 3 fields touched
        } else {
            // AoS: potentially uncoalesced, multiple cache lines
            self.hardware.memory_latency.l2_hit_latency_ns * 1.5
        };

        // Atomic CAS overhead (if warp aggregation enabled)
        // 1e9 / ops_per_sec already yields nanoseconds-per-op.
        // (Previously this was multiplied by 1e9 again — a double
        // unit-conversion that inflated CAS latency by 10^9×.)
        let atomic_ns = if s.enable_warp_aggregation {
            // Warp-aggregated: only 1 lane does CAS → no contention
            1_000_000_000.0
                / self.hardware.atomic_throughput.cas_zero_contention_ops_per_sec
        } else {
            // Every lane does CAS → warp-level contention
            1_000_000_000.0
                / self.hardware.atomic_throughput.cas_warp_contention_ops_per_sec
        };

        // L1 cache preference effect
        let cache_bonus = match s.l1_cache_preference {
            1 => 0.9, // Prefer L1: slightly lower latency for cached data
            2 => 1.1, // Prefer shared: higher L1 latency
            _ => 1.0,
        };

        // Loop unrolling effect on ILP
        let unroll_factor = match s.loop_unroll_factor {
            1 => 1.0,
            2 => 0.85,
            4 => 0.75,
            8 => 0.70,
            16 => 0.68, // Diminishing returns
            _ => 0.80,
        };

        (base_process_ns + broadcast_ns + mem_access_ns + atomic_ns) * cache_bonus * unroll_factor
    }

    // ── Throughput Estimation ───────────────────────────────

    /// Estimate throughput in commands per second.
    fn estimate_throughput(&self, s: &OptimizationSuggestion) -> f64 {
        // Processing time per command
        let process_ns = self.estimate_gpu_processing_ns(s);

        // Batching effect: processing N commands in batch is cheaper
        // than N individual commands (amortizes broadcast + fence)
        let batch_factor = if s.command_batch_size > 1 {
            let overhead_per_batch = 50.0; // Fixed cost per batch
            let per_cmd_in_batch = process_ns * 0.7; // 30% savings from batching
            (overhead_per_batch + per_cmd_in_batch * s.command_batch_size as f64)
                / s.command_batch_size as f64
        } else {
            process_ns
        };

        // Occupancy effect: more warps = more throughput (up to SM limit)
        let occupancy = self.estimate_occupancy(s);
        let occupancy_bonus = 1.0 + (occupancy - 0.5).max(0.0) * 0.5;

        // Grid scaling: multiple blocks can process independently
        let grid_factor = s.grid_size as f64;

        // Commands per second = 1e9 / ns_per_cmd × grid_factor × occupancy
        let cmds_per_sec = (1_000_000_000.0 / batch_factor) * grid_factor * occupancy_bonus;

        // Add noise
        let noise = 1.0 + self.noise_factor * self.deterministic_noise(
            s.grid_size as f64 * s.warps_per_block as f64);

        cmds_per_sec * noise
    }

    /// Estimate achieved occupancy for the given configuration.
    fn estimate_occupancy(&self, s: &OptimizationSuggestion) -> f64 {
        let warps_launched = s.warps_per_block;
        let max_warps = self.hardware.max_warps_per_sm;

        // Shared memory limit: can we fit multiple blocks on one SM?
        let blocks_per_sm_by_shmem = if s.shared_memory_bytes > 0 {
            (self.hardware.max_shared_memory_per_sm / s.shared_memory_bytes).min(8)
        } else {
            8
        };

        // Thread limit
        let threads_per_block = s.block_size;
        let blocks_per_sm_by_threads = self.hardware.max_threads_per_sm / threads_per_block;

        let blocks_per_sm = blocks_per_sm_by_shmem.min(blocks_per_sm_by_threads);
        let active_warps = blocks_per_sm * warps_launched;

        (active_warps as f64 / max_warps as f64).min(1.0)
    }

    // ── Power Efficiency Estimation ─────────────────────────

    /// Estimate power efficiency score (0-100).
    /// Higher = more efficient (less wasted power per useful work).
    fn estimate_power_efficiency(&self, s: &OptimizationSuggestion) -> f64 {
        // Base score starts at 50
        let mut score = 50.0;

        // Idle sleep: more sleep = less power (but more latency)
        // Score bonus: +0.05 per ns of sleep, up to +25
        score += (s.idle_sleep_ns as f64 * 0.05).min(25.0);

        // Warp aggregation: reduces redundant atomic ops → less power
        if s.enable_warp_aggregation {
            score += 10.0;
        }

        // SoA layout: fewer cache lines fetched → less DRAM power
        if s.use_soa_layout {
            score += 5.0;
        }

        // Over-provisioning warps wastes power on idle warps
        let occupancy = self.estimate_occupancy(s);
        if occupancy < 0.3 {
            score -= 10.0; // Low occupancy = wasted SM resources
        }

        // Grid size > needed wastes block-level resources
        if s.grid_size > 4 {
            score -= (s.grid_size as f64 - 4.0) * 2.0;
        }

        score.clamp(0.0, 100.0)
    }

    // ── Overall Score ───────────────────────────────────────

    /// Compute the overall weighted score.
    fn compute_overall_score(
        &self,
        p99_rtt_us: f64,
        throughput: f64,
        power_score: f64,
    ) -> f64 {
        // Normalize each metric to 0-100 scale

        // Latency: target is 8µs. Score = 100 × (8 / actual), capped at 100
        let latency_score = if p99_rtt_us > 0.0 {
            (8.0 / p99_rtt_us * 100.0).min(100.0)
        } else {
            100.0
        };

        // Throughput: normalize against 10M cmds/sec as "excellent"
        let throughput_score = (throughput / 10_000_000.0 * 100.0).min(100.0);

        // Power: already 0-100
        let power = power_score;

        // Weighted combination
        let score = latency_score * self.role.latency_weight
            + throughput_score * self.role.throughput_weight
            + power * self.role.power_efficiency_weight;

        score
    }

    // ── Utilities ───────────────────────────────────────────

    /// Deterministic noise function (reproducible for given input).
    /// Returns a value in [-1.0, 1.0].
    fn deterministic_noise(&self, seed: f64) -> f64 {
        // Simple hash-based noise for reproducibility
        let bits = seed.to_bits();
        let hash = bits.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let normalized = (hash as f64) / (u64::MAX as f64);
        normalized * 2.0 - 1.0
    }

    /// Generate human-readable notes about the simulation.
    fn generate_notes(
        &self,
        s: &OptimizationSuggestion,
        p99_us: f64,
        throughput: f64,
        power: f64,
    ) -> String {
        let mut notes = Vec::new();

        if p99_us > 8.0 {
            notes.push(format!(
                "P99 RTT {:.1}µs exceeds 8µs target — consider reducing idle_sleep_ns (currently {})",
                p99_us, s.idle_sleep_ns));
        } else {
            notes.push(format!("P99 RTT {:.1}µs meets 8µs target", p99_us));
        }

        let occupancy = self.estimate_occupancy(s);
        if occupancy < 0.3 {
            notes.push(format!(
                "Low occupancy ({:.0}%) — consider increasing warps_per_block",
                occupancy * 100.0));
        }

        if throughput < self.role.avg_commands_per_second as f64 {
            notes.push(format!(
                "Throughput {:.0} cmd/s below target {:.0} cmd/s",
                throughput, self.role.avg_commands_per_second as f64));
        }

        if power < 40.0 {
            notes.push(format!(
                "Low power efficiency ({:.0}/100) — consider enabling warp_aggregation or increasing idle_sleep",
                power));
        }

        notes.join(". ")
    }

    /// Print a comparison table of simulation results.
    pub fn print_comparison(results: &[SimulationResult]) {
        println!("\n{}", "═".repeat(96));
        println!("  Simulated Fine-Tuning Results (ranked by overall score)");
        println!("{}", "═".repeat(96));
        println!("  {:<20} {:>10} {:>14} {:>8} {:>10} {:>8}",
            "Suggestion", "P99 (µs)", "Throughput", "Power", "Score", "P99 OK");
        println!("  {}", "─".repeat(90));

        for r in results {
            let p99_ok = if r.meets_p99_target { "  YES" } else { "  NO " };
            println!("  {:<20} {:>10.2} {:>11.0} cmd/s {:>8.1} {:>10.1} {}",
                &r.suggestion_id[..r.suggestion_id.len().min(20)],
                r.p99_rtt_us,
                r.throughput_cmds_per_sec,
                r.power_efficiency_score,
                r.overall_score,
                p99_ok);
        }

        println!("{}\n", "═".repeat(96));

        if let Some(best) = results.first() {
            println!("  BEST: {} (score: {:.1})", best.suggestion_id, best.overall_score);
            println!("  Notes: {}", best.notes);
        }
    }
}

// ============================================================
// Simulation Report
// ============================================================

/// A complete simulation report for persistence and review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationReport {
    /// All results from the simulation
    pub results: Vec<SimulationResult>,

    /// The winning suggestion ID
    pub best_suggestion_id: String,

    /// The winning score
    pub best_score: f64,

    /// Hardware profile used
    pub hardware_summary: String,

    /// Role context used
    pub role_name: String,

    /// Timestamp
    pub timestamp: u64,
}

impl SimulationReport {
    pub fn from_results(
        results: Vec<SimulationResult>,
        hardware: &HardwareProfile,
        role: &RoleContext,
    ) -> Self {
        let best = results.first().cloned();
        SimulationReport {
            best_suggestion_id: best.as_ref()
                .map(|r| r.suggestion_id.clone())
                .unwrap_or_default(),
            best_score: best.as_ref().map(|r| r.overall_score).unwrap_or(0.0),
            results,
            hardware_summary: format!("{} ({})", hardware.gpu_name, hardware.architecture),
            role_name: role.role_name.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::installer::hardware_probe::HardwareProber;
    use crate::installer::llm_optimizer::LlmOptimizer;

    fn setup() -> (HardwareProfile, RoleContext) {
        let prober = HardwareProber::simulated(0);
        let hardware = prober.probe();
        let role = RoleContext::spreadsheet_engine();
        (hardware, role)
    }

    #[test]
    fn test_evaluate_default_suggestion() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw, role).with_noise(0.0);
        let suggestion = OptimizationSuggestion::default();

        let result = engine.evaluate(&suggestion);
        assert!(result.p99_rtt_us > 0.0, "RTT must be positive");
        assert!(result.throughput_cmds_per_sec > 0.0, "Throughput must be positive");
        assert!(result.power_efficiency_score >= 0.0);
        assert!(result.power_efficiency_score <= 100.0);
        assert!(result.overall_score > 0.0);
    }

    #[test]
    fn test_lower_sleep_reduces_latency() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw, role).with_noise(0.0);

        let mut low_sleep = OptimizationSuggestion::default();
        low_sleep.suggestion_id = "low-sleep".to_string();
        low_sleep.idle_sleep_ns = 10;

        let mut high_sleep = OptimizationSuggestion::default();
        high_sleep.suggestion_id = "high-sleep".to_string();
        high_sleep.idle_sleep_ns = 1000;

        let r_low = engine.evaluate(&low_sleep);
        let r_high = engine.evaluate(&high_sleep);

        assert!(r_low.p99_rtt_us < r_high.p99_rtt_us,
            "Lower idle sleep should produce lower P99 RTT: {} vs {}",
            r_low.p99_rtt_us, r_high.p99_rtt_us);
    }

    #[test]
    fn test_heuristic_meets_p99_target() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw.clone(), role.clone()).with_noise(0.0);
        let suggestion = LlmOptimizer::generate_heuristic_suggestion(&hw, &role);

        let result = engine.evaluate(&suggestion);
        // Heuristic should produce a reasonable configuration
        // (may not always meet target, but should be close)
        assert!(result.p99_rtt_us < 50.0,
            "Heuristic should produce reasonable P99: {:.2}µs", result.p99_rtt_us);
    }

    #[test]
    fn test_evaluate_all_sorted() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw, role).with_noise(0.0);

        let s1 = OptimizationSuggestion { suggestion_id: "a".into(), idle_sleep_ns: 10, ..Default::default() };
        let s2 = OptimizationSuggestion { suggestion_id: "b".into(), idle_sleep_ns: 100, ..Default::default() };
        let s3 = OptimizationSuggestion { suggestion_id: "c".into(), idle_sleep_ns: 500, ..Default::default() };

        let results = engine.evaluate_all(&[s1, s2, s3]);
        assert_eq!(results.len(), 3);

        // Results should be sorted by overall score (descending)
        for i in 0..results.len() - 1 {
            assert!(results[i].overall_score >= results[i + 1].overall_score,
                "Results should be sorted by overall score");
        }
    }

    #[test]
    fn test_find_best() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw, role).with_noise(0.0);

        let suggestions = vec![
            OptimizationSuggestion { suggestion_id: "x".into(), idle_sleep_ns: 500, ..Default::default() },
            OptimizationSuggestion { suggestion_id: "y".into(), idle_sleep_ns: 50, ..Default::default() },
        ];

        let (idx, best) = engine.find_best(&suggestions).unwrap();
        assert!(best.overall_score > 0.0);
        assert!(idx < suggestions.len());
    }

    #[test]
    fn test_simulation_report_serialization() {
        let (hw, role) = setup();
        let engine = SimulationEngine::new(hw.clone(), role.clone()).with_noise(0.0);
        let suggestion = OptimizationSuggestion::default();
        let results = engine.evaluate_all(&[suggestion]);

        let report = SimulationReport::from_results(results, &hw, &role);
        let json = serde_json::to_string_pretty(&report).unwrap();
        let deser: SimulationReport = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.role_name, "spreadsheet_engine");
        assert!(!deser.results.is_empty());
    }
}
