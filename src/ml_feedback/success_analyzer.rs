// ============================================================
// Success Analyzer — Pattern Detection & Mutation Recommendations
// ============================================================
//
// Analyzes the execution log to detect performance patterns and
// recommend DNA mutations. This is the "brain" of the feedback
// loop — it decides what constraints need updating.
//
// ANALYSIS PIPELINE:
//   1. Per-fiber statistics (latency, success rate, resource use)
//   2. Trend detection (improving, degrading, stable)
//   3. Constraint headroom calculation (how close to the limit)
//   4. Mutation recommendation generation
//
// ============================================================

use serde::{Deserialize, Serialize};

use super::execution_log::ExecutionLog;
use crate::constraint_theory::dna::{ConstraintDna, ConstraintCategory, ConstraintValue};

// ============================================================
// Fiber Report
// ============================================================

/// Per-fiber analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberReport {
    /// Fiber type name.
    pub fiber_type: String,
    /// Number of samples analyzed.
    pub sample_count: usize,
    /// Success rate (0.0-1.0).
    pub success_rate: f64,
    /// Average latency in microseconds.
    pub avg_latency_us: f64,
    /// P99 latency in microseconds.
    pub p99_latency_us: f64,
    /// Minimum latency observed.
    pub min_latency_us: f64,
    /// Maximum latency observed.
    pub max_latency_us: f64,
    /// Average registers used.
    pub avg_registers: f64,
    /// Average shared memory used (bytes).
    pub avg_shared_memory: f64,
    /// Average coalescing ratio.
    pub avg_coalescing_ratio: f64,
    /// Average warp occupancy.
    pub avg_warp_occupancy: f64,
    /// Whether latency is trending up (degrading).
    pub latency_trending_up: bool,
    /// Headroom to P99 constraint (positive = within budget).
    pub p99_headroom_us: f64,
}

// ============================================================
// Mutation Recommendation
// ============================================================

/// A recommended mutation to a constraint in the DNA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecommendation {
    /// Constraint ID to mutate.
    pub constraint_id: String,
    /// What to do.
    pub action: MutationAction,
    /// New proposed value.
    pub proposed_value: ConstraintValue,
    /// Why this mutation is recommended.
    pub reason: String,
    /// Confidence in this recommendation (0.0-1.0).
    pub confidence: f64,
    /// Which fiber(s) drove this recommendation.
    pub source_fibers: Vec<String>,
}

/// Mutation action type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationAction {
    /// Tighten the constraint (make it stricter).
    Tighten,
    /// Relax the constraint (make it more lenient).
    Relax,
    /// No change needed.
    Hold,
}

// ============================================================
// Analysis Report
// ============================================================

/// Full analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Per-fiber reports.
    pub fiber_reports: Vec<FiberReport>,
    /// Mutation recommendations.
    pub recommendations: Vec<MutationRecommendation>,
    /// Total samples analyzed.
    pub total_samples: usize,
    /// DNA version analyzed against.
    pub dna_version: String,
}

// ============================================================
// Success Analyzer
// ============================================================

/// Analyzes execution logs and recommends DNA mutations.
pub struct SuccessAnalyzer {
    dna: ConstraintDna,
}

impl SuccessAnalyzer {
    /// Create a new analyzer with the current DNA.
    pub fn new(dna: ConstraintDna) -> Self {
        SuccessAnalyzer { dna }
    }

    /// Analyze the execution log and produce a report.
    pub fn analyze(&self, log: &ExecutionLog) -> AnalysisReport {
        let fiber_types = log.fiber_types();
        let mut fiber_reports = Vec::new();

        for ft in &fiber_types {
            let entries = log.entries_for_fiber(ft);
            if entries.is_empty() {
                continue;
            }

            let report = self.analyze_fiber(ft, &entries);
            fiber_reports.push(report);
        }

        let recommendations = self.generate_recommendations(&fiber_reports);

        let dna_version = format!(
            "{}.{}.{}",
            self.dna.version.major, self.dna.version.minor, self.dna.version.patch
        );

        AnalysisReport {
            fiber_reports,
            recommendations,
            total_samples: log.len(),
            dna_version,
        }
    }

    /// Analyze a single fiber type.
    fn analyze_fiber(
        &self,
        fiber_type: &str,
        entries: &[&super::execution_log::ExecutionEntry],
    ) -> FiberReport {
        let n = entries.len();

        // Success rate.
        let successes = entries.iter().filter(|e| e.success).count();
        let success_rate = successes as f64 / n as f64;

        // Latency stats.
        let mut latencies: Vec<f64> = entries.iter().map(|e| e.execution_time_us).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let total_lat: f64 = latencies.iter().sum();
        let avg_lat = total_lat / n as f64;
        let p99_idx = ((n as f64) * 0.99).ceil() as usize - 1;
        let p99_lat = latencies[p99_idx.min(n - 1)];
        let min_lat = latencies[0];
        let max_lat = latencies[n - 1];

        // Resource averages.
        let avg_regs = entries.iter().map(|e| e.registers_used as f64).sum::<f64>() / n as f64;
        let avg_shmem = entries.iter().map(|e| e.shared_memory_used as f64).sum::<f64>() / n as f64;
        let avg_coal = entries.iter().map(|e| e.coalescing_ratio).sum::<f64>() / n as f64;
        let avg_occ = entries.iter().map(|e| e.warp_occupancy).sum::<f64>() / n as f64;

        // Trend detection: compare first-half avg to second-half avg.
        let mid = n / 2;
        let first_half_avg = if mid > 0 {
            latencies[..mid].iter().sum::<f64>() / mid as f64
        } else {
            avg_lat
        };
        let second_half_avg = if n - mid > 0 {
            latencies[mid..].iter().sum::<f64>() / (n - mid) as f64
        } else {
            avg_lat
        };
        let trending_up = second_half_avg > first_half_avg * 1.1; // 10% threshold.

        // Headroom to P99 constraint.
        let p99_ceiling = self
            .dna
            .get("latency.p99_rtt_ceiling")
            .and_then(|c| match &c.value {
                ConstraintValue::FloatMax(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(8.0);
        let headroom = p99_ceiling - p99_lat;

        FiberReport {
            fiber_type: fiber_type.into(),
            sample_count: n,
            success_rate,
            avg_latency_us: avg_lat,
            p99_latency_us: p99_lat,
            min_latency_us: min_lat,
            max_latency_us: max_lat,
            avg_registers: avg_regs,
            avg_shared_memory: avg_shmem,
            avg_coalescing_ratio: avg_coal,
            avg_warp_occupancy: avg_occ,
            latency_trending_up: trending_up,
            p99_headroom_us: headroom,
        }
    }

    /// Generate mutation recommendations based on fiber reports.
    fn generate_recommendations(&self, reports: &[FiberReport]) -> Vec<MutationRecommendation> {
        let mut recs = Vec::new();

        // ── P99 latency ceiling ──
        let worst_p99 = reports.iter().map(|r| r.p99_latency_us).fold(0.0f64, f64::max);
        let best_p99 = reports
            .iter()
            .map(|r| r.p99_latency_us)
            .fold(f64::MAX, f64::min);

        if let Some(constraint) = self.dna.get("latency.p99_rtt_ceiling") {
            if let ConstraintValue::FloatMax(ceiling) = &constraint.value {
                if worst_p99 > *ceiling {
                    // Some fibers violate the P99 ceiling → relax or investigate.
                    let violators: Vec<String> = reports
                        .iter()
                        .filter(|r| r.p99_latency_us > *ceiling)
                        .map(|r| r.fiber_type.clone())
                        .collect();

                    recs.push(MutationRecommendation {
                        constraint_id: "latency.p99_rtt_ceiling".into(),
                        action: MutationAction::Relax,
                        proposed_value: ConstraintValue::FloatMax(worst_p99 * 1.2),
                        reason: format!(
                            "Fibers {:?} exceed P99 ceiling of {:.1}µs (worst: {:.1}µs). \
                             Relaxing to {:.1}µs.",
                            violators, ceiling, worst_p99, worst_p99 * 1.2
                        ),
                        confidence: 0.7,
                        source_fibers: violators,
                    });
                } else if best_p99 < *ceiling * 0.5 {
                    // All fibers are well within budget → tighten.
                    let all_fibers: Vec<String> =
                        reports.iter().map(|r| r.fiber_type.clone()).collect();
                    let new_ceiling = worst_p99 * 1.3; // 30% headroom.

                    recs.push(MutationRecommendation {
                        constraint_id: "latency.p99_rtt_ceiling".into(),
                        action: MutationAction::Tighten,
                        proposed_value: ConstraintValue::FloatMax(new_ceiling),
                        reason: format!(
                            "All fibers well within P99 budget (worst: {:.1}µs vs ceiling {:.1}µs). \
                             Tightening to {:.1}µs to prevent regression.",
                            worst_p99, ceiling, new_ceiling
                        ),
                        confidence: 0.8,
                        source_fibers: all_fibers,
                    });
                }
            }
        }

        // ── Coalescing ratio ──
        let avg_coal = if reports.is_empty() {
            1.0
        } else {
            reports.iter().map(|r| r.avg_coalescing_ratio).sum::<f64>() / reports.len() as f64
        };

        if let Some(constraint) = self.dna.get("efficiency.min_coalescing_ratio") {
            if let ConstraintValue::FloatMin(floor) = &constraint.value {
                if avg_coal > *floor * 1.2 {
                    // Coalescing is excellent → tighten.
                    let new_floor = avg_coal * 0.9;
                    recs.push(MutationRecommendation {
                        constraint_id: "efficiency.min_coalescing_ratio".into(),
                        action: MutationAction::Tighten,
                        proposed_value: ConstraintValue::FloatMin(new_floor),
                        reason: format!(
                            "Average coalescing ratio {:.2} exceeds floor {:.2} by >20%. \
                             Tightening to {:.2}.",
                            avg_coal, floor, new_floor
                        ),
                        confidence: 0.6,
                        source_fibers: reports.iter().map(|r| r.fiber_type.clone()).collect(),
                    });
                }
            }
        }

        // ── Warp occupancy ──
        let low_occ_fibers: Vec<&FiberReport> =
            reports.iter().filter(|r| r.avg_warp_occupancy < 0.25).collect();
        if !low_occ_fibers.is_empty() {
            recs.push(MutationRecommendation {
                constraint_id: "efficiency.min_warp_occupancy".into(),
                action: MutationAction::Relax,
                proposed_value: ConstraintValue::FloatMin(0.15),
                reason: format!(
                    "Fibers {:?} have very low warp occupancy (<25%). \
                     Relaxing floor to accommodate specialized kernels.",
                    low_occ_fibers.iter().map(|r| &r.fiber_type).collect::<Vec<_>>()
                ),
                confidence: 0.5,
                source_fibers: low_occ_fibers.iter().map(|r| r.fiber_type.clone()).collect(),
            });
        }

        // ── Register budget ──
        let max_regs = reports.iter().map(|r| r.avg_registers).fold(0.0f64, f64::max);
        if let Some(constraint) = self.dna.get("resource.register_budget") {
            if let ConstraintValue::IntMax(budget) = &constraint.value {
                if max_regs < (*budget as f64) * 0.3 {
                    // Using < 30% of budget → tighten.
                    let new_budget = (max_regs * 2.0) as u64;
                    recs.push(MutationRecommendation {
                        constraint_id: "resource.register_budget".into(),
                        action: MutationAction::Tighten,
                        proposed_value: ConstraintValue::IntMax(new_budget),
                        reason: format!(
                            "Max register usage ({:.0}) is <30% of budget ({}). \
                             Tightening to {} to detect regressions.",
                            max_regs, budget, new_budget
                        ),
                        confidence: 0.5,
                        source_fibers: reports.iter().map(|r| r.fiber_type.clone()).collect(),
                    });
                }
            }
        }

        recs
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::execution_log::{ExecutionEntry, ExecutionLog};

    fn make_entry(fiber: &str, latency: f64, success: bool) -> ExecutionEntry {
        ExecutionEntry {
            agent_id: "test".into(),
            fiber_type: fiber.into(),
            execution_time_us: latency,
            registers_used: 24,
            shared_memory_used: 2048,
            coalescing_ratio: 0.9,
            warp_occupancy: 0.7,
            success,
            timestamp_epoch: 1000,
        }
    }

    #[test]
    fn test_analyzer_basic() {
        let mut log = ExecutionLog::new(1000);
        for i in 0..100 {
            log.record(make_entry("cell_update", 2.0 + (i as f64 * 0.05), true));
        }

        let dna = ConstraintDna::default_system_dna();
        let analyzer = SuccessAnalyzer::new(dna);
        let report = analyzer.analyze(&log);

        assert_eq!(report.fiber_reports.len(), 1);
        assert_eq!(report.total_samples, 100);
        assert!(!report.fiber_reports[0].fiber_type.is_empty());
    }

    #[test]
    fn test_p99_violation_detection() {
        let mut log = ExecutionLog::new(1000);
        // Insert records with some P99 > 8µs.
        for i in 0..100 {
            let lat = if i >= 98 { 10.0 } else { 3.0 };
            log.record(make_entry("formula_eval", lat, true));
        }

        let dna = ConstraintDna::default_system_dna();
        let analyzer = SuccessAnalyzer::new(dna);
        let report = analyzer.analyze(&log);

        let p99_rec = report
            .recommendations
            .iter()
            .find(|r| r.constraint_id == "latency.p99_rtt_ceiling");
        assert!(p99_rec.is_some());
    }

    #[test]
    fn test_tighten_recommendation() {
        let mut log = ExecutionLog::new(1000);
        // All latencies well below 8µs.
        for _ in 0..100 {
            log.record(make_entry("cell_update", 1.5, true));
        }

        let dna = ConstraintDna::default_system_dna();
        let analyzer = SuccessAnalyzer::new(dna);
        let report = analyzer.analyze(&log);

        let tighten = report
            .recommendations
            .iter()
            .find(|r| matches!(r.action, MutationAction::Tighten) && r.constraint_id == "latency.p99_rtt_ceiling");
        assert!(tighten.is_some());
    }

    #[test]
    fn test_empty_log() {
        let log = ExecutionLog::new(100);
        let dna = ConstraintDna::default_system_dna();
        let analyzer = SuccessAnalyzer::new(dna);
        let report = analyzer.analyze(&log);

        assert!(report.fiber_reports.is_empty());
        assert!(report.recommendations.is_empty());
    }

    #[test]
    fn test_multi_fiber_analysis() {
        let mut log = ExecutionLog::new(1000);
        for _ in 0..50 {
            log.record(make_entry("cell_update", 2.0, true));
        }
        for _ in 0..50 {
            log.record(make_entry("crdt_merge", 5.0, true));
        }

        let dna = ConstraintDna::default_system_dna();
        let analyzer = SuccessAnalyzer::new(dna);
        let report = analyzer.analyze(&log);

        assert_eq!(report.fiber_reports.len(), 2);
    }
}
