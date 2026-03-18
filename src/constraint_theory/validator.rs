// ============================================================
// Constraint Validator — Operation-Level Constraint Checking
// ============================================================
//
// The validator evaluates a proposed operation against the full
// Constraint DNA and returns a structured verdict. Every
// cudaclaw operation must pass through this gate.
//
// ============================================================

use serde::{Deserialize, Serialize};

use super::dna::{
    ConstraintCategory, ConstraintDna, ConstraintSeverity, ConstraintValue, SuperConstraint,
};

// ============================================================
// Operation Context
// ============================================================

/// Context describing the operation being validated.
/// The validator checks each super-constraint against these fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationContext {
    /// Agent performing the operation.
    pub agent_id: String,
    /// Name of the operation (e.g., "set_cell", "batch_update").
    pub operation_name: String,
    /// Target SM index (for resource constraints).
    pub sm_index: u32,

    // Resource fields
    /// 32-bit registers needed by this operation.
    pub registers_needed: u64,
    /// Shared memory bytes needed.
    pub shared_memory_needed: u64,
    /// Warp slots needed.
    pub warps_needed: u64,
    /// Thread count needed.
    pub threads_needed: u64,

    // Latency fields
    /// Estimated latency in microseconds.
    pub estimated_latency_us: f64,

    // Correctness fields
    /// Whether this is a CRDT write (needs monotonicity check).
    pub is_crdt_write: bool,
    /// Timestamp of this operation.
    pub timestamp: u64,
    /// Timestamp of the predecessor operation (if known).
    pub predecessor_timestamp: Option<u64>,

    // Efficiency fields
    /// Fraction of memory transactions that are coalesced (0.0-1.0).
    pub coalescing_ratio: f64,
    /// Fraction of SM warp slots occupied (0.0-1.0).
    pub warp_occupancy: f64,
}

impl Default for OperationContext {
    fn default() -> Self {
        OperationContext {
            agent_id: "unknown".into(),
            operation_name: "unknown".into(),
            sm_index: 0,
            registers_needed: 0,
            shared_memory_needed: 0,
            warps_needed: 0,
            threads_needed: 0,
            estimated_latency_us: 0.0,
            is_crdt_write: false,
            timestamp: 0,
            predecessor_timestamp: None,
            coalescing_ratio: 1.0,
            warp_occupancy: 1.0,
        }
    }
}

// ============================================================
// Verdict
// ============================================================

/// The outcome of checking one constraint against an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictKind {
    /// All critical constraints passed, no warnings.
    Pass,
    /// All critical constraints passed, but some warnings.
    Warn,
    /// At least one critical constraint failed.
    Fail,
}

/// Full validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintVerdict {
    /// Overall verdict.
    pub kind: VerdictKind,
    /// Constraints that passed.
    pub passed: Vec<String>,
    /// Warning messages (from Warning-severity constraints).
    pub warnings: Vec<String>,
    /// Failure messages (from Critical-severity constraints).
    pub failures: Vec<String>,
    /// Info messages (from Info-severity constraints).
    pub info: Vec<String>,
}

impl ConstraintVerdict {
    fn new() -> Self {
        ConstraintVerdict {
            kind: VerdictKind::Pass,
            passed: Vec::new(),
            warnings: Vec::new(),
            failures: Vec::new(),
            info: Vec::new(),
        }
    }

    fn add_pass(&mut self, msg: String) {
        self.passed.push(msg);
    }

    fn add_warning(&mut self, msg: String) {
        self.warnings.push(msg);
        if self.kind == VerdictKind::Pass {
            self.kind = VerdictKind::Warn;
        }
    }

    fn add_failure(&mut self, msg: String) {
        self.failures.push(msg);
        self.kind = VerdictKind::Fail;
    }

    fn add_info(&mut self, msg: String) {
        self.info.push(msg);
    }
}

// ============================================================
// Constraint Validator
// ============================================================

/// Validates operations against the Constraint DNA.
pub struct ConstraintValidator {
    dna: ConstraintDna,
}

impl ConstraintValidator {
    /// Create a new validator with the given DNA.
    pub fn new(dna: ConstraintDna) -> Self {
        ConstraintValidator { dna }
    }

    /// Get a reference to the DNA.
    pub fn dna(&self) -> &ConstraintDna {
        &self.dna
    }

    /// Get a mutable reference to the DNA (for ML mutations).
    pub fn dna_mut(&mut self) -> &mut ConstraintDna {
        &mut self.dna
    }

    /// Validate an operation context against all enabled constraints.
    pub fn validate(&self, ctx: &OperationContext) -> ConstraintVerdict {
        let mut verdict = ConstraintVerdict::new();

        for constraint in &self.dna.constraints {
            if !constraint.enabled {
                continue;
            }

            let (satisfied, detail) = self.check_constraint(constraint, ctx);

            let msg = format!(
                "[{}] {} — {} (actual: {})",
                constraint.id, constraint.name, constraint.value.describe(), detail
            );

            if satisfied {
                verdict.add_pass(msg);
            } else {
                match constraint.severity {
                    ConstraintSeverity::Critical => verdict.add_failure(msg),
                    ConstraintSeverity::Warning => verdict.add_warning(msg),
                    ConstraintSeverity::Info => verdict.add_info(msg),
                }
            }
        }

        verdict
    }

    /// Check a single constraint against the operation context.
    /// Returns (satisfied, actual_value_description).
    fn check_constraint(
        &self,
        constraint: &SuperConstraint,
        ctx: &OperationContext,
    ) -> (bool, String) {
        match constraint.id.as_str() {
            // ── Resource ──
            "resource.register_budget" => {
                let actual = ctx.registers_needed as f64;
                (constraint.value.check_f64(actual), format!("{}", ctx.registers_needed))
            }
            "resource.shared_memory_ceiling" => {
                let actual = ctx.shared_memory_needed as f64;
                (constraint.value.check_f64(actual), format!("{} bytes", ctx.shared_memory_needed))
            }
            "resource.warp_slot_limit" => {
                let actual = ctx.warps_needed as f64;
                (constraint.value.check_f64(actual), format!("{}", ctx.warps_needed))
            }
            "resource.thread_budget" => {
                let actual = ctx.threads_needed as f64;
                (constraint.value.check_f64(actual), format!("{}", ctx.threads_needed))
            }

            // ── Latency ──
            "latency.p99_rtt_ceiling" => {
                let actual = ctx.estimated_latency_us;
                (constraint.value.check_f64(actual), format!("{:.2} µs", actual))
            }
            "latency.push_latency_ceiling" => {
                // Push latency is a fraction of total RTT; approximate as 25%.
                let push_est = ctx.estimated_latency_us * 0.25;
                (constraint.value.check_f64(push_est), format!("{:.2} µs (est)", push_est))
            }
            "latency.pcie_transfer_budget" => {
                // PCIe is another fraction; approximate as 40%.
                let pcie_est = ctx.estimated_latency_us * 0.4;
                (constraint.value.check_f64(pcie_est), format!("{:.2} µs (est)", pcie_est))
            }

            // ── Correctness ──
            "correctness.crdt_monotonicity" => {
                if !ctx.is_crdt_write {
                    return (true, "not a CRDT write".into());
                }
                // Monotonicity: timestamp must be > predecessor.
                let ok = match ctx.predecessor_timestamp {
                    Some(pred) => ctx.timestamp > pred,
                    None => true, // First write is always valid.
                };
                (ok, format!(
                    "ts={}, pred={:?}",
                    ctx.timestamp, ctx.predecessor_timestamp
                ))
            }
            "correctness.timestamp_ordering" => {
                let ok = match ctx.predecessor_timestamp {
                    Some(pred) => ctx.timestamp > pred,
                    None => true,
                };
                (ok, format!(
                    "ts={}, pred={:?}",
                    ctx.timestamp, ctx.predecessor_timestamp
                ))
            }

            // ── Efficiency ──
            "efficiency.min_warp_occupancy" => {
                (constraint.value.check_f64(ctx.warp_occupancy),
                 format!("{:.2}", ctx.warp_occupancy))
            }
            "efficiency.min_coalescing_ratio" => {
                (constraint.value.check_f64(ctx.coalescing_ratio),
                 format!("{:.2}", ctx.coalescing_ratio))
            }
            "efficiency.idle_power_ceiling" => {
                // We don't have live power data; always pass.
                (true, "no live data".into())
            }

            // ── Biological ──
            "biological.nutrient_floor" | "biological.prune_cooldown_ms" | "biological.branch_hysteresis_ms" => {
                // These are evaluated by the Ramify engine, not per-operation.
                (true, "evaluated by Ramify engine".into())
            }

            // Unknown constraint — pass by default.
            _ => (true, "unknown constraint, skipped".into()),
        }
    }

    /// Quick check: does the operation pass all critical constraints?
    pub fn passes_critical(&self, ctx: &OperationContext) -> bool {
        let verdict = self.validate(ctx);
        verdict.kind != VerdictKind::Fail
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_validator() -> ConstraintValidator {
        ConstraintValidator::new(ConstraintDna::default_system_dna())
    }

    fn make_valid_ctx() -> OperationContext {
        OperationContext {
            agent_id: "test_agent".into(),
            operation_name: "set_cell".into(),
            sm_index: 0,
            registers_needed: 8192,
            shared_memory_needed: 4096,
            warps_needed: 4,
            threads_needed: 128,
            estimated_latency_us: 3.0,
            is_crdt_write: true,
            timestamp: 100,
            predecessor_timestamp: Some(99),
            coalescing_ratio: 0.95,
            warp_occupancy: 0.65,
        }
    }

    #[test]
    fn test_valid_operation_passes() {
        let v = make_validator();
        let ctx = make_valid_ctx();
        let verdict = v.validate(&ctx);
        assert_eq!(verdict.kind, VerdictKind::Pass);
        assert!(verdict.failures.is_empty());
    }

    #[test]
    fn test_register_budget_violation() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.registers_needed = 70000; // Exceeds 32768
        let verdict = v.validate(&ctx);
        assert_eq!(verdict.kind, VerdictKind::Fail);
        assert!(verdict.failures.iter().any(|f| f.contains("register_budget")));
    }

    #[test]
    fn test_shared_memory_violation() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.shared_memory_needed = 200000; // Exceeds 49152
        let verdict = v.validate(&ctx);
        assert_eq!(verdict.kind, VerdictKind::Fail);
    }

    #[test]
    fn test_latency_violation() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.estimated_latency_us = 15.0; // Exceeds 8.0
        let verdict = v.validate(&ctx);
        assert_eq!(verdict.kind, VerdictKind::Fail);
        assert!(verdict.failures.iter().any(|f| f.contains("p99_rtt_ceiling")));
    }

    #[test]
    fn test_crdt_monotonicity_violation() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.is_crdt_write = true;
        ctx.timestamp = 50;
        ctx.predecessor_timestamp = Some(100); // 50 < 100 → violation
        let verdict = v.validate(&ctx);
        assert_eq!(verdict.kind, VerdictKind::Fail);
        assert!(verdict.failures.iter().any(|f| f.contains("crdt_monotonicity")));
    }

    #[test]
    fn test_coalescing_warning() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.coalescing_ratio = 0.3; // Below 0.75
        let verdict = v.validate(&ctx);
        // Coalescing is Warning severity, not Critical.
        assert!(verdict.warnings.iter().any(|w| w.contains("coalescing_ratio")));
    }

    #[test]
    fn test_passes_critical_helper() {
        let v = make_validator();
        let ctx = make_valid_ctx();
        assert!(v.passes_critical(&ctx));

        let mut bad_ctx = make_valid_ctx();
        bad_ctx.registers_needed = 70000;
        assert!(!v.passes_critical(&bad_ctx));
    }

    #[test]
    fn test_non_crdt_write_skips_monotonicity() {
        let v = make_validator();
        let mut ctx = make_valid_ctx();
        ctx.is_crdt_write = false;
        ctx.timestamp = 1;
        ctx.predecessor_timestamp = Some(100);
        // Even though timestamps are "wrong", it's not a CRDT write.
        let verdict = v.validate(&ctx);
        // timestamp_ordering still catches this
        assert!(verdict.failures.iter().any(|f| f.contains("timestamp_ordering")));
    }
}
