// ============================================================
// Constraint DNA — The Fundamental Invariants of cudaclaw
// ============================================================
//
// Every super-constraint is a named rule that encodes a system
// invariant. The DNA is the complete set of these rules. It is:
//   - Versioned (so ML mutations are traceable)
//   - Serializable (JSON for persistence / LLM context)
//   - Self-validating (internal consistency checks)
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================
// Constraint Value Types
// ============================================================

/// The typed value carried by a constraint bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintValue {
    /// Integer bound (e.g., register count, thread count).
    IntMax(u64),
    /// Integer minimum (e.g., minimum occupancy threads).
    IntMin(u64),
    /// Float ceiling (e.g., latency µs, power watts).
    FloatMax(f64),
    /// Float floor (e.g., minimum coalescing ratio).
    FloatMin(f64),
    /// Boolean invariant (e.g., CRDT monotonicity must be true).
    BoolRequired(bool),
    /// Range bound (inclusive min, inclusive max).
    Range { min: f64, max: f64 },
}

impl ConstraintValue {
    /// Check whether a given numeric value satisfies this constraint.
    pub fn check_f64(&self, actual: f64) -> bool {
        match self {
            ConstraintValue::IntMax(max) => actual <= *max as f64,
            ConstraintValue::IntMin(min) => actual >= *min as f64,
            ConstraintValue::FloatMax(max) => actual <= *max,
            ConstraintValue::FloatMin(min) => actual >= *min,
            ConstraintValue::BoolRequired(req) => {
                // Treat non-zero as true.
                (actual != 0.0) == *req
            }
            ConstraintValue::Range { min, max } => actual >= *min && actual <= *max,
        }
    }

    /// Human-readable description of the bound.
    pub fn describe(&self) -> String {
        match self {
            ConstraintValue::IntMax(v) => format!("<= {}", v),
            ConstraintValue::IntMin(v) => format!(">= {}", v),
            ConstraintValue::FloatMax(v) => format!("<= {:.2}", v),
            ConstraintValue::FloatMin(v) => format!(">= {:.2}", v),
            ConstraintValue::BoolRequired(v) => format!("must be {}", v),
            ConstraintValue::Range { min, max } => format!("[{:.2}, {:.2}]", min, max),
        }
    }
}

// ============================================================
// Constraint Categories
// ============================================================

/// Category of a super-constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintCategory {
    /// GPU register / shared memory / warp / thread budgets.
    Resource,
    /// P99 RTT, push latency, PCIe transfer budget.
    Latency,
    /// CRDT monotonicity, timestamp ordering.
    Correctness,
    /// Warp occupancy, coalescing ratio, power ceiling.
    Efficiency,
    /// Nutrient floor, prune cooldown, branch hysteresis.
    Biological,
}

// ============================================================
// Constraint Severity
// ============================================================

/// How severe a violation of this constraint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    /// The operation MUST satisfy this constraint. Violation
    /// causes the operation to be rejected.
    Critical,
    /// Violation emits a warning but does not reject.
    Warning,
    /// Informational only — logged but not enforced.
    Info,
}

// ============================================================
// Super-Constraint
// ============================================================

/// A single super-constraint — one gene in the DNA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperConstraint {
    /// Unique identifier (e.g., "resource.register_budget").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Longer description.
    pub description: String,
    /// Category for grouping.
    pub category: ConstraintCategory,
    /// Severity on violation.
    pub severity: ConstraintSeverity,
    /// The constraint bound.
    pub value: ConstraintValue,
    /// Whether this constraint is currently enabled.
    pub enabled: bool,
    /// How many times this constraint has been mutated by ML.
    pub mutation_count: u32,
    /// The epoch timestamp of the last mutation.
    pub last_mutated_epoch: u64,
}

impl SuperConstraint {
    /// Create a new critical constraint.
    pub fn critical(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        category: ConstraintCategory,
        value: ConstraintValue,
    ) -> Self {
        SuperConstraint {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            category,
            severity: ConstraintSeverity::Critical,
            value,
            enabled: true,
            mutation_count: 0,
            last_mutated_epoch: 0,
        }
    }

    /// Create a warning-level constraint.
    pub fn warning(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        category: ConstraintCategory,
        value: ConstraintValue,
    ) -> Self {
        let mut c = Self::critical(id, name, description, category, value);
        c.severity = ConstraintSeverity::Warning;
        c
    }

    /// Create an info-level constraint.
    pub fn info(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        category: ConstraintCategory,
        value: ConstraintValue,
    ) -> Self {
        let mut c = Self::critical(id, name, description, category, value);
        c.severity = ConstraintSeverity::Info;
        c
    }

    /// Mutate this constraint's value (ML feedback loop).
    pub fn mutate(&mut self, new_value: ConstraintValue) {
        self.value = new_value;
        self.mutation_count += 1;
        self.last_mutated_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

// ============================================================
// Constraint DNA
// ============================================================

/// Version of the DNA — incremented on each ML mutation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDnaVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// Epoch timestamp when this version was created.
    pub created_epoch: u64,
}

impl Default for ConstraintDnaVersion {
    fn default() -> Self {
        ConstraintDnaVersion {
            major: 1,
            minor: 0,
            patch: 0,
            created_epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// The complete set of super-constraints — the system's DNA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDna {
    /// Version of this DNA.
    pub version: ConstraintDnaVersion,
    /// All super-constraints.
    pub constraints: Vec<SuperConstraint>,
    /// Total ML mutation passes applied.
    pub total_mutations: u64,
    /// Description of the DNA's purpose.
    pub description: String,
}

impl ConstraintDna {
    /// Create the default system DNA with all standard constraints.
    pub fn default_system_dna() -> Self {
        let mut dna = ConstraintDna {
            version: ConstraintDnaVersion::default(),
            constraints: Vec::new(),
            total_mutations: 0,
            description: "cudaclaw default system DNA — encodes GPU resource, \
                          latency, correctness, efficiency, and biological constraints"
                .into(),
        };

        // ──────────────────────────────────────────────
        // Resource Constraints
        // ──────────────────────────────────────────────
        dna.constraints.push(SuperConstraint::critical(
            "resource.register_budget",
            "Register Budget per Agent",
            "Maximum 32-bit registers an agent may consume on a single SM. \
             Exceeding this starves co-located agents.",
            ConstraintCategory::Resource,
            ConstraintValue::IntMax(32768), // Half of 65536 per SM
        ));

        dna.constraints.push(SuperConstraint::critical(
            "resource.shared_memory_ceiling",
            "Shared Memory Ceiling per Agent",
            "Maximum shared memory (bytes) an agent may request on a single SM.",
            ConstraintCategory::Resource,
            ConstraintValue::IntMax(49152), // 48 KB default
        ));

        dna.constraints.push(SuperConstraint::critical(
            "resource.warp_slot_limit",
            "Warp Slot Limit per Agent",
            "Maximum warp slots an agent may occupy on a single SM.",
            ConstraintCategory::Resource,
            ConstraintValue::IntMax(32),
        ));

        dna.constraints.push(SuperConstraint::critical(
            "resource.thread_budget",
            "Thread Budget per Agent",
            "Maximum threads an agent may launch on a single SM.",
            ConstraintCategory::Resource,
            ConstraintValue::IntMax(2048),
        ));

        // ──────────────────────────────────────────────
        // Latency Constraints
        // ──────────────────────────────────────────────
        dna.constraints.push(SuperConstraint::critical(
            "latency.p99_rtt_ceiling",
            "P99 RTT Ceiling",
            "Maximum P99 round-trip time in microseconds for any single \
             command. Exceeding this violates the system SLA.",
            ConstraintCategory::Latency,
            ConstraintValue::FloatMax(8.0), // <8 µs target
        ));

        dna.constraints.push(SuperConstraint::warning(
            "latency.push_latency_ceiling",
            "Push Latency Ceiling",
            "Maximum host-side push latency in microseconds.",
            ConstraintCategory::Latency,
            ConstraintValue::FloatMax(2.0),
        ));

        dna.constraints.push(SuperConstraint::warning(
            "latency.pcie_transfer_budget",
            "PCIe Transfer Budget",
            "Maximum PCIe delay in microseconds for host-device data movement.",
            ConstraintCategory::Latency,
            ConstraintValue::FloatMax(5.0),
        ));

        // ──────────────────────────────────────────────
        // Correctness Constraints
        // ──────────────────────────────────────────────
        dna.constraints.push(SuperConstraint::critical(
            "correctness.crdt_monotonicity",
            "CRDT Monotonicity",
            "CRDT cell timestamps must be monotonically increasing. \
             A write with a timestamp <= the existing cell's timestamp \
             must be rejected or silently ignored by the merge rule.",
            ConstraintCategory::Correctness,
            ConstraintValue::BoolRequired(true),
        ));

        dna.constraints.push(SuperConstraint::critical(
            "correctness.timestamp_ordering",
            "Timestamp Ordering",
            "Operations must have timestamps > their predecessor. \
             Violated when reordering or replay produces out-of-order writes.",
            ConstraintCategory::Correctness,
            ConstraintValue::BoolRequired(true),
        ));

        // ──────────────────────────────────────────────
        // Efficiency Constraints
        // ──────────────────────────────────────────────
        dna.constraints.push(SuperConstraint::warning(
            "efficiency.min_warp_occupancy",
            "Minimum Warp Occupancy",
            "Minimum fraction of SM warp slots occupied during execution. \
             Low occupancy wastes hardware.",
            ConstraintCategory::Efficiency,
            ConstraintValue::FloatMin(0.25),
        ));

        dna.constraints.push(SuperConstraint::warning(
            "efficiency.min_coalescing_ratio",
            "Minimum Coalescing Ratio",
            "Minimum fraction of memory transactions that are coalesced. \
             Below this threshold, performance degrades by up to 80%.",
            ConstraintCategory::Efficiency,
            ConstraintValue::FloatMin(0.75),
        ));

        dna.constraints.push(SuperConstraint::info(
            "efficiency.idle_power_ceiling",
            "Idle Power Ceiling",
            "Maximum power draw (watts) when the GPU is idle-polling.",
            ConstraintCategory::Efficiency,
            ConstraintValue::FloatMax(25.0),
        ));

        // ──────────────────────────────────────────────
        // Biological Constraints
        // ──────────────────────────────────────────────
        dna.constraints.push(SuperConstraint::warning(
            "biological.nutrient_floor",
            "Nutrient Floor per SM",
            "Minimum nutrient score (1.0 - max resource utilization) \
             for an SM. Below this, agents must be pruned or branched.",
            ConstraintCategory::Biological,
            ConstraintValue::FloatMin(0.15),
        ));

        dna.constraints.push(SuperConstraint::info(
            "biological.prune_cooldown_ms",
            "Prune Cooldown",
            "Minimum milliseconds between successive prune actions \
             on the same agent.",
            ConstraintCategory::Biological,
            ConstraintValue::FloatMin(100.0),
        ));

        dna.constraints.push(SuperConstraint::info(
            "biological.branch_hysteresis_ms",
            "Branch Hysteresis",
            "Minimum milliseconds an agent must run on its current SM \
             before it can be branched to another.",
            ConstraintCategory::Biological,
            ConstraintValue::FloatMin(500.0),
        ));

        dna
    }

    /// Look up a constraint by ID.
    pub fn get(&self, id: &str) -> Option<&SuperConstraint> {
        self.constraints.iter().find(|c| c.id == id)
    }

    /// Look up a mutable constraint by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut SuperConstraint> {
        self.constraints.iter_mut().find(|c| c.id == id)
    }

    /// Get all enabled constraints in a category.
    pub fn by_category(&self, category: ConstraintCategory) -> Vec<&SuperConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.category == category && c.enabled)
            .collect()
    }

    /// Mutate a constraint's value and bump the version.
    pub fn mutate_constraint(&mut self, id: &str, new_value: ConstraintValue) -> bool {
        if let Some(c) = self.get_mut(id) {
            c.mutate(new_value);
            self.total_mutations += 1;
            self.version.patch += 1;
            true
        } else {
            false
        }
    }

    /// Self-validate the DNA for internal consistency.
    pub fn self_validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for duplicate IDs.
        let mut seen = std::collections::HashSet::new();
        for c in &self.constraints {
            if !seen.insert(&c.id) {
                issues.push(format!("Duplicate constraint ID: {}", c.id));
            }
        }

        // Check that critical constraints are enabled.
        for c in &self.constraints {
            if c.severity == ConstraintSeverity::Critical && !c.enabled {
                issues.push(format!(
                    "Critical constraint '{}' is disabled — this is dangerous",
                    c.id
                ));
            }
        }

        // Check that ranges are valid.
        for c in &self.constraints {
            if let ConstraintValue::Range { min, max } = &c.value {
                if min > max {
                    issues.push(format!(
                        "Constraint '{}' has inverted range: [{}, {}]",
                        c.id, min, max
                    ));
                }
            }
        }

        issues
    }

    /// Count enabled constraints.
    pub fn enabled_count(&self) -> usize {
        self.constraints.iter().filter(|c| c.enabled).count()
    }

    /// Get all constraint IDs.
    pub fn constraint_ids(&self) -> Vec<&str> {
        self.constraints.iter().map(|c| c.id.as_str()).collect()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_dna_creation() {
        let dna = ConstraintDna::default_system_dna();
        assert!(dna.constraints.len() >= 14);
        assert_eq!(dna.version.major, 1);
        assert_eq!(dna.total_mutations, 0);
    }

    #[test]
    fn test_constraint_lookup() {
        let dna = ConstraintDna::default_system_dna();
        let c = dna.get("resource.register_budget").unwrap();
        assert_eq!(c.category, ConstraintCategory::Resource);
        assert_eq!(c.severity, ConstraintSeverity::Critical);
    }

    #[test]
    fn test_constraint_value_check() {
        assert!(ConstraintValue::IntMax(100).check_f64(50.0));
        assert!(!ConstraintValue::IntMax(100).check_f64(150.0));
        assert!(ConstraintValue::FloatMin(0.5).check_f64(0.75));
        assert!(!ConstraintValue::FloatMin(0.5).check_f64(0.25));
        assert!(ConstraintValue::BoolRequired(true).check_f64(1.0));
        assert!(!ConstraintValue::BoolRequired(true).check_f64(0.0));
        assert!(ConstraintValue::Range { min: 1.0, max: 10.0 }.check_f64(5.0));
        assert!(!ConstraintValue::Range { min: 1.0, max: 10.0 }.check_f64(15.0));
    }

    #[test]
    fn test_mutation() {
        let mut dna = ConstraintDna::default_system_dna();
        let old_mutations = dna.total_mutations;
        dna.mutate_constraint("resource.register_budget", ConstraintValue::IntMax(40000));
        assert_eq!(dna.total_mutations, old_mutations + 1);
        let c = dna.get("resource.register_budget").unwrap();
        assert_eq!(c.mutation_count, 1);
        assert!(c.value.check_f64(35000.0));
    }

    #[test]
    fn test_self_validate() {
        let dna = ConstraintDna::default_system_dna();
        let issues = dna.self_validate();
        assert!(issues.is_empty(), "Default DNA should be valid: {:?}", issues);
    }

    #[test]
    fn test_by_category() {
        let dna = ConstraintDna::default_system_dna();
        let resource = dna.by_category(ConstraintCategory::Resource);
        assert!(resource.len() >= 4);
        for c in resource {
            assert_eq!(c.category, ConstraintCategory::Resource);
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dna = ConstraintDna::default_system_dna();
        let json = serde_json::to_string(&dna).unwrap();
        let dna2: ConstraintDna = serde_json::from_str(&json).unwrap();
        assert_eq!(dna.constraints.len(), dna2.constraints.len());
        assert_eq!(dna.version.major, dna2.version.major);
    }
}
