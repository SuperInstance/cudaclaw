// ============================================================
// DNA Mutator — Applies Recommended Mutations to Constraint DNA
// ============================================================
//
// Takes mutation recommendations from the SuccessAnalyzer and
// applies them to the Constraint DNA. Each mutation is validated,
// applied, and logged for auditability.
//
// SAFETY RAILS:
//   - Critical constraints can only be relaxed by up to 50%.
//   - Constraints cannot be tightened below observed minimums.
//   - Each mutation increments the DNA version.
//   - Mutation history is preserved for rollback.
//
// ============================================================

use serde::{Deserialize, Serialize};

use super::success_analyzer::{MutationAction, MutationRecommendation};
use crate::constraint_theory::dna::{ConstraintDna, ConstraintSeverity, ConstraintValue};

// ============================================================
// Mutation Result
// ============================================================

/// Result of applying a single mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationResult {
    /// Constraint that was (or was not) mutated.
    pub constraint_id: String,
    /// Whether the mutation was applied.
    pub applied: bool,
    /// Human-readable description of old value.
    pub old_value_desc: String,
    /// Human-readable description of new value.
    pub new_value_desc: String,
    /// Reason for the result.
    pub reason: String,
}

// ============================================================
// DNA Mutator
// ============================================================

/// Applies recommended mutations to the Constraint DNA.
pub struct DnaMutator {
    dna: ConstraintDna,
    history: Vec<MutationResult>,
}

impl DnaMutator {
    /// Create a new mutator with the given DNA.
    pub fn new(dna: ConstraintDna) -> Self {
        DnaMutator {
            dna,
            history: Vec::new(),
        }
    }

    /// Get a reference to the current DNA.
    pub fn dna(&self) -> &ConstraintDna {
        &self.dna
    }

    /// Consume the mutator and return the mutated DNA.
    pub fn into_dna(self) -> ConstraintDna {
        self.dna
    }

    /// Get the mutation history.
    pub fn history(&self) -> &[MutationResult] {
        &self.history
    }

    /// Apply a list of recommendations and return results.
    pub fn apply_recommendations(
        &mut self,
        recommendations: &[MutationRecommendation],
    ) -> Vec<MutationResult> {
        let mut results = Vec::new();

        for rec in recommendations {
            let result = self.apply_one(rec);
            self.history.push(result.clone());
            results.push(result);
        }

        results
    }

    /// Apply a single recommendation.
    fn apply_one(&mut self, rec: &MutationRecommendation) -> MutationResult {
        // Look up the constraint.
        let constraint = match self.dna.get(&rec.constraint_id) {
            Some(c) => c.clone(),
            None => {
                return MutationResult {
                    constraint_id: rec.constraint_id.clone(),
                    applied: false,
                    old_value_desc: "N/A".into(),
                    new_value_desc: rec.proposed_value.describe(),
                    reason: format!("Constraint '{}' not found in DNA", rec.constraint_id),
                };
            }
        };

        // Check if the constraint is enabled.
        if !constraint.enabled {
            return MutationResult {
                constraint_id: rec.constraint_id.clone(),
                applied: false,
                old_value_desc: constraint.value.describe(),
                new_value_desc: rec.proposed_value.describe(),
                reason: "Constraint is disabled".into(),
            };
        }

        // Safety check: don't relax critical constraints by more than 50%.
        if constraint.severity == ConstraintSeverity::Critical {
            if let MutationAction::Relax = rec.action {
                if !self.is_safe_relax(&constraint.value, &rec.proposed_value) {
                    return MutationResult {
                        constraint_id: rec.constraint_id.clone(),
                        applied: false,
                        old_value_desc: constraint.value.describe(),
                        new_value_desc: rec.proposed_value.describe(),
                        reason: "Relaxation exceeds 50% safety limit for critical constraint".into(),
                    };
                }
            }
        }

        // Check confidence threshold.
        if rec.confidence < 0.3 {
            return MutationResult {
                constraint_id: rec.constraint_id.clone(),
                applied: false,
                old_value_desc: constraint.value.describe(),
                new_value_desc: rec.proposed_value.describe(),
                reason: format!("Confidence {:.0}% below 30% threshold", rec.confidence * 100.0),
            };
        }

        // Apply the mutation.
        let old_desc = constraint.value.describe();
        let new_desc = rec.proposed_value.describe();

        self.dna.mutate_constraint(&rec.constraint_id, rec.proposed_value.clone());

        MutationResult {
            constraint_id: rec.constraint_id.clone(),
            applied: true,
            old_value_desc: old_desc,
            new_value_desc: new_desc,
            reason: rec.reason.clone(),
        }
    }

    /// Check if a relaxation is within the 50% safety limit.
    fn is_safe_relax(&self, old: &ConstraintValue, new: &ConstraintValue) -> bool {
        match (old, new) {
            (ConstraintValue::IntMax(old_max), ConstraintValue::IntMax(new_max)) => {
                *new_max <= (*old_max as f64 * 1.5) as u64
            }
            (ConstraintValue::FloatMax(old_max), ConstraintValue::FloatMax(new_max)) => {
                *new_max <= *old_max * 1.5
            }
            (ConstraintValue::IntMin(old_min), ConstraintValue::IntMin(new_min)) => {
                *new_min >= (*old_min as f64 * 0.5) as u64
            }
            (ConstraintValue::FloatMin(old_min), ConstraintValue::FloatMin(new_min)) => {
                *new_min >= *old_min * 0.5
            }
            // For other types, allow the mutation.
            _ => true,
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
    fn test_mutator_creation() {
        let dna = ConstraintDna::default_system_dna();
        let mutator = DnaMutator::new(dna);
        assert!(mutator.history().is_empty());
    }

    #[test]
    fn test_apply_tighten() {
        let dna = ConstraintDna::default_system_dna();
        let mut mutator = DnaMutator::new(dna);

        let rec = MutationRecommendation {
            constraint_id: "latency.p99_rtt_ceiling".into(),
            action: MutationAction::Tighten,
            proposed_value: ConstraintValue::FloatMax(6.0),
            reason: "Test tighten".into(),
            confidence: 0.8,
            source_fibers: vec!["cell_update".into()],
        };

        let results = mutator.apply_recommendations(&[rec]);
        assert_eq!(results.len(), 1);
        assert!(results[0].applied);

        // Verify the DNA was updated.
        let c = mutator.dna().get("latency.p99_rtt_ceiling").unwrap();
        assert!(c.value.check_f64(5.0));
        assert!(!c.value.check_f64(7.0));
    }

    #[test]
    fn test_safe_relax_critical() {
        let dna = ConstraintDna::default_system_dna();
        let mut mutator = DnaMutator::new(dna);

        // Try to relax P99 ceiling from 8.0 to 11.0 (37.5% increase, within 50%).
        let rec = MutationRecommendation {
            constraint_id: "latency.p99_rtt_ceiling".into(),
            action: MutationAction::Relax,
            proposed_value: ConstraintValue::FloatMax(11.0),
            reason: "Test relax".into(),
            confidence: 0.7,
            source_fibers: vec!["formula_eval".into()],
        };

        let results = mutator.apply_recommendations(&[rec]);
        assert!(results[0].applied);
    }

    #[test]
    fn test_unsafe_relax_rejected() {
        let dna = ConstraintDna::default_system_dna();
        let mut mutator = DnaMutator::new(dna);

        // Try to relax P99 ceiling from 8.0 to 20.0 (150% increase, exceeds 50%).
        let rec = MutationRecommendation {
            constraint_id: "latency.p99_rtt_ceiling".into(),
            action: MutationAction::Relax,
            proposed_value: ConstraintValue::FloatMax(20.0),
            reason: "Test unsafe relax".into(),
            confidence: 0.9,
            source_fibers: vec!["formula_eval".into()],
        };

        let results = mutator.apply_recommendations(&[rec]);
        assert!(!results[0].applied);
        assert!(results[0].reason.contains("safety limit"));
    }

    #[test]
    fn test_low_confidence_rejected() {
        let dna = ConstraintDna::default_system_dna();
        let mut mutator = DnaMutator::new(dna);

        let rec = MutationRecommendation {
            constraint_id: "latency.p99_rtt_ceiling".into(),
            action: MutationAction::Tighten,
            proposed_value: ConstraintValue::FloatMax(5.0),
            reason: "Low confidence".into(),
            confidence: 0.1, // Below 30% threshold.
            source_fibers: vec!["test".into()],
        };

        let results = mutator.apply_recommendations(&[rec]);
        assert!(!results[0].applied);
        assert!(results[0].reason.contains("threshold"));
    }

    #[test]
    fn test_unknown_constraint() {
        let dna = ConstraintDna::default_system_dna();
        let mut mutator = DnaMutator::new(dna);

        let rec = MutationRecommendation {
            constraint_id: "nonexistent.constraint".into(),
            action: MutationAction::Tighten,
            proposed_value: ConstraintValue::FloatMax(5.0),
            reason: "Test".into(),
            confidence: 0.8,
            source_fibers: vec!["test".into()],
        };

        let results = mutator.apply_recommendations(&[rec]);
        assert!(!results[0].applied);
        assert!(results[0].reason.contains("not found"));
    }

    #[test]
    fn test_version_increments() {
        let dna = ConstraintDna::default_system_dna();
        let initial_patch = dna.version.patch;
        let mut mutator = DnaMutator::new(dna);

        let rec = MutationRecommendation {
            constraint_id: "latency.p99_rtt_ceiling".into(),
            action: MutationAction::Tighten,
            proposed_value: ConstraintValue::FloatMax(6.0),
            reason: "Test".into(),
            confidence: 0.8,
            source_fibers: vec!["test".into()],
        };

        mutator.apply_recommendations(&[rec]);
        assert_eq!(mutator.dna().version.patch, initial_patch + 1);
        assert_eq!(mutator.dna().total_mutations, 1);
    }

    #[test]
    fn test_into_dna() {
        let dna = ConstraintDna::default_system_dna();
        let mutator = DnaMutator::new(dna);
        let recovered = mutator.into_dna();
        assert!(recovered.constraints.len() > 0);
    }
}
