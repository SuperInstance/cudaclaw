// ============================================================
// ML Feedback Module — Success Loop & DNA Mutation
// ============================================================
//
// Implements a closed-loop learning system that observes agent
// execution outcomes and mutates the Constraint-Theory DNA to
// improve future performance. This is the "Machine Learn the
// right thing" component.
//
// ARCHITECTURE:
//
//   ┌──────────────┐    execution     ┌──────────────────┐
//   │  Cell Agent   │───records──────►│  ExecutionLog    │
//   │  (GPU Kernel) │                 │  (ring buffer)   │
//   └──────────────┘                  └────────┬─────────┘
//                                              │
//                                     ┌────────▼─────────┐
//                                     │  SuccessAnalyzer  │
//                                     │  (statistics +    │
//                                     │   pattern detect) │
//                                     └────────┬─────────┘
//                                              │
//                                     ┌────────▼─────────┐
//                                     │  DnaMutator       │
//                                     │  (constraint      │
//                                     │   value updates)  │
//                                     └────────┬─────────┘
//                                              │
//                                     ┌────────▼─────────┐
//                                     │  Constraint DNA   │
//                                     │  (updated bounds) │
//                                     └──────────────────┘
//
// FEEDBACK SIGNALS:
//   - Execution time vs. constraint ceiling
//   - Success rate per fiber type
//   - Resource utilization vs. budget
//   - Coalescing ratio trends
//   - Warp occupancy trends
//
// MUTATION STRATEGIES:
//   - Tighten: If all agents comfortably pass a constraint,
//     tighten the bound to prevent regression.
//   - Relax: If too many agents fail a constraint, relax the
//     bound to allow experimentation.
//   - Specialize: If different fibers need different bounds,
//     split a constraint into per-fiber variants.
//
// ============================================================

pub mod execution_log;
pub mod success_analyzer;
pub mod dna_mutator;

use serde::{Deserialize, Serialize};

pub use execution_log::{ExecutionLog, ExecutionEntry, ExecutionSummary};
pub use success_analyzer::{SuccessAnalyzer, AnalysisReport, FiberReport, MutationRecommendation};
pub use dna_mutator::{DnaMutator, MutationResult};
pub use success_analyzer::MutationAction;

// ============================================================
// CLI Integration
// ============================================================

/// Print help for the `feedback` CLI subcommand.
pub fn print_feedback_help() {
    println!("cudaclaw feedback — ML Feedback Loop Management");
    println!();
    println!("USAGE:");
    println!("  cudaclaw feedback [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo                Run a demonstration of the feedback loop");
    println!("  --status              Show current feedback loop status");
    println!("  --analyze             Analyze execution log and suggest mutations");
    println!("  --apply               Apply suggested DNA mutations");
    println!("  --export <PATH>       Export analysis report to JSON");
    println!("  --help, -h            Show this help message");
}

/// Parse CLI arguments for the `feedback` subcommand.
pub fn parse_feedback_args(args: &[String]) -> Option<FeedbackCliAction> {
    if args.is_empty() {
        return Some(FeedbackCliAction::Demo);
    }

    for i in 0..args.len() {
        match args[i].as_str() {
            "--demo" => return Some(FeedbackCliAction::Demo),
            "--status" => return Some(FeedbackCliAction::Status),
            "--analyze" => return Some(FeedbackCliAction::Analyze),
            "--apply" => return Some(FeedbackCliAction::Apply),
            "--export" => {
                if i + 1 < args.len() {
                    return Some(FeedbackCliAction::Export(args[i + 1].clone()));
                }
                return Some(FeedbackCliAction::Demo);
            }
            "--help" | "-h" => return None,
            _ => {}
        }
    }

    Some(FeedbackCliAction::Demo)
}

/// CLI actions for the feedback subcommand.
pub enum FeedbackCliAction {
    Demo,
    Status,
    Analyze,
    Apply,
    Export(String),
}

/// Run the feedback loop demonstration.
pub fn run_demo() {
    use crate::constraint_theory::ConstraintDna;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   cudaclaw ML Feedback — Success Loop Demonstration    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create execution log and populate with simulated data.
    let mut log = ExecutionLog::new(10000);

    println!("━━━ Generating Simulated Execution Records ━━━");

    // Simulate cell_update fiber: mostly fast, occasional spikes.
    for i in 0..500 {
        let latency = 2.0 + (i as f64 % 7.0) * 0.3;
        let success = i % 50 != 0; // 98% success rate
        log.record(ExecutionEntry {
            agent_id: format!("cell_{}_{}", i / 100, i % 100),
            fiber_type: "cell_update".into(),
            execution_time_us: latency,
            registers_used: 24,
            shared_memory_used: 2048,
            coalescing_ratio: 0.92 + (i as f64 % 10.0) * 0.008,
            warp_occupancy: 0.7 + (i as f64 % 5.0) * 0.02,
            success,
            timestamp_epoch: 1000 + i as u64,
        });
    }

    // Simulate crdt_merge fiber: higher latency, more variation.
    for i in 0..200 {
        let latency = 4.0 + (i as f64 % 11.0) * 0.5;
        let success = i % 20 != 0; // 95% success rate
        log.record(ExecutionEntry {
            agent_id: format!("crdt_{}", i),
            fiber_type: "crdt_merge".into(),
            execution_time_us: latency,
            registers_used: 40,
            shared_memory_used: 49152,
            coalescing_ratio: 0.85 + (i as f64 % 8.0) * 0.01,
            warp_occupancy: 0.45 + (i as f64 % 10.0) * 0.01,
            success,
            timestamp_epoch: 1000 + i as u64,
        });
    }

    // Simulate formula_eval fiber: some P99 violations.
    for i in 0..100 {
        let latency = if i % 15 == 0 { 9.5 } else { 5.0 + (i as f64 % 6.0) * 0.3 };
        log.record(ExecutionEntry {
            agent_id: format!("formula_{}", i),
            fiber_type: "formula_eval".into(),
            execution_time_us: latency,
            registers_used: 32,
            shared_memory_used: 16384,
            coalescing_ratio: 0.88,
            warp_occupancy: 0.5,
            success: true,
            timestamp_epoch: 1000 + i as u64,
        });
    }

    let summary = log.summary();
    println!("  Total records: {}", summary.total_entries);
    println!("  Fiber types: {:?}", summary.entries_by_fiber.keys().collect::<Vec<_>>());
    println!("  Overall success rate: {:.1}%", summary.overall_success_rate * 100.0);

    // Run analysis.
    println!("\n━━━ Success Analysis ━━━");
    let dna = ConstraintDna::default_system_dna();
    let analyzer = SuccessAnalyzer::new(dna.clone());
    let report = analyzer.analyze(&log);

    println!("  Fibers analyzed: {}", report.fiber_reports.len());
    for fr in &report.fiber_reports {
        println!(
            "  [{}] samples={}, success={:.1}%, avg_lat={:.2}µs, p99_lat={:.2}µs",
            fr.fiber_type, fr.sample_count, fr.success_rate * 100.0,
            fr.avg_latency_us, fr.p99_latency_us,
        );
    }

    println!("\n━━━ Mutation Recommendations ━━━");
    println!("  Recommendations: {}", report.recommendations.len());
    for rec in &report.recommendations {
        println!(
            "  {:?} constraint '{}': {} (confidence: {:.0}%)",
            rec.action, rec.constraint_id, rec.reason, rec.confidence * 100.0,
        );
    }

    // Apply mutations.
    println!("\n━━━ Applying DNA Mutations ━━━");
    let mut mutator = DnaMutator::new(dna);
    let results = mutator.apply_recommendations(&report.recommendations);
    println!("  Mutations applied: {}", results.len());
    for result in &results {
        println!(
            "  {} '{}': {} -> {} ({})",
            if result.applied { "APPLIED" } else { "SKIPPED" },
            result.constraint_id,
            result.old_value_desc,
            result.new_value_desc,
            result.reason,
        );
    }

    let updated_dna = mutator.dna();
    println!("\n  DNA version after mutations: v{}.{}.{}",
        updated_dna.version.major, updated_dna.version.minor, updated_dna.version.patch);
    println!("  Total mutations applied: {}", updated_dna.total_mutations);

    println!("\nDemonstration complete.");
}

/// Show feedback status.
pub fn show_status() {
    println!("ML Feedback Loop Status:");
    println!("  Execution log: empty (no active session)");
    println!("  DNA version: 1.0.0 (default)");
    println!("  Total mutations: 0");
    println!("  Run 'cudaclaw feedback --demo' for a demonstration.");
}
