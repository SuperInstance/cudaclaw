// ============================================================
// Constraint-Theory Module — System DNA
// ============================================================
//
// The Constraint-Theory module defines the "DNA" of cudaclaw:
// a set of super-constraints that every operation must satisfy.
// These constraints are derived from the Constraint-Theory
// project and encode the fundamental invariants of the system.
//
// DESIGN:
// - SuperConstraint: A named, typed constraint with a validation
//   function. Examples: "register budget per agent", "shared
//   memory ceiling per SM", "minimum warp occupancy", "latency
//   SLA".
// - ConstraintDna: A collection of super-constraints that define
//   the system's operational envelope. The DNA is versioned and
//   can be mutated by the ML feedback loop.
// - ConstraintValidator: Evaluates a proposed operation against
//   the DNA and returns a verdict (Pass, Warn, Fail) with
//   diagnostics.
// - GeometricTwinBinding: Maps each spreadsheet cell to a
//   "geometric twin" node in the constraint graph, so cell
//   operations inherit the constraints of their twin.
//
// CONSTRAINT TAXONOMY:
//   Resource Constraints  — register budget, shmem ceiling,
//                           warp slots, thread budget
//   Latency Constraints   — P99 RTT ceiling, push latency,
//                           PCIe transfer budget
//   Correctness Constraints — CRDT monotonicity, timestamp
//                             ordering, no data races
//   Efficiency Constraints — minimum occupancy, coalescing
//                            ratio, idle power ceiling
//   Biological Constraints — nutrient floor per SM, prune
//                            cooldown, branch hysteresis
//
// ============================================================

pub mod dna;
pub mod validator;
pub mod geometric_twin;

use serde::{Deserialize, Serialize};

pub use dna::{
    ConstraintDna, ConstraintDnaVersion, SuperConstraint, ConstraintCategory,
    ConstraintSeverity, ConstraintValue,
};
pub use validator::{
    ConstraintValidator, ConstraintVerdict, VerdictKind, OperationContext,
};
pub use geometric_twin::{
    GeometricTwinMap, TwinNode, TwinBinding, TwinTopology,
};

// ============================================================
// CLI Integration
// ============================================================

/// Print help for the `constraint` CLI subcommand.
pub fn print_constraint_help() {
    println!("cudaclaw constraint — Constraint-Theory DNA Management");
    println!();
    println!("USAGE:");
    println!("  cudaclaw constraint [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --show-dna            Display the current Constraint DNA");
    println!("  --validate            Validate the DNA for internal consistency");
    println!("  --export <PATH>       Export DNA to JSON file");
    println!("  --import <PATH>       Import DNA from JSON file");
    println!("  --show-twins          Display the geometric twin topology");
    println!("  --demo                Run a demonstration of constraint validation");
    println!("  --help, -h            Show this help message");
}

/// Parse CLI arguments for the `constraint` subcommand.
pub fn parse_constraint_args(args: &[String]) -> Option<ConstraintCliAction> {
    if args.is_empty() {
        return Some(ConstraintCliAction::Demo);
    }

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--show-dna" => return Some(ConstraintCliAction::ShowDna),
            "--validate" => return Some(ConstraintCliAction::Validate),
            "--export" => {
                i += 1;
                if i < args.len() {
                    return Some(ConstraintCliAction::Export(args[i].clone()));
                }
                return Some(ConstraintCliAction::Demo);
            }
            "--import" => {
                i += 1;
                if i < args.len() {
                    return Some(ConstraintCliAction::Import(args[i].clone()));
                }
                return Some(ConstraintCliAction::Demo);
            }
            "--show-twins" => return Some(ConstraintCliAction::ShowTwins),
            "--demo" => return Some(ConstraintCliAction::Demo),
            "--help" | "-h" => return None,
            _ => {}
        }
        i += 1;
    }

    Some(ConstraintCliAction::Demo)
}

/// CLI actions for the constraint subcommand.
pub enum ConstraintCliAction {
    ShowDna,
    Validate,
    Export(String),
    Import(String),
    ShowTwins,
    Demo,
}

/// Run the constraint demonstration.
pub fn run_demo() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   cudaclaw Constraint-Theory — DNA Demonstration       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create the default DNA.
    let dna = ConstraintDna::default_system_dna();
    println!("━━━ System DNA (v{}) ━━━", dna.version.major);
    println!("Constraints: {}", dna.constraints.len());
    for c in &dna.constraints {
        println!(
            "  [{:?}] {:?} — {} ({})",
            c.severity, c.category, c.name, c.description
        );
    }

    // Create a validator.
    let validator = ConstraintValidator::new(dna.clone());

    // Test a valid operation.
    println!("\n━━━ Validating Operations ━━━");

    let valid_ctx = OperationContext {
        agent_id: "agent_1".into(),
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
    };

    let verdict = validator.validate(&valid_ctx);
    println!(
        "  set_cell (8192 regs, 4KB shmem, 3µs): {:?} — {} pass, {} warn, {} fail",
        verdict.kind,
        verdict.passed.len(),
        verdict.warnings.len(),
        verdict.failures.len()
    );

    // Test an operation that violates constraints.
    let heavy_ctx = OperationContext {
        agent_id: "greedy_agent".into(),
        operation_name: "heavy_batch".into(),
        sm_index: 0,
        registers_needed: 70000,
        shared_memory_needed: 200000,
        warps_needed: 70,
        threads_needed: 3000,
        estimated_latency_us: 15.0,
        is_crdt_write: true,
        timestamp: 50,
        predecessor_timestamp: Some(100), // violation: timestamp < predecessor
        coalescing_ratio: 0.3,
        warp_occupancy: 0.1,
    };

    let verdict = validator.validate(&heavy_ctx);
    println!(
        "  heavy_batch (70k regs, 200KB shmem, 15µs): {:?} — {} pass, {} warn, {} fail",
        verdict.kind,
        verdict.passed.len(),
        verdict.warnings.len(),
        verdict.failures.len()
    );
    for f in &verdict.failures {
        println!("    FAIL: {}", f);
    }
    for w in &verdict.warnings {
        println!("    WARN: {}", w);
    }

    // Demonstrate geometric twin topology.
    println!("\n━━━ Geometric Twin Topology ━━━");
    let mut twin_map = GeometricTwinMap::new(4, 4);
    twin_map.build_default_topology();

    println!("Grid: {}x{} = {} cells", 4, 4, 16);
    println!("Twin nodes: {}", twin_map.node_count());
    println!("Bindings: {}", twin_map.binding_count());

    // Show a few bindings.
    for binding in twin_map.bindings().iter().take(4) {
        println!(
            "  Cell({},{}) <-> Twin '{}' (constraints: {})",
            binding.cell_row,
            binding.cell_col,
            binding.twin_node_id,
            binding.inherited_constraint_ids.len()
        );
    }

    println!("\nDemonstration complete.");
}

/// Show the current DNA.
pub fn show_dna() {
    let dna = ConstraintDna::default_system_dna();
    match serde_json::to_string_pretty(&dna) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing DNA: {}", e),
    }
}

/// Validate the DNA.
pub fn validate_dna() {
    let dna = ConstraintDna::default_system_dna();
    let issues = dna.self_validate();
    if issues.is_empty() {
        println!("DNA is internally consistent. {} constraints validated.", dna.constraints.len());
    } else {
        println!("DNA validation found {} issues:", issues.len());
        for issue in &issues {
            println!("  - {}", issue);
        }
    }
}
