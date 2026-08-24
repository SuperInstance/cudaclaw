// ============================================================
// GPU Cell Agent — Spreadsheet-Cell Agents on the GPU
// ============================================================
//
// Every spreadsheet cell is now a "claw agent" running directly
// inside a GPU kernel. Each cell is a "geometric twin" node
// that inherits constraints from the Constraint-Theory DNA and
// executes operations within its assigned "Muscle Fiber"
// (optimized kernel variant).
//
// DESIGN:
//   CellAgent:       GPU-compatible (repr(C)) agent structure.
//                    Contains the cell's value, CRDT state,
//                    constraint mask, fiber affinity, and
//                    execution metrics.
//
//   MuscleFiber:     An optimized kernel configuration for a
//                    particular workload pattern. Agents are
//                    assigned to fibers based on their observed
//                    access patterns and ML feedback.
//
//   CellAgentGrid:   The host-side manager for a grid of cell
//                    agents. Handles creation, dispatch, metrics
//                    collection, and fiber assignment.
//
//   AgentKernelConfig: GPU launch parameters derived from the
//                    agent grid and constraint DNA.
//
// GPU LAYOUT:
//   Cell agents are stored in Structure-of-Arrays (SoA) format
//   for coalesced memory access. Each array is contiguous in
//   GPU memory so that adjacent threads in a warp process
//   adjacent elements.
//
// ============================================================

pub mod cell_agent;
pub mod muscle_fiber;

use serde::{Deserialize, Serialize};

pub use cell_agent::{
    CellAgent, CellAgentState, CellAgentGrid, AgentKernelConfig,
    CellAgentSoA, AgentExecutionRecord,
};
pub use muscle_fiber::{
    MuscleFiber, FiberRegistry, FiberType, FiberPerformanceProfile,
};

// ============================================================
// CLI Integration
// ============================================================

/// Print help for the `agent` CLI subcommand.
pub fn print_agent_help() {
    println!("cudaclaw agent — GPU Cell Agent Management");
    println!();
    println!("USAGE:");
    println!("  cudaclaw agent [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo                Run a demonstration of cell agents");
    println!("  --status              Show current agent grid status");
    println!("  --fibers              List available Muscle Fibers");
    println!("  --grid <ROWS>x<COLS>  Set grid dimensions (default: 8x8)");
    println!("  --help, -h            Show this help message");
}

/// Parse CLI arguments for the `agent` subcommand.
pub fn parse_agent_args(args: &[String]) -> Option<AgentCliAction> {
    if args.is_empty() {
        return Some(AgentCliAction::Demo { rows: 8, cols: 8 });
    }

    let mut rows = 8u32;
    let mut cols = 8u32;

    // Parse --grid first.
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--grid" && i + 1 < args.len() {
            if let Some((r, c)) = parse_grid_dim(&args[i + 1]) {
                rows = r;
                cols = c;
            }
            i += 2;
            continue;
        }
        i += 1;
    }

    for arg in args {
        match arg.as_str() {
            "--demo" => return Some(AgentCliAction::Demo { rows, cols }),
            "--status" => return Some(AgentCliAction::Status { rows, cols }),
            "--fibers" => return Some(AgentCliAction::ListFibers),
            "--help" | "-h" => return None,
            _ => {}
        }
    }

    Some(AgentCliAction::Demo { rows, cols })
}

fn parse_grid_dim(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() == 2 {
        let r = parts[0].parse().ok()?;
        let c = parts[1].parse().ok()?;
        Some((r, c))
    } else {
        None
    }
}

/// CLI actions for the agent subcommand.
pub enum AgentCliAction {
    Demo { rows: u32, cols: u32 },
    Status { rows: u32, cols: u32 },
    ListFibers,
}

/// Run the agent demonstration.
pub fn run_demo(rows: u32, cols: u32) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   cudaclaw GPU Cell Agent — Demonstration              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Create agent grid.
    let mut grid = CellAgentGrid::new(rows, cols);
    println!("━━━ Agent Grid: {}x{} = {} cells ━━━", rows, cols, rows * cols);

    // Create fiber registry.
    let registry = FiberRegistry::default_fibers();
    println!("━━━ Muscle Fibers: {} available ━━━", registry.fiber_count());
    for fiber in registry.all_fibers() {
        println!(
            "  [{}] {} — block_size={}, regs_per_thread={}, shmem={}",
            fiber.fiber_type.as_str(),
            fiber.name,
            fiber.block_size,
            fiber.registers_per_thread,
            fiber.shared_memory_bytes,
        );
    }

    // Simulate some cell accesses.
    println!("\n━━━ Simulating Cell Accesses ━━━");
    let access_patterns = vec![
        (0, 0, 42.0f64, "Sequential write"),
        (0, 1, 43.0, "Adjacent cell"),
        (1, 0, 100.0, "Row below"),
        (3, 3, 999.0, "Random access"),
    ];

    for (r, c, val, desc) in &access_patterns {
        grid.set_cell_value(*r, *c, *val);
        println!("  Set cell({},{}) = {:.1} — {}", r, c, val, desc);
    }

    // Record some execution metrics.
    println!("\n━━━ Recording Execution Metrics ━━━");
    for i in 0..4 {
        let record = AgentExecutionRecord {
            agent_row: i,
            agent_col: 0,
            fiber_type: "cell_update".into(),
            execution_time_us: 2.5 + (i as f64 * 0.3),
            registers_used: 32 * (i + 1),
            shared_memory_used: 256 * (i as u32 + 1),
            success: true,
            timestamp_epoch: 1000 + i as u64,
        };
        grid.record_execution(record.clone());
        println!(
            "  Cell({},0): {:.1} µs, {} regs, {} shmem, fiber={}",
            i, record.execution_time_us, record.registers_used,
            record.shared_memory_used, record.fiber_type
        );
    }

    // Assign fibers based on access patterns.
    println!("\n━━━ Fiber Assignment ━━━");
    grid.assign_fiber(0, 0, "cell_update".into());
    grid.assign_fiber(0, 1, "cell_update".into());
    grid.assign_fiber(3, 3, "crdt_merge".into());
    grid.assign_fiber(1, 0, "formula_eval".into());

    for (r, c) in &[(0u32, 0u32), (0, 1), (1, 0), (3, 3)] {
        if let Some(agent) = grid.get_agent(*r, *c) {
            println!(
                "  Cell({},{}) -> fiber='{}', state={:?}",
                r, c, agent.fiber_affinity, agent.state
            );
        }
    }

    // Show kernel launch config.
    println!("\n━━━ Kernel Launch Config ━━━");
    let config = grid.kernel_config(&registry);
    println!("  Grid dim:  {} blocks", config.grid_dim);
    println!("  Block dim: {} threads", config.block_dim);
    println!("  Shared mem: {} bytes", config.shared_memory_bytes);
    println!("  Regs/thread: {}", config.registers_per_thread);

    // Show SoA layout.
    println!("\n━━━ SoA Layout (for GPU transfer) ━━━");
    let soa = grid.to_soa();
    println!("  Total agents: {}", soa.count);
    println!("  Values array: {} elements", soa.values.len());
    println!("  Timestamps array: {} elements", soa.timestamps.len());
    println!("  Fiber IDs array: {} elements", soa.fiber_ids.len());

    // Summary.
    let stats = grid.stats();
    println!("\n━━━ Grid Statistics ━━━");
    println!("  Total cells: {}", stats.total_cells);
    println!("  Active cells: {}", stats.active_cells);
    println!("  Execution records: {}", stats.execution_records);
    println!("  Fiber assignments: {:?}", stats.fiber_distribution);

    println!("\nDemonstration complete.");
}

/// Show agent grid status.
pub fn show_status(rows: u32, cols: u32) {
    let grid = CellAgentGrid::new(rows, cols);
    let stats = grid.stats();
    println!("GPU Cell Agent Grid Status:");
    println!("  Grid: {}x{}", rows, cols);
    println!("  Total cells: {}", stats.total_cells);
    println!("  Active cells: {}", stats.active_cells);
    println!("  Execution records: {}", stats.execution_records);
}

/// List available muscle fibers.
pub fn list_fibers() {
    let registry = FiberRegistry::default_fibers();
    println!("Available Muscle Fibers:");
    for fiber in registry.all_fibers() {
        println!(
            "  {:16} — block={:4}, regs={:3}, shmem={:6}, occupancy={:.0}%",
            fiber.fiber_type.as_str(),
            fiber.block_size,
            fiber.registers_per_thread,
            fiber.shared_memory_bytes,
            fiber.target_occupancy * 100.0,
        );
    }
}
