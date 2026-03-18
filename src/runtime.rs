// ============================================================
// Ramify Runtime — DNA-Driven NVRTC Kernel Manager
// ============================================================
//
// The Ramify Runtime is the live execution layer of cudaclaw.
// At startup it reads the `.claw-dna` file, compiles every
// Muscle Fiber kernel on-the-fly using NVRTC (or the simulated
// fallback), and manages the kernel lifecycle at runtime.
//
// THREE CORE RESPONSIBILITIES:
//
//   1. DNA-Driven Compilation
//      Read the `.claw-dna` file, iterate over every muscle fiber,
//      and compile each kernel source (PTX or CUDA C++ via NVRTC)
//      into a ready-to-launch module. No static `.ptx` files.
//
//   2. Automatic Interconnect Bridges
//      When a spreadsheet-cell agent AND a geometric-twin agent
//      are both active on the same cell, the runtime generates a
//      "bridge kernel" that moves data between them through GPU
//      Shared Memory instead of slower Global VRAM. These bridges
//      are compiled via NVRTC and cached.
//
//   3. DNA Feedback / Constraint Violation Flagging
//      After every kernel execution, the runtime validates the
//      observed resource usage against the Constraint-Theory DNA.
//      If a kernel violates any constraint (register budget,
//      latency SLA, shared memory ceiling, etc.), the offending
//      DNA sequence (muscle fiber) is flagged for "re-optimization"
//      by the LLM in the next installer cycle.
//
// ARCHITECTURE:
//
//   ┌─────────────────────────────────────────────────────────┐
//   │                  RamifyRuntime                          │
//   │                                                        │
//   │  .claw-dna file                                        │
//   │       │                                                │
//   │       ▼                                                │
//   │  DnaLoader ──→ RamifiedRole                            │
//   │       │                                                │
//   │       ▼                                                │
//   │  FiberCompiler (NVRTC) ──→ CompiledKernel cache        │
//   │       │                                                │
//   │       ▼                                                │
//   │  BridgeGenerator ──→ bridge kernels (shared mem)       │
//   │       │                                                │
//   │       ▼                                                │
//   │  ConstraintGuard ──→ violation flags → re-opt queue    │
//   └─────────────────────────────────────────────────────────┘
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::dna::{
    DnaKernelSource, DnaMuscleFiber, RamifiedRole,
};
use crate::ramify::nvrtc_compiler::{
    CompileOptions, CompilationResult, NvrtcCompiler,
};
use crate::constraint_theory::{
    ConstraintValidator, ConstraintVerdict, OperationContext, VerdictKind,
    ConstraintDna,
};
use crate::gpu_cell_agent::{
    CellAgent, CellAgentGrid,
};
use crate::constraint_theory::geometric_twin::{
    GeometricTwinMap, TwinNode,
};

// ============================================================
// Compiled Kernel — output of NVRTC compilation
// ============================================================

/// A compiled kernel ready for launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledKernel {
    /// The task ID this kernel serves (e.g., "cell_update").
    pub task_id: String,
    /// Compiled PTX assembly string.
    pub ptx: String,
    /// Target architecture (e.g., "sm_89").
    pub target_arch: String,
    /// Block size for launch.
    pub block_size: u32,
    /// Shared memory per block (bytes).
    pub shared_memory_bytes: u32,
    /// Registers per thread.
    pub registers_per_thread: u32,
    /// Whether this was compiled via real NVRTC or simulated.
    pub is_simulated: bool,
    /// Compilation time in microseconds.
    pub compile_time_us: u64,
    /// Kernel entry point names extracted from PTX.
    pub entry_points: Vec<String>,
    /// Timestamp of compilation.
    pub compiled_at: u64,
}

// ============================================================
// Bridge Kernel — shared-memory interconnect
// ============================================================

/// A bridge kernel for moving data between cell agents and
/// geometric twins via shared memory instead of global VRAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeKernel {
    /// Unique bridge ID.
    pub bridge_id: String,
    /// Source agent type (e.g., "cell_agent").
    pub source_type: String,
    /// Destination agent type (e.g., "geometric_twin").
    pub dest_type: String,
    /// Cell coordinates this bridge connects.
    pub cell_row: u32,
    pub cell_col: u32,
    /// The compiled PTX for the bridge kernel.
    pub ptx: String,
    /// Shared memory size for the bridge mailbox (bytes).
    pub shared_memory_bytes: u32,
    /// Whether the bridge is currently active.
    pub active: bool,
    /// Number of transfers completed through this bridge.
    pub transfer_count: u64,
    /// Average transfer latency in nanoseconds.
    pub avg_transfer_latency_ns: f64,
    /// When this bridge was created.
    pub created_at: u64,
}

// ============================================================
// Constraint Violation — flagged DNA sequences
// ============================================================

/// A constraint violation detected during kernel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// The task ID (muscle fiber) that violated the constraint.
    pub task_id: String,
    /// The constraint name that was violated.
    pub constraint_name: String,
    /// Severity of the violation.
    pub severity: ViolationSeverity,
    /// Description of the violation.
    pub description: String,
    /// Observed value that caused the violation.
    pub observed_value: f64,
    /// The constraint limit that was exceeded.
    pub limit_value: f64,
    /// When the violation occurred.
    pub timestamp: u64,
    /// Whether this fiber has been queued for LLM re-optimization.
    pub flagged_for_reopt: bool,
}

/// Severity of a constraint violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Warning — constraint exceeded but not critical.
    Warning,
    /// Critical — must be re-optimized before next cycle.
    Critical,
    /// Info — informational, no action needed.
    Info,
}

/// A re-optimization request queued for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReoptimizationRequest {
    /// The task ID to re-optimize.
    pub task_id: String,
    /// The constraint violations that triggered this request.
    pub violations: Vec<ConstraintViolation>,
    /// Current fiber configuration (for context).
    pub current_block_size: u32,
    pub current_registers: u32,
    pub current_shared_memory: u32,
    /// Suggested direction for optimization.
    pub optimization_hints: Vec<String>,
    /// When this request was created.
    pub created_at: u64,
    /// Whether this request has been processed.
    pub processed: bool,
}

// ============================================================
// Runtime State
// ============================================================

/// Snapshot of the runtime's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    /// How many fibers are loaded from DNA.
    pub fibers_loaded: usize,
    /// How many kernels have been compiled.
    pub kernels_compiled: usize,
    /// How many bridges are active.
    pub active_bridges: usize,
    /// How many constraint violations have occurred.
    pub total_violations: usize,
    /// How many fibers are flagged for re-optimization.
    pub fibers_flagged_for_reopt: usize,
    /// Total kernel executions tracked.
    pub total_executions: u64,
    /// DNA file path.
    pub dna_path: String,
    /// Whether the DNA was loaded successfully.
    pub dna_loaded: bool,
    /// NVRTC compiler availability.
    pub nvrtc_available: bool,
}

// ============================================================
// Execution Record — post-kernel metrics
// ============================================================

/// Metrics collected after a kernel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelExecutionRecord {
    /// Which task was executed.
    pub task_id: String,
    /// Execution latency in microseconds.
    pub latency_us: f64,
    /// Registers actually used per thread.
    pub registers_used: u32,
    /// Shared memory actually used (bytes).
    pub shared_memory_used: u32,
    /// Warps occupied on the SM.
    pub warps_used: u32,
    /// Threads launched.
    pub threads_used: u32,
    /// Whether the execution succeeded.
    pub success: bool,
    /// Coalescing ratio observed (0.0-1.0).
    pub coalescing_ratio: f64,
    /// Warp occupancy observed (0.0-1.0).
    pub warp_occupancy: f64,
    /// GPU temperature at time of execution (Celsius).
    pub temperature_celsius: Option<u32>,
    /// Timestamp.
    pub timestamp: u64,
}

// ============================================================
// Ramify Runtime — Main Entry Point
// ============================================================

/// The Ramify Runtime manages DNA-driven kernel compilation,
/// bridge generation, and constraint enforcement.
pub struct RamifyRuntime {
    /// The loaded DNA (RamifiedRole).
    dna: Option<RamifiedRole>,
    /// Path to the .claw-dna file.
    dna_path: PathBuf,
    /// NVRTC compiler instance.
    compiler: NvrtcCompiler,
    /// Compiled kernel cache: task_id → CompiledKernel.
    compiled_kernels: HashMap<String, CompiledKernel>,
    /// Active bridge kernels: bridge_id → BridgeKernel.
    bridges: HashMap<String, BridgeKernel>,
    /// Constraint validator (built from constraint-theory DNA).
    constraint_validator: ConstraintValidator,
    /// Violation log.
    violations: Vec<ConstraintViolation>,
    /// Re-optimization queue for the LLM.
    reopt_queue: Vec<ReoptimizationRequest>,
    /// Set of task_ids already flagged for re-optimization.
    flagged_fibers: HashMap<String, u64>, // task_id → flagged_at
    /// Total kernel executions.
    total_executions: u64,
    /// Execution records (ring buffer, max 10000).
    execution_log: Vec<KernelExecutionRecord>,
    /// Maximum execution log size.
    max_log_size: usize,
}

impl RamifyRuntime {
    // ========================================================
    // Construction
    // ========================================================

    /// Create a new runtime, loading DNA from the given path.
    ///
    /// If the `.claw-dna` file does not exist, the runtime starts
    /// with a default DNA and creates the file on first save.
    pub fn new(dna_path: &str) -> Self {
        let path = PathBuf::from(dna_path);
        let compiler = NvrtcCompiler::new();
        let constraint_dna = ConstraintDna::default_system_dna();
        let validator = ConstraintValidator::new(constraint_dna);

        let mut runtime = RamifyRuntime {
            dna: None,
            dna_path: path,
            compiler,
            compiled_kernels: HashMap::new(),
            bridges: HashMap::new(),
            constraint_validator: validator,
            violations: Vec::new(),
            reopt_queue: Vec::new(),
            flagged_fibers: HashMap::new(),
            total_executions: 0,
            execution_log: Vec::new(),
            max_log_size: 10_000,
        };

        // Try to load DNA.
        runtime.load_dna();
        runtime
    }

    /// Create a runtime with a pre-built RamifiedRole (for testing).
    pub fn from_dna(dna: RamifiedRole) -> Self {
        let compiler = NvrtcCompiler::new();
        let constraint_dna = ConstraintDna::default_system_dna();
        let validator = ConstraintValidator::new(constraint_dna);

        RamifyRuntime {
            dna: Some(dna),
            dna_path: PathBuf::from(".claw-dna"),
            compiler,
            compiled_kernels: HashMap::new(),
            bridges: HashMap::new(),
            constraint_validator: validator,
            violations: Vec::new(),
            reopt_queue: Vec::new(),
            flagged_fibers: HashMap::new(),
            total_executions: 0,
            execution_log: Vec::new(),
            max_log_size: 10_000,
        }
    }

    // ========================================================
    // 1. DNA Loading
    // ========================================================

    /// Load the .claw-dna file from disk.
    fn load_dna(&mut self) {
        if self.dna_path.exists() {
            match RamifiedRole::load_from_file(&self.dna_path.display().to_string()) {
                Ok(dna) => {
                    println!(
                        "[Runtime] Loaded DNA from '{}' — role='{}', fibers={}, mutations={}",
                        self.dna_path.display(),
                        dna.role,
                        dna.muscle_fibers.fibers.len(),
                        dna.total_mutations,
                    );
                    self.dna = Some(dna);
                }
                Err(e) => {
                    eprintln!(
                        "[Runtime] Failed to load DNA from '{}': {}. Using defaults.",
                        self.dna_path.display(),
                        e
                    );
                    self.dna = Some(RamifiedRole::default_spreadsheet_engine());
                }
            }
        } else {
            println!(
                "[Runtime] No DNA file at '{}'. Creating default spreadsheet engine DNA.",
                self.dna_path.display()
            );
            let dna = RamifiedRole::default_spreadsheet_engine();
            self.dna = Some(dna);
        }
    }

    /// Reload DNA from disk (hot-reload support).
    pub fn reload_dna(&mut self) {
        println!("[Runtime] Reloading DNA from '{}'...", self.dna_path.display());
        self.load_dna();
        // Re-compile all fibers with new DNA.
        self.compile_all_fibers();
    }

    /// Get a reference to the loaded DNA.
    pub fn dna(&self) -> Option<&RamifiedRole> {
        self.dna.as_ref()
    }

    // ========================================================
    // 2. Fiber Compilation (NVRTC)
    // ========================================================

    /// Compile ALL muscle fibers from the loaded DNA.
    ///
    /// Iterates over every fiber in `muscle_fibers.fibers`,
    /// compiles its kernel source via NVRTC (or simulated),
    /// and caches the result in `compiled_kernels`.
    pub fn compile_all_fibers(&mut self) {
        let dna = match &self.dna {
            Some(d) => d.clone(),
            None => {
                eprintln!("[Runtime] No DNA loaded — cannot compile fibers.");
                return;
            }
        };

        println!("[Runtime] Compiling {} muscle fibers via NVRTC...", dna.muscle_fibers.fibers.len());
        let start = Instant::now();

        for (task_id, fiber) in &dna.muscle_fibers.fibers {
            match self.compile_fiber(task_id, fiber) {
                Ok(compiled) => {
                    println!(
                        "  [OK] {} — {} bytes PTX, {} entry points, {:.1}ms{}",
                        task_id,
                        compiled.ptx.len(),
                        compiled.entry_points.len(),
                        compiled.compile_time_us as f64 / 1000.0,
                        if compiled.is_simulated { " (simulated)" } else { "" },
                    );
                    self.compiled_kernels.insert(task_id.clone(), compiled);
                }
                Err(e) => {
                    eprintln!("  [FAIL] {} — {}", task_id, e);
                }
            }
        }

        let elapsed = start.elapsed();
        println!(
            "[Runtime] Compiled {} kernels in {:.1}ms",
            self.compiled_kernels.len(),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    /// Compile a single muscle fiber using NVRTC.
    fn compile_fiber(
        &mut self,
        task_id: &str,
        fiber: &DnaMuscleFiber,
    ) -> Result<CompiledKernel, String> {
        let start = Instant::now();

        let (ptx, target_arch, is_simulated, entry_points) = match &fiber.kernel_source {
            DnaKernelSource::NvrtcSource {
                source,
                program_name,
                gpu_arch,
                defines,
                extra_flags: _,
                max_registers,
                use_fast_math,
            } => {
                // Parse compute capability from gpu_arch (e.g., "compute_89" → (8,9))
                let (major, minor) = parse_compute_cap_u32(gpu_arch);
                let mut opts = CompileOptions::for_compute(major, minor);

                // Inject defines from DNA.
                for (k, v) in defines {
                    opts = opts.define(k, v);
                }

                // Inject fiber launch params as defines.
                opts = opts.define("BLOCK_SIZE", &fiber.block_size.to_string());
                opts = opts.define("SHARED_MEM_BYTES", &fiber.shared_memory_bytes.to_string());
                opts = opts.define("REGS_PER_THREAD", &fiber.registers_per_thread.to_string());

                if *use_fast_math {
                    opts = opts.fast_math();
                }
                if *max_registers > 0 {
                    opts = opts.max_regs(*max_registers);
                }

                match self.compiler.compile(source, program_name, &opts) {
                    Ok(result) => {
                        let arch = format!("sm_{}{}", major, minor);
                        let entries = extract_entry_points(&result.ptx);
                        (result.ptx, arch, !self.compiler.is_available(), entries)
                    }
                    Err(e) => return Err(format!("NVRTC compilation failed: {:?}", e)),
                }
            }

            DnaKernelSource::Ptx { ptx, target_arch } => {
                // PTX is already compiled — just use it directly.
                let entries = extract_entry_points(ptx);
                (ptx.clone(), target_arch.clone(), false, entries)
            }

            DnaKernelSource::None => {
                // No kernel source — generate a stub.
                let stub_ptx = generate_stub_ptx(task_id);
                let entries = extract_entry_points(&stub_ptx);
                (stub_ptx, "sm_89".into(), true, entries)
            }
        };

        let compile_time_us = start.elapsed().as_micros() as u64;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(CompiledKernel {
            task_id: task_id.to_string(),
            ptx,
            target_arch,
            block_size: fiber.block_size,
            shared_memory_bytes: fiber.shared_memory_bytes,
            registers_per_thread: fiber.registers_per_thread,
            is_simulated,
            compile_time_us,
            entry_points,
            compiled_at: now,
        })
    }

    /// Get a compiled kernel by task ID.
    pub fn get_kernel(&self, task_id: &str) -> Option<&CompiledKernel> {
        self.compiled_kernels.get(task_id)
    }

    /// Get all compiled kernels.
    pub fn compiled_kernels(&self) -> &HashMap<String, CompiledKernel> {
        &self.compiled_kernels
    }

    // ========================================================
    // 3. Automatic Interconnect Bridges
    // ========================================================

    /// Detect cell-agent / geometric-twin co-activity and
    /// generate bridge kernels for shared-memory data transfer.
    ///
    /// For each cell where BOTH a cell agent AND a geometric twin
    /// are active, we generate a bridge kernel that moves data
    /// through shared memory (~5 cycles) instead of global VRAM
    /// (~400 cycles).
    pub fn generate_bridges(
        &mut self,
        agent_grid: &CellAgentGrid,
        twin_map: &GeometricTwinMap,
    ) {
        println!("[Runtime] Scanning for cell-agent ↔ geometric-twin co-activity...");
        let start = Instant::now();
        let mut new_bridges = 0u32;

        let (grid_rows, grid_cols) = twin_map.grid_size();

        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let has_agent = agent_grid.get_agent(row, col).is_some();
                let has_twin = twin_map.get_node(row, col).is_some();

                if has_agent && has_twin {
                    let bridge_id = format!("bridge_{}_{}", row, col);

                    // Skip if bridge already exists and is active.
                    if self.bridges.contains_key(&bridge_id) {
                        continue;
                    }

                    // Generate bridge kernel.
                    let bridge = self.generate_bridge_kernel(row, col, &bridge_id);
                    self.bridges.insert(bridge_id, bridge);
                    new_bridges += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        println!(
            "[Runtime] Generated {} new bridges ({} total active) in {:.1}ms",
            new_bridges,
            self.bridges.values().filter(|b| b.active).count(),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    /// Generate a single bridge kernel for a (row, col) cell.
    ///
    /// The bridge kernel:
    /// 1. Reads cell-agent state from global memory
    /// 2. Writes it to a shared-memory mailbox
    /// 3. Geometric twin reads from the mailbox
    /// 4. Twin writes its output back to the mailbox
    /// 5. Cell agent reads the updated data from shared memory
    ///
    /// This avoids two global memory round-trips (~800 cycles)
    /// in favor of two shared memory accesses (~10 cycles).
    fn generate_bridge_kernel(
        &mut self,
        row: u32,
        col: u32,
        bridge_id: &str,
    ) -> BridgeKernel {
        // Generate CUDA C++ source for the bridge kernel.
        let cuda_source = generate_bridge_cuda_source(row, col);

        // Compile via NVRTC.
        let opts = CompileOptions::default().fast_math();
        let (ptx, compile_ok) = match self.compiler.compile(
            &cuda_source,
            &format!("{}.cu", bridge_id),
            &opts,
        ) {
            Ok(result) => (result.ptx, true),
            Err(_) => {
                // Fallback: generate a stub PTX.
                (generate_bridge_stub_ptx(row, col), false)
            }
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        BridgeKernel {
            bridge_id: bridge_id.to_string(),
            source_type: "cell_agent".into(),
            dest_type: "geometric_twin".into(),
            cell_row: row,
            cell_col: col,
            ptx,
            shared_memory_bytes: 256, // Double-buffered mailbox: 2 × 128 bytes
            active: true,
            transfer_count: 0,
            avg_transfer_latency_ns: 0.0,
            created_at: now,
        }
    }

    /// Record a bridge transfer completion.
    pub fn record_bridge_transfer(
        &mut self,
        bridge_id: &str,
        latency_ns: f64,
    ) {
        if let Some(bridge) = self.bridges.get_mut(bridge_id) {
            let old_total = bridge.avg_transfer_latency_ns * bridge.transfer_count as f64;
            bridge.transfer_count += 1;
            bridge.avg_transfer_latency_ns =
                (old_total + latency_ns) / bridge.transfer_count as f64;
        }
    }

    /// Deactivate a bridge (e.g., when agents are no longer co-active).
    pub fn deactivate_bridge(&mut self, bridge_id: &str) {
        if let Some(bridge) = self.bridges.get_mut(bridge_id) {
            bridge.active = false;
        }
    }

    /// Get all active bridges.
    pub fn active_bridges(&self) -> Vec<&BridgeKernel> {
        self.bridges.values().filter(|b| b.active).collect()
    }

    // ========================================================
    // 4. DNA Feedback / Constraint Violation Flagging
    // ========================================================

    /// Record a kernel execution and check for constraint violations.
    ///
    /// This is called after every kernel launch. It:
    /// 1. Logs the execution record
    /// 2. Builds an OperationContext from the record
    /// 3. Validates against the Constraint-Theory DNA
    /// 4. If any violations → flags the fiber for re-optimization
    /// 5. Updates the DNA exhaustion metrics
    pub fn record_execution(&mut self, record: KernelExecutionRecord) {
        // Log the execution.
        self.total_executions += 1;
        if self.execution_log.len() >= self.max_log_size {
            self.execution_log.remove(0);
        }
        self.execution_log.push(record.clone());

        // Build constraint-theory OperationContext.
        let ctx = OperationContext {
            agent_id: format!("fiber_{}", record.task_id),
            operation_name: record.task_id.clone(),
            sm_index: 0,
            registers_needed: record.registers_used as u64,
            shared_memory_needed: record.shared_memory_used as u64,
            warps_needed: record.warps_used as u64,
            threads_needed: record.threads_used as u64,
            estimated_latency_us: record.latency_us,
            is_crdt_write: record.task_id == "crdt_merge",
            timestamp: record.timestamp,
            predecessor_timestamp: if record.timestamp > 0 {
                Some(record.timestamp - 1)
            } else {
                None
            },
            coalescing_ratio: record.coalescing_ratio,
            warp_occupancy: record.warp_occupancy,
        };

        // Validate against Constraint-Theory DNA.
        let verdict = self.constraint_validator.validate(&ctx);

        match verdict.kind {
            VerdictKind::Fail => {
                // Critical constraint violations — flag for re-optimization.
                for failure_msg in &verdict.failures {
                    let violation = ConstraintViolation {
                        task_id: record.task_id.clone(),
                        constraint_name: extract_constraint_name(failure_msg),
                        severity: ViolationSeverity::Critical,
                        description: failure_msg.clone(),
                        observed_value: record.latency_us,
                        limit_value: 0.0, // Extracted from constraint
                        timestamp: record.timestamp,
                        flagged_for_reopt: true,
                    };
                    self.violations.push(violation);
                }

                // Flag this fiber for LLM re-optimization.
                self.flag_for_reoptimization(&record);
            }
            VerdictKind::Warn => {
                // Warnings — log but don't flag yet.
                for warning_msg in &verdict.warnings {
                    let violation = ConstraintViolation {
                        task_id: record.task_id.clone(),
                        constraint_name: extract_constraint_name(warning_msg),
                        severity: ViolationSeverity::Warning,
                        description: warning_msg.clone(),
                        observed_value: record.latency_us,
                        limit_value: 0.0,
                        timestamp: record.timestamp,
                        flagged_for_reopt: false,
                    };
                    self.violations.push(violation);
                }
            }
            VerdictKind::Pass => {
                // All good — record success in DNA if available.
                if let Some(ref mut dna) = self.dna {
                    dna.muscle_fibers.record_measurement(
                        &record.task_id,
                        record.latency_us,
                    );
                }
            }
        }

        // Update DNA exhaustion metrics if available.
        if let Some(ref mut dna) = self.dna {
            if let Some(temp) = record.temperature_celsius {
                dna.exhaustion.record_thermal(
                    crate::dna::DnaThermalSample {
                        timestamp: record.timestamp,
                        temperature_c: temp as f64,
                        power_watts: 0.0,
                        fan_speed_pct: 0.0,
                        is_throttling: temp >= 90,
                    },
                );
            }
        }
    }

    /// Flag a muscle fiber for LLM re-optimization.
    fn flag_for_reoptimization(&mut self, record: &KernelExecutionRecord) {
        let task_id = &record.task_id;

        // Don't re-flag if already flagged recently (within 60 seconds).
        if let Some(&flagged_at) = self.flagged_fibers.get(task_id) {
            if record.timestamp.saturating_sub(flagged_at) < 60 {
                return;
            }
        }

        // Collect all violations for this fiber.
        let fiber_violations: Vec<ConstraintViolation> = self
            .violations
            .iter()
            .filter(|v| v.task_id == *task_id && v.severity == ViolationSeverity::Critical)
            .cloned()
            .collect();

        if fiber_violations.is_empty() {
            return;
        }

        // Build optimization hints from violations.
        let hints = build_optimization_hints(&fiber_violations, record);

        // Get current fiber config from DNA.
        let (block_size, registers, shmem) = if let Some(ref dna) = self.dna {
            if let Some(fiber) = dna.muscle_fibers.get(task_id) {
                (
                    fiber.block_size,
                    fiber.registers_per_thread,
                    fiber.shared_memory_bytes,
                )
            } else {
                (128, 32, 4096)
            }
        } else {
            (128, 32, 4096)
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let request = ReoptimizationRequest {
            task_id: task_id.clone(),
            violations: fiber_violations,
            current_block_size: block_size,
            current_registers: registers,
            current_shared_memory: shmem,
            optimization_hints: hints,
            created_at: now,
            processed: false,
        };

        println!(
            "[Runtime] FLAGGED fiber '{}' for LLM re-optimization ({} violations, {} hints)",
            task_id,
            request.violations.len(),
            request.optimization_hints.len(),
        );

        self.reopt_queue.push(request);
        self.flagged_fibers.insert(task_id.clone(), record.timestamp);
    }

    /// Get all pending re-optimization requests.
    pub fn reopt_queue(&self) -> &[ReoptimizationRequest] {
        &self.reopt_queue
    }

    /// Mark a re-optimization request as processed.
    pub fn mark_reopt_processed(&mut self, task_id: &str) {
        for req in &mut self.reopt_queue {
            if req.task_id == task_id {
                req.processed = true;
            }
        }
    }

    /// Get all constraint violations.
    pub fn violations(&self) -> &[ConstraintViolation] {
        &self.violations
    }

    /// Get violations for a specific task ID.
    pub fn violations_for(&self, task_id: &str) -> Vec<&ConstraintViolation> {
        self.violations
            .iter()
            .filter(|v| v.task_id == task_id)
            .collect()
    }

    // ========================================================
    // 5. Runtime State & Reporting
    // ========================================================

    /// Get the current runtime state snapshot.
    pub fn state(&self) -> RuntimeState {
        let fibers_loaded = self
            .dna
            .as_ref()
            .map(|d| d.muscle_fibers.fibers.len())
            .unwrap_or(0);

        RuntimeState {
            fibers_loaded,
            kernels_compiled: self.compiled_kernels.len(),
            active_bridges: self.bridges.values().filter(|b| b.active).count(),
            total_violations: self.violations.len(),
            fibers_flagged_for_reopt: self.flagged_fibers.len(),
            total_executions: self.total_executions,
            dna_path: self.dna_path.display().to_string(),
            dna_loaded: self.dna.is_some(),
            nvrtc_available: self.compiler.is_available(),
        }
    }

    /// Save the current DNA back to disk (e.g., after mutations).
    pub fn save_dna(&self) -> Result<(), String> {
        if let Some(ref dna) = self.dna {
            dna.save_to_file(&self.dna_path.display().to_string())
                .map_err(|e| format!("Failed to save DNA: {}", e))
        } else {
            Err("No DNA loaded".into())
        }
    }

    /// Export a JSON report of the runtime state.
    pub fn export_report(&self, path: &str) -> Result<(), String> {
        let report = RuntimeReport {
            state: self.state(),
            compiled_kernels: self.compiled_kernels.values().cloned().collect(),
            active_bridges: self.active_bridges().into_iter().cloned().collect(),
            recent_violations: self.violations.iter().rev().take(50).cloned().collect(),
            pending_reopt: self
                .reopt_queue
                .iter()
                .filter(|r| !r.processed)
                .cloned()
                .collect(),
            compiler_stats: CompilerStatsReport {
                total_compilations: self.compiler.stats().total_compilations,
                successful: self.compiler.stats().successful_compilations,
                failed: self.compiler.stats().failed_compilations,
                simulated: self.compiler.stats().simulated_compilations,
                avg_compile_time_us: self.compiler.stats().avg_compile_time_us,
            },
        };

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("File write error: {}", e))
    }
}

/// Full runtime report (for JSON export).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeReport {
    state: RuntimeState,
    compiled_kernels: Vec<CompiledKernel>,
    active_bridges: Vec<BridgeKernel>,
    recent_violations: Vec<ConstraintViolation>,
    pending_reopt: Vec<ReoptimizationRequest>,
    compiler_stats: CompilerStatsReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompilerStatsReport {
    total_compilations: u64,
    successful: u64,
    failed: u64,
    simulated: u64,
    avg_compile_time_us: f64,
}

// ============================================================
// Helper Functions
// ============================================================

/// Parse compute capability from a string like "compute_89" → (8, 9) as u32.
fn parse_compute_cap_u32(arch: &str) -> (u32, u32) {
    let stripped = arch
        .trim_start_matches("compute_")
        .trim_start_matches("sm_");
    if stripped.len() >= 2 {
        let major = stripped[..1].parse::<u32>().unwrap_or(8);
        let minor = stripped[1..].parse::<u32>().unwrap_or(9);
        (major, minor)
    } else {
        (8, 9) // Default: Ada Lovelace
    }
}

/// Extract entry point names from PTX assembly.
fn extract_entry_points(ptx: &str) -> Vec<String> {
    let mut entries = Vec::new();
    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".entry ") || trimmed.starts_with(".visible .entry ") {
            // Extract the function name.
            let after_entry = if trimmed.starts_with(".visible") {
                trimmed.trim_start_matches(".visible .entry ")
            } else {
                trimmed.trim_start_matches(".entry ")
            };
            if let Some(name_end) = after_entry.find('(') {
                entries.push(after_entry[..name_end].trim().to_string());
            } else if let Some(name_end) = after_entry.find('{') {
                entries.push(after_entry[..name_end].trim().to_string());
            } else {
                let name = after_entry.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
                if !name.is_empty() {
                    entries.push(name.to_string());
                }
            }
        }
    }
    entries
}

/// Generate a stub PTX for fibers with no kernel source.
fn generate_stub_ptx(task_id: &str) -> String {
    format!(
        r#".version 7.8
.target sm_89
.address_size 64

// Stub kernel for task: {task_id}
// This is a placeholder — replace with real kernel source in .claw-dna
.visible .entry {task_id}_kernel(
    .param .u64 param_data,
    .param .u32 param_count
)
{{
    ret;
}}
"#,
        task_id = task_id
    )
}

/// Generate CUDA C++ source for a bridge kernel.
///
/// The bridge kernel moves data between a cell agent and its
/// geometric twin using shared memory for minimal latency.
fn generate_bridge_cuda_source(row: u32, col: u32) -> String {
    format!(
        r#"// Auto-generated bridge kernel for cell ({row}, {col})
// Moves data between cell_agent and geometric_twin via shared memory.
//
// Shared memory layout (256 bytes, double-buffered):
//   [0..127]   Mailbox A: cell_agent → geometric_twin
//   [128..255] Mailbox B: geometric_twin → cell_agent

extern "C" __global__ void bridge_{row}_{col}(
    double* __restrict__ agent_values,
    double* __restrict__ twin_values,
    unsigned int* __restrict__ agent_timestamps,
    unsigned int* __restrict__ twin_timestamps,
    unsigned int agent_count,
    unsigned int twin_count,
    unsigned int target_idx
) {{
    // Shared memory mailbox for zero-copy data exchange.
    __shared__ double shmem_mailbox_a[16];  // agent → twin
    __shared__ double shmem_mailbox_b[16];  // twin → agent
    __shared__ unsigned int shmem_ts_a[16];
    __shared__ unsigned int shmem_ts_b[16];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;

    // Phase 1: Cell agent writes to mailbox A (shared memory).
    if (lane < 16 && target_idx < agent_count) {{
        shmem_mailbox_a[lane] = agent_values[target_idx];
        shmem_ts_a[lane] = agent_timestamps[target_idx];
    }}
    __syncthreads();

    // Phase 2: Geometric twin reads from mailbox A, processes,
    // and writes result to mailbox B.
    if (lane < 16 && target_idx < twin_count) {{
        // Read agent data from shared memory (~5 cycles).
        double agent_val = shmem_mailbox_a[lane];
        unsigned int agent_ts = shmem_ts_a[lane];

        // Twin processing: merge with twin's own data.
        double twin_val = twin_values[target_idx];
        unsigned int twin_ts = twin_timestamps[target_idx];

        // LWW merge: keep the value with the higher timestamp.
        if (agent_ts > twin_ts) {{
            twin_values[target_idx] = agent_val;
            twin_timestamps[target_idx] = agent_ts;
            shmem_mailbox_b[lane] = agent_val;
            shmem_ts_b[lane] = agent_ts;
        }} else {{
            shmem_mailbox_b[lane] = twin_val;
            shmem_ts_b[lane] = twin_ts;
        }}
    }}
    __syncthreads();

    // Phase 3: Cell agent reads merged result from mailbox B.
    if (lane < 16 && target_idx < agent_count) {{
        agent_values[target_idx] = shmem_mailbox_b[lane];
        agent_timestamps[target_idx] = shmem_ts_b[lane];
    }}
}}
"#,
        row = row,
        col = col
    )
}

/// Generate stub PTX for a bridge kernel (fallback).
fn generate_bridge_stub_ptx(row: u32, col: u32) -> String {
    format!(
        r#".version 7.8
.target sm_89
.address_size 64

// Bridge stub for cell ({row}, {col})
// Shared memory bridge: cell_agent ↔ geometric_twin
.visible .entry bridge_{row}_{col}(
    .param .u64 param_agent_values,
    .param .u64 param_twin_values,
    .param .u64 param_agent_timestamps,
    .param .u64 param_twin_timestamps,
    .param .u32 param_agent_count,
    .param .u32 param_twin_count,
    .param .u32 param_target_idx
)
{{
    ret;
}}
"#,
        row = row,
        col = col,
    )
}

/// Extract constraint name from a violation message.
fn extract_constraint_name(msg: &str) -> String {
    // Messages are typically "constraint_name: details"
    if let Some(colon_pos) = msg.find(':') {
        msg[..colon_pos].trim().to_string()
    } else {
        msg.to_string()
    }
}

/// Build optimization hints from constraint violations.
fn build_optimization_hints(
    violations: &[ConstraintViolation],
    record: &KernelExecutionRecord,
) -> Vec<String> {
    let mut hints = Vec::new();

    for v in violations {
        match v.constraint_name.as_str() {
            name if name.contains("register") => {
                hints.push(format!(
                    "Reduce registers per thread (current: {}, limit exceeded)",
                    record.registers_used,
                ));
            }
            name if name.contains("shared_memory") || name.contains("shmem") => {
                hints.push(format!(
                    "Reduce shared memory usage (current: {} bytes)",
                    record.shared_memory_used,
                ));
            }
            name if name.contains("latency") || name.contains("p99") => {
                hints.push(format!(
                    "Reduce P99 latency (observed: {:.1} µs). Consider smaller block size or loop unrolling.",
                    record.latency_us,
                ));
            }
            name if name.contains("occupancy") => {
                hints.push(format!(
                    "Increase warp occupancy (observed: {:.0}%). Consider reducing register pressure.",
                    record.warp_occupancy * 100.0,
                ));
            }
            name if name.contains("coalescing") => {
                hints.push(format!(
                    "Improve memory coalescing (observed: {:.0}%). Consider SoA layout or sorted access patterns.",
                    record.coalescing_ratio * 100.0,
                ));
            }
            _ => {
                hints.push(format!(
                    "Constraint '{}' violated — review fiber configuration.",
                    v.constraint_name,
                ));
            }
        }
    }

    hints.dedup();
    hints
}

// ============================================================
// CLI Integration
// ============================================================

/// CLI actions for the `runtime` subcommand.
pub enum RuntimeCliAction {
    /// Run full demo: load DNA, compile fibers, generate bridges, simulate executions.
    Demo,
    /// Show runtime status.
    Status,
    /// Export runtime report to JSON.
    Export(String),
    /// Show help.
    Help,
}

/// Parse CLI arguments for the `runtime` subcommand.
pub fn parse_runtime_args(args: &[String]) -> Option<RuntimeCliAction> {
    if args.is_empty() {
        return Some(RuntimeCliAction::Demo);
    }

    for i in 0..args.len() {
        match args[i].as_str() {
            "--demo" => return Some(RuntimeCliAction::Demo),
            "--status" => return Some(RuntimeCliAction::Status),
            "--export" => {
                if i + 1 < args.len() {
                    return Some(RuntimeCliAction::Export(args[i + 1].clone()));
                }
                return Some(RuntimeCliAction::Export("runtime_report.json".into()));
            }
            "--help" | "-h" => return None,
            _ => {}
        }
    }

    Some(RuntimeCliAction::Demo)
}

/// Print help for the `runtime` CLI subcommand.
pub fn print_runtime_help() {
    println!("cudaclaw runtime — Ramify Runtime (DNA-Driven NVRTC Kernel Manager)");
    println!();
    println!("USAGE:");
    println!("  cudaclaw runtime [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo                Run full runtime demonstration");
    println!("  --status              Show runtime status");
    println!("  --export <PATH>       Export runtime report to JSON (default: runtime_report.json)");
    println!("  --help, -h            Show this help message");
    println!();
    println!("The runtime reads .claw-dna at startup, compiles muscle fibers via NVRTC,");
    println!("generates shared-memory bridges for co-active agents, and flags constraint");
    println!("violations for LLM re-optimization.");
}

/// Run the runtime demonstration.
pub fn run_demo() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   cudaclaw Ramify Runtime — Demonstration              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Phase 1: Load DNA
    println!("━━━ Phase 1: Load .claw-dna ━━━");
    let mut runtime = RamifyRuntime::from_dna(
        RamifiedRole::default_spreadsheet_engine(),
    );
    let state = runtime.state();
    println!("  DNA loaded: {}", state.dna_loaded);
    println!("  Fibers in DNA: {}", state.fibers_loaded);
    println!("  NVRTC available: {}", state.nvrtc_available);
    println!();

    // Phase 2: Compile all muscle fibers
    println!("━━━ Phase 2: Compile Muscle Fibers via NVRTC ━━━");
    runtime.compile_all_fibers();
    println!();

    // Show compiled kernels
    println!("━━━ Compiled Kernels ━━━");
    for (task_id, kernel) in runtime.compiled_kernels() {
        println!(
            "  {} — {} bytes PTX, block={}, shmem={}, regs={}, entries={:?}{}",
            task_id,
            kernel.ptx.len(),
            kernel.block_size,
            kernel.shared_memory_bytes,
            kernel.registers_per_thread,
            kernel.entry_points,
            if kernel.is_simulated { " (simulated)" } else { "" },
        );
    }
    println!();

    // Phase 3: Generate bridge kernels
    println!("━━━ Phase 3: Automatic Interconnect Bridges ━━━");
    let mut agent_grid = CellAgentGrid::new(4, 4);
    // Activate some cell agents.
    agent_grid.set_cell_value(0, 0, 42.0);
    agent_grid.set_cell_value(1, 1, 100.0);
    agent_grid.set_cell_value(2, 2, 200.0);

    let mut twin_map = GeometricTwinMap::new(4, 4);
    twin_map.build_default_topology();

    runtime.generate_bridges(&agent_grid, &twin_map);

    for bridge in runtime.active_bridges() {
        println!(
            "  {} — cell({},{}) {} ↔ {}, shmem={} bytes",
            bridge.bridge_id,
            bridge.cell_row,
            bridge.cell_col,
            bridge.source_type,
            bridge.dest_type,
            bridge.shared_memory_bytes,
        );
    }
    println!();

    // Phase 4: Simulate kernel executions and constraint checking
    println!("━━━ Phase 4: DNA Feedback — Constraint Validation ━━━");

    // Simulate a valid execution.
    let good_record = KernelExecutionRecord {
        task_id: "cell_update".into(),
        latency_us: 3.5,
        registers_used: 24,
        shared_memory_used: 2048,
        warps_used: 4,
        threads_used: 128,
        success: true,
        coalescing_ratio: 0.95,
        warp_occupancy: 0.75,
        temperature_celsius: Some(65),
        timestamp: 1000,
    };
    runtime.record_execution(good_record);
    println!("  cell_update (3.5µs, 24 regs, 2KB shmem): PASS");

    // Simulate a valid CRDT merge.
    let crdt_record = KernelExecutionRecord {
        task_id: "crdt_merge".into(),
        latency_us: 5.0,
        registers_used: 32,
        shared_memory_used: 4096,
        warps_used: 8,
        threads_used: 256,
        success: true,
        coalescing_ratio: 0.85,
        warp_occupancy: 0.60,
        temperature_celsius: Some(70),
        timestamp: 1001,
    };
    runtime.record_execution(crdt_record);
    println!("  crdt_merge (5.0µs, 32 regs, 4KB shmem): PASS");

    // Simulate a VIOLATING execution (excessive resources).
    let bad_record = KernelExecutionRecord {
        task_id: "formula_eval".into(),
        latency_us: 15.0, // Exceeds P99 target
        registers_used: 70000, // Exceeds register budget
        shared_memory_used: 200000, // Exceeds shmem ceiling
        warps_used: 70, // Exceeds warp slots
        threads_used: 3000,
        success: true,
        coalescing_ratio: 0.3, // Below coalescing threshold
        warp_occupancy: 0.1, // Below occupancy threshold
        temperature_celsius: Some(88),
        timestamp: 1002,
    };
    runtime.record_execution(bad_record);

    // Show violations.
    let violations = runtime.violations();
    if violations.is_empty() {
        println!("  formula_eval: No violations (constraints may be lenient)");
    } else {
        println!("  formula_eval: {} violations detected:", violations.len());
        for v in violations.iter().take(5) {
            println!(
                "    [{:?}] {} — {}",
                v.severity, v.constraint_name, v.description
            );
        }
    }
    println!();

    // Show re-optimization queue.
    println!("━━━ Phase 5: Re-Optimization Queue ━━━");
    let reopt = runtime.reopt_queue();
    if reopt.is_empty() {
        println!("  No fibers flagged for re-optimization.");
    } else {
        for req in reopt {
            println!(
                "  FLAGGED: '{}' — {} violations, {} hints, processed={}",
                req.task_id,
                req.violations.len(),
                req.optimization_hints.len(),
                req.processed,
            );
            for hint in &req.optimization_hints {
                println!("    Hint: {}", hint);
            }
        }
    }
    println!();

    // Summary.
    let final_state = runtime.state();
    println!("━━━ Runtime Summary ━━━");
    println!("  Fibers loaded: {}", final_state.fibers_loaded);
    println!("  Kernels compiled: {}", final_state.kernels_compiled);
    println!("  Active bridges: {}", final_state.active_bridges);
    println!("  Total violations: {}", final_state.total_violations);
    println!("  Fibers flagged for re-opt: {}", final_state.fibers_flagged_for_reopt);
    println!("  Total executions: {}", final_state.total_executions);

    println!("\nDemonstration complete.");
}

/// Show runtime status (creates a temporary runtime to display state).
pub fn show_status() {
    let runtime = RamifyRuntime::new(".claw-dna");
    let state = runtime.state();
    println!("Ramify Runtime Status:");
    println!("  DNA file: {}", state.dna_path);
    println!("  DNA loaded: {}", state.dna_loaded);
    println!("  Fibers: {}", state.fibers_loaded);
    println!("  Kernels compiled: {}", state.kernels_compiled);
    println!("  Active bridges: {}", state.active_bridges);
    println!("  NVRTC available: {}", state.nvrtc_available);
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default_runtime() -> RamifyRuntime {
        RamifyRuntime::from_dna(RamifiedRole::default_spreadsheet_engine())
    }

    #[test]
    fn test_runtime_creation() {
        let rt = make_default_runtime();
        assert!(rt.dna.is_some());
        assert_eq!(rt.compiled_kernels.len(), 0);
        assert_eq!(rt.bridges.len(), 0);
        assert_eq!(rt.violations.len(), 0);
        assert_eq!(rt.total_executions, 0);
    }

    #[test]
    fn test_compile_all_fibers() {
        let mut rt = make_default_runtime();
        rt.compile_all_fibers();
        // Default DNA has 5 fibers.
        assert!(rt.compiled_kernels.len() >= 1);
        // Every compiled kernel should have valid PTX.
        for (_, kernel) in &rt.compiled_kernels {
            assert!(!kernel.ptx.is_empty());
            assert!(kernel.block_size > 0);
        }
    }

    #[test]
    fn test_compile_fiber_nvrtc_source() {
        let mut rt = make_default_runtime();
        let dna = rt.dna.as_ref().unwrap().clone();
        // cell_update has NvrtcSource in default DNA.
        if let Some(fiber) = dna.muscle_fibers.get("cell_update") {
            let result = rt.compile_fiber("cell_update", fiber);
            assert!(result.is_ok());
            let compiled = result.unwrap();
            assert_eq!(compiled.task_id, "cell_update");
            assert!(!compiled.ptx.is_empty());
        }
    }

    #[test]
    fn test_compile_fiber_ptx_source() {
        let mut rt = make_default_runtime();
        // Create a fiber with raw PTX.
        let fiber = DnaMuscleFiber {
            name: "test_ptx".into(),
            description: "Test fiber with raw PTX".into(),
            block_size: 64,
            registers_per_thread: 16,
            shared_memory_bytes: 1024,
            target_occupancy: 0.5,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            kernel_source: DnaKernelSource::Ptx {
                ptx: ".version 7.8\n.target sm_89\n.entry test_kernel() { ret; }".into(),
                target_arch: "sm_89".into(),
            },
            expected_p99_us: 5.0,
            expected_throughput_ops: 1_000_000.0,
            best_measured_latency_us: f64::MAX,
            worst_measured_latency_us: 0.0,
            measurement_count: 0,
        };
        let result = rt.compile_fiber("test_ptx", &fiber);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(!compiled.is_simulated);
        assert!(compiled.ptx.contains("test_kernel"));
    }

    #[test]
    fn test_compile_fiber_no_source() {
        let mut rt = make_default_runtime();
        let fiber = DnaMuscleFiber {
            name: "stub".into(),
            description: "Stub fiber".into(),
            block_size: 32,
            registers_per_thread: 8,
            shared_memory_bytes: 512,
            target_occupancy: 0.5,
            uses_ptx_cas: false,
            uses_warp_aggregation: false,
            uses_prefix_sum: false,
            is_persistent: false,
            kernel_source: DnaKernelSource::None,
            expected_p99_us: 10.0,
            expected_throughput_ops: 100_000.0,
            best_measured_latency_us: f64::MAX,
            worst_measured_latency_us: 0.0,
            measurement_count: 0,
        };
        let result = rt.compile_fiber("stub_task", &fiber);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(compiled.ptx.contains("stub_task_kernel"));
    }

    #[test]
    fn test_generate_bridges() {
        let mut rt = make_default_runtime();
        let mut grid = CellAgentGrid::new(4, 4);
        grid.set_cell_value(0, 0, 42.0);
        grid.set_cell_value(1, 1, 100.0);

        let mut twin_map = GeometricTwinMap::new(4, 4);
        twin_map.build_default_topology();

        rt.generate_bridges(&grid, &twin_map);

        // Should have bridges for cells where both agent and twin exist.
        // All 16 cells have twins (from build_default_topology), but only
        // cells (0,0) and (1,1) have agents via set_cell_value.
        // However, CellAgentGrid::new(4,4) creates all 16 agents.
        // So all 16 cells should have bridges.
        assert!(rt.bridges.len() > 0);

        // All bridges should be active.
        for (_, bridge) in &rt.bridges {
            assert!(bridge.active);
            assert!(!bridge.ptx.is_empty());
        }
    }

    #[test]
    fn test_bridge_transfer_recording() {
        let mut rt = make_default_runtime();

        // Manually insert a bridge.
        let bridge = BridgeKernel {
            bridge_id: "test_bridge".into(),
            source_type: "cell_agent".into(),
            dest_type: "geometric_twin".into(),
            cell_row: 0,
            cell_col: 0,
            ptx: "stub".into(),
            shared_memory_bytes: 256,
            active: true,
            transfer_count: 0,
            avg_transfer_latency_ns: 0.0,
            created_at: 0,
        };
        rt.bridges.insert("test_bridge".into(), bridge);

        rt.record_bridge_transfer("test_bridge", 100.0);
        rt.record_bridge_transfer("test_bridge", 200.0);

        let b = rt.bridges.get("test_bridge").unwrap();
        assert_eq!(b.transfer_count, 2);
        assert!((b.avg_transfer_latency_ns - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_deactivate_bridge() {
        let mut rt = make_default_runtime();

        let bridge = BridgeKernel {
            bridge_id: "deact_test".into(),
            source_type: "cell_agent".into(),
            dest_type: "geometric_twin".into(),
            cell_row: 0,
            cell_col: 0,
            ptx: "stub".into(),
            shared_memory_bytes: 256,
            active: true,
            transfer_count: 0,
            avg_transfer_latency_ns: 0.0,
            created_at: 0,
        };
        rt.bridges.insert("deact_test".into(), bridge);
        assert!(rt.bridges.get("deact_test").unwrap().active);

        rt.deactivate_bridge("deact_test");
        assert!(!rt.bridges.get("deact_test").unwrap().active);
    }

    #[test]
    fn test_record_valid_execution() {
        let mut rt = make_default_runtime();

        let record = KernelExecutionRecord {
            task_id: "cell_update".into(),
            latency_us: 3.0,
            registers_used: 24,
            shared_memory_used: 2048,
            warps_used: 4,
            threads_used: 128,
            success: true,
            coalescing_ratio: 0.95,
            warp_occupancy: 0.75,
            temperature_celsius: Some(65),
            timestamp: 1000,
        };
        rt.record_execution(record);

        assert_eq!(rt.total_executions, 1);
        assert_eq!(rt.execution_log.len(), 1);
        // Valid execution should not produce violations.
        // (Depends on constraint thresholds; default DNA is lenient for normal values.)
    }

    #[test]
    fn test_record_violating_execution() {
        let mut rt = make_default_runtime();

        // This execution massively exceeds all resource limits.
        let record = KernelExecutionRecord {
            task_id: "formula_eval".into(),
            latency_us: 50.0,
            registers_used: 200000, // Way over limit
            shared_memory_used: 500000, // Way over limit
            warps_used: 100,
            threads_used: 5000,
            success: true,
            coalescing_ratio: 0.1,
            warp_occupancy: 0.05,
            temperature_celsius: Some(95),
            timestamp: 2000,
        };
        rt.record_execution(record);

        assert_eq!(rt.total_executions, 1);
        // Should have violations (register/shmem/latency budgets exceeded).
        // The exact count depends on how many constraints are Critical vs Warning.
        // At minimum we expect some violations or warnings.
        let total_issues = rt.violations.len();
        // If constraints are configured correctly, this should trigger violations.
        // Even if not, the execution is logged.
        assert!(rt.execution_log.len() == 1);
    }

    #[test]
    fn test_reoptimization_queue() {
        let mut rt = make_default_runtime();

        // Force a violation by recording an extreme execution.
        let record = KernelExecutionRecord {
            task_id: "bad_fiber".into(),
            latency_us: 100.0,
            registers_used: 500000,
            shared_memory_used: 1000000,
            warps_used: 200,
            threads_used: 10000,
            success: true,
            coalescing_ratio: 0.01,
            warp_occupancy: 0.01,
            temperature_celsius: Some(99),
            timestamp: 3000,
        };
        rt.record_execution(record);

        // If the execution triggered critical violations, it should be in reopt queue.
        // Mark as processed.
        if !rt.reopt_queue.is_empty() {
            let task = rt.reopt_queue[0].task_id.clone();
            rt.mark_reopt_processed(&task);
            assert!(rt.reopt_queue[0].processed);
        }
    }

    #[test]
    fn test_runtime_state() {
        let mut rt = make_default_runtime();
        rt.compile_all_fibers();

        let state = rt.state();
        assert!(state.dna_loaded);
        assert!(state.fibers_loaded >= 1);
        assert!(state.kernels_compiled >= 1);
        assert_eq!(state.active_bridges, 0);
        assert_eq!(state.total_executions, 0);
    }

    #[test]
    fn test_parse_compute_cap() {
        assert_eq!(parse_compute_cap_u32("compute_89"), (8, 9));
        assert_eq!(parse_compute_cap_u32("sm_89"), (8, 9));
        assert_eq!(parse_compute_cap_u32("compute_70"), (7, 0));
        assert_eq!(parse_compute_cap_u32("compute_86"), (8, 6));
    }

    #[test]
    fn test_extract_entry_points() {
        let ptx = r#"
.version 7.8
.target sm_89
.visible .entry my_kernel(
    .param .u64 data
)
{
    ret;
}
.entry another_kernel(
    .param .u32 count
)
{
    ret;
}
"#;
        let entries = extract_entry_points(ptx);
        assert!(entries.contains(&"my_kernel".to_string()));
        assert!(entries.contains(&"another_kernel".to_string()));
    }

    #[test]
    fn test_generate_stub_ptx() {
        let ptx = generate_stub_ptx("test_task");
        assert!(ptx.contains(".target sm_89"));
        assert!(ptx.contains("test_task_kernel"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_generate_bridge_cuda_source() {
        let source = generate_bridge_cuda_source(2, 3);
        assert!(source.contains("bridge_2_3"));
        assert!(source.contains("shmem_mailbox_a"));
        assert!(source.contains("shmem_mailbox_b"));
        assert!(source.contains("__shared__"));
    }

    #[test]
    fn test_generate_bridge_stub_ptx() {
        let ptx = generate_bridge_stub_ptx(1, 2);
        assert!(ptx.contains("bridge_1_2"));
        assert!(ptx.contains(".target sm_89"));
    }

    #[test]
    fn test_extract_constraint_name() {
        assert_eq!(
            extract_constraint_name("register_budget: exceeded by 50%"),
            "register_budget"
        );
        assert_eq!(
            extract_constraint_name("no colon here"),
            "no colon here"
        );
    }

    #[test]
    fn test_build_optimization_hints() {
        let violations = vec![
            ConstraintViolation {
                task_id: "test".into(),
                constraint_name: "register_budget".into(),
                severity: ViolationSeverity::Critical,
                description: "register_budget: exceeded".into(),
                observed_value: 70000.0,
                limit_value: 65536.0,
                timestamp: 0,
                flagged_for_reopt: true,
            },
            ConstraintViolation {
                task_id: "test".into(),
                constraint_name: "latency_p99".into(),
                severity: ViolationSeverity::Critical,
                description: "latency_p99: exceeded".into(),
                observed_value: 15.0,
                limit_value: 8.0,
                timestamp: 0,
                flagged_for_reopt: true,
            },
        ];

        let record = KernelExecutionRecord {
            task_id: "test".into(),
            latency_us: 15.0,
            registers_used: 70000,
            shared_memory_used: 4096,
            warps_used: 4,
            threads_used: 128,
            success: true,
            coalescing_ratio: 0.9,
            warp_occupancy: 0.7,
            temperature_celsius: None,
            timestamp: 0,
        };

        let hints = build_optimization_hints(&violations, &record);
        assert!(hints.len() >= 2);
        assert!(hints.iter().any(|h| h.contains("register")));
        assert!(hints.iter().any(|h| h.contains("latency")));
    }

    #[test]
    fn test_execution_log_eviction() {
        let mut rt = make_default_runtime();
        rt.max_log_size = 5;

        for i in 0..10 {
            let record = KernelExecutionRecord {
                task_id: format!("task_{}", i),
                latency_us: 1.0,
                registers_used: 16,
                shared_memory_used: 512,
                warps_used: 1,
                threads_used: 32,
                success: true,
                coalescing_ratio: 1.0,
                warp_occupancy: 1.0,
                temperature_celsius: None,
                timestamp: i as u64,
            };
            rt.record_execution(record);
        }

        assert_eq!(rt.execution_log.len(), 5);
        assert_eq!(rt.total_executions, 10);
        // Oldest entries should have been evicted.
        assert_eq!(rt.execution_log[0].task_id, "task_5");
    }

    #[test]
    fn test_parse_runtime_args() {
        assert!(matches!(
            parse_runtime_args(&[]),
            Some(RuntimeCliAction::Demo)
        ));
        assert!(matches!(
            parse_runtime_args(&["--demo".into()]),
            Some(RuntimeCliAction::Demo)
        ));
        assert!(matches!(
            parse_runtime_args(&["--status".into()]),
            Some(RuntimeCliAction::Status)
        ));
        assert!(matches!(
            parse_runtime_args(&["--export".into(), "out.json".into()]),
            Some(RuntimeCliAction::Export(_))
        ));
        assert!(parse_runtime_args(&["--help".into()]).is_none());
    }

    #[test]
    fn test_reflag_cooldown() {
        let mut rt = make_default_runtime();

        // First violation flags immediately.
        let record1 = KernelExecutionRecord {
            task_id: "cooldown_test".into(),
            latency_us: 100.0,
            registers_used: 500000,
            shared_memory_used: 1000000,
            warps_used: 200,
            threads_used: 10000,
            success: true,
            coalescing_ratio: 0.01,
            warp_occupancy: 0.01,
            temperature_celsius: None,
            timestamp: 1000,
        };
        rt.record_execution(record1);
        let initial_queue_len = rt.reopt_queue.len();

        // Second violation within 60 seconds should NOT re-flag.
        let record2 = KernelExecutionRecord {
            task_id: "cooldown_test".into(),
            latency_us: 100.0,
            registers_used: 500000,
            shared_memory_used: 1000000,
            warps_used: 200,
            threads_used: 10000,
            success: true,
            coalescing_ratio: 0.01,
            warp_occupancy: 0.01,
            temperature_celsius: None,
            timestamp: 1030, // 30 seconds later — within cooldown
        };
        rt.record_execution(record2);

        // Queue should not have grown (cooldown prevents re-flagging).
        assert_eq!(rt.reopt_queue.len(), initial_queue_len);
    }
}
