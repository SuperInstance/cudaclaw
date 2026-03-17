// ============================================================
// Ramify Module — Adaptive GPU Kernel Specialization Engine
// ============================================================
//
// The Ramify module is cudaclaw's runtime adaptation layer. It
// observes data patterns, inter-agent communication, and GPU
// resource utilization, then dynamically responds by:
//
// 1. BRANCHING — Recompiling/relinking specialized PTX sub-kernels
//    on-the-fly when it detects that the current kernel constants
//    are suboptimal for the observed access pattern.
//
// 2. BRIDGING — Establishing Direct GPU Shared Memory Bridges
//    between agents that frequently share data, bypassing the
//    ~400-cycle global memory path in favor of ~5-cycle shared
//    memory access.
//
// 3. RESOURCE MANAGEMENT — Treating GPU registers and shared
//    memory as "soil nutrients" and intelligently pruning,
//    branching, or throttling agents that exhaust resources on
//    a given SM (Streaming Multiprocessor).
//
// ARCHITECTURE:
//
//   ┌─────────────────────────────────────────────────────┐
//   │                  RamifyEngine                       │
//   │                                                     │
//   │  ┌──────────────┐  ┌───────────┐  ┌──────────────┐ │
//   │  │ BranchRegistry│  │BridgeMgr  │  │ResourceMgr   │ │
//   │  │              │  │           │  │              │ │
//   │  │ PTX Templates│  │ ShmemBudget│ │ SM Snapshots │ │
//   │  │ AccessWindow │  │ CommTracker│ │ Agent Usage  │ │
//   │  │ Compiled     │  │ Hot Pairs  │ │ Nutrient     │ │
//   │  │ Branches     │  │ Bridges    │ │ Scores       │ │
//   │  └──────────────┘  └───────────┘  └──────────────┘ │
//   │                                                     │
//   │              ┌─────────────┐                        │
//   │              │ SmRebalancer │                        │
//   │              └─────────────┘                        │
//   └─────────────────────────────────────────────────────┘
//
// USAGE:
//   let mut engine = RamifyEngine::new(config);
//   engine.register_default_templates();
//
//   // In the command processing loop:
//   engine.observe_cell_access(cell_index);
//   engine.record_agent_transfer(src, dst, bytes);
//   let events = engine.tick();
//
//   // Periodically:
//   let stats = engine.stats();
//
// ============================================================

pub mod ptx_branching;
pub mod resource_exhaustion;
pub mod shared_memory_bridge;

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ptx_branching::{
    AccessPattern, BranchRegistry, BranchRegistryStats, CompiledBranch,
};
use resource_exhaustion::{
    AgentResourceUsage, ExhaustionAction, ExhaustionThresholds,
    ResourceExhaustionManager, ResourceExhaustionStats, SmRebalancer,
    SmResourcePool,
};
use shared_memory_bridge::{
    AgentEndpoint, AgentPair, BridgeConfig, BridgeEvent, BridgeManager,
    BridgeManagerStats, CoLocationHint,
};

// ============================================================
// Engine Configuration
// ============================================================

/// Configuration for the Ramify engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamifyConfig {
    /// Spreadsheet grid column count (for pattern detection).
    pub grid_cols: u32,
    /// Number of recent accesses to track for pattern detection.
    pub access_window_size: usize,
    /// Number of SMs on the GPU.
    pub sm_count: u32,
    /// Total shared memory per SM in bytes.
    pub shmem_per_sm: u32,
    /// Shared memory reserved for kernel working sets per SM.
    pub shmem_reserved_per_sm: u32,
    /// Maximum simultaneous shared memory bridges.
    pub max_bridges: usize,
    /// GPU architecture preset for resource pool.
    pub gpu_arch: GpuArch,
    /// Exhaustion policy thresholds.
    pub exhaustion_thresholds: ExhaustionThresholds,
    /// How often to run the tick cycle (target interval).
    pub tick_interval: Duration,
    /// Whether to enable PTX branching.
    pub enable_branching: bool,
    /// Whether to enable shared memory bridges.
    pub enable_bridges: bool,
    /// Whether to enable resource exhaustion management.
    pub enable_resource_mgmt: bool,
    /// Whether to enable periodic SM rebalancing.
    pub enable_rebalancing: bool,
    /// Rebalancer: maximum nutrient score imbalance before action.
    pub rebalance_max_imbalance: f64,
    /// Rebalancer: minimum improvement to justify migration.
    pub rebalance_min_improvement: f64,
}

/// GPU architecture preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArch {
    Volta,       // sm_70
    Ampere,      // sm_80/86
    AdaLovelace, // sm_89
}

impl Default for RamifyConfig {
    fn default() -> Self {
        RamifyConfig {
            grid_cols: 256,
            access_window_size: 512,
            sm_count: 128,
            shmem_per_sm: 102400,
            shmem_reserved_per_sm: 65536,
            max_bridges: 16,
            gpu_arch: GpuArch::AdaLovelace,
            exhaustion_thresholds: ExhaustionThresholds::default(),
            tick_interval: Duration::from_millis(100),
            enable_branching: true,
            enable_bridges: true,
            enable_resource_mgmt: true,
            enable_rebalancing: true,
            rebalance_max_imbalance: 0.30,
            rebalance_min_improvement: 0.10,
        }
    }
}

impl RamifyConfig {
    /// Create a config for a known GPU model.
    pub fn for_gpu(model: &str) -> Self {
        let mut config = Self::default();
        match model.to_lowercase().as_str() {
            s if s.contains("4090") || s.contains("4080") => {
                config.gpu_arch = GpuArch::AdaLovelace;
                config.sm_count = 128;
                config.shmem_per_sm = 102400;
            }
            s if s.contains("3090") || s.contains("3080") || s.contains("a100") => {
                config.gpu_arch = GpuArch::Ampere;
                config.sm_count = 82;
                config.shmem_per_sm = 167936;
            }
            s if s.contains("v100") || s.contains("titan v") => {
                config.gpu_arch = GpuArch::Volta;
                config.sm_count = 80;
                config.shmem_per_sm = 98304;
            }
            _ => {} // use defaults
        }
        config
    }
}

// ============================================================
// Engine Events
// ============================================================

/// Events emitted by the Ramify engine during a tick cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RamifyEvent {
    /// New PTX branches were compiled for a changed access pattern.
    BranchesCompiled {
        pattern: AccessPattern,
        branch_count: usize,
        branch_ids: Vec<String>,
    },
    /// A shared memory bridge event occurred.
    Bridge(BridgeEvent),
    /// A resource exhaustion action was taken.
    Exhaustion(ExhaustionAction),
    /// The SM rebalancer suggested a migration.
    Rebalance {
        agent_id: String,
        from_sm: u32,
        to_sm: u32,
        improvement: f64,
    },
}

// ============================================================
// Ramify Engine
// ============================================================

/// The main Ramify engine orchestrating all three subsystems.
pub struct RamifyEngine {
    /// Configuration.
    config: RamifyConfig,
    /// PTX branch registry (Lower-Level Branching).
    branch_registry: BranchRegistry,
    /// Shared memory bridge manager (Fast Interconnects).
    bridge_manager: BridgeManager,
    /// Resource exhaustion manager (Soil Nutrients).
    resource_manager: ResourceExhaustionManager,
    /// SM load rebalancer.
    rebalancer: SmRebalancer,
    /// Last tick timestamp.
    last_tick: Instant,
    /// Total tick count.
    tick_count: u64,
    /// Total events emitted.
    total_events: u64,
    /// Engine start time.
    start_time: Instant,
}

impl RamifyEngine {
    /// Create a new Ramify engine with the given configuration.
    pub fn new(config: RamifyConfig) -> Self {
        let resource_pool = match config.gpu_arch {
            GpuArch::Volta => SmResourcePool::volta(),
            GpuArch::Ampere => SmResourcePool::ampere(),
            GpuArch::AdaLovelace => SmResourcePool::ada_lovelace(),
        };

        RamifyEngine {
            branch_registry: BranchRegistry::new(
                config.access_window_size,
                config.grid_cols,
            ),
            bridge_manager: BridgeManager::new(
                config.sm_count,
                config.shmem_per_sm,
                config.shmem_reserved_per_sm,
                config.max_bridges,
            ),
            resource_manager: ResourceExhaustionManager::new(
                resource_pool,
                config.sm_count,
                config.exhaustion_thresholds.clone(),
            ),
            rebalancer: SmRebalancer::new(
                config.rebalance_max_imbalance,
                config.rebalance_min_improvement,
            ),
            config,
            last_tick: Instant::now(),
            tick_count: 0,
            total_events: 0,
            start_time: Instant::now(),
        }
    }

    /// Register the default PTX templates for cell update and CRDT merge.
    pub fn register_default_templates(&mut self) {
        if self.config.enable_branching {
            self.branch_registry
                .register_template(ptx_branching::default_cell_update_template());
            self.branch_registry
                .register_template(ptx_branching::default_crdt_merge_template());
        }
    }

    // --------------------------------------------------------
    // Observation API — call these from the command processing loop
    // --------------------------------------------------------

    /// Observe a cell access for pattern detection.
    ///
    /// The branch registry tracks recent accesses and may trigger
    /// PTX recompilation if the detected pattern changes.
    ///
    /// Returns compiled branches if recompilation occurred.
    pub fn observe_cell_access(&mut self, cell_index: u32) -> Option<Vec<CompiledBranch>> {
        if !self.config.enable_branching {
            return None;
        }
        self.branch_registry.observe_access(cell_index)
    }

    /// Record a data transfer between two agents.
    ///
    /// The bridge manager tracks inter-agent communication frequency
    /// and may establish shared memory bridges for hot pairs.
    pub fn record_agent_transfer(
        &mut self,
        src: &AgentEndpoint,
        dst: &AgentEndpoint,
        bytes: u64,
    ) {
        if !self.config.enable_bridges {
            return;
        }
        self.bridge_manager.record_transfer(src, dst, bytes);
    }

    /// Register an agent's resource usage on an SM.
    pub fn register_agent(&mut self, usage: AgentResourceUsage) {
        if self.config.enable_resource_mgmt {
            self.resource_manager.register_agent(usage);
        }
    }

    /// Update resource usage for an existing agent.
    pub fn update_agent_usage(
        &mut self,
        agent_id: &str,
        updater: impl FnOnce(&mut AgentResourceUsage),
    ) {
        if self.config.enable_resource_mgmt {
            self.resource_manager.update_usage(agent_id, updater);
        }
    }

    // --------------------------------------------------------
    // Tick — periodic evaluation and action
    // --------------------------------------------------------

    /// Run a tick cycle: evaluate all subsystems and return events.
    ///
    /// Call this periodically (every `config.tick_interval`).
    pub fn tick(&mut self) -> Vec<RamifyEvent> {
        self.tick_count += 1;
        self.last_tick = Instant::now();

        let mut events = Vec::new();

        // 1. Bridge management tick.
        if self.config.enable_bridges {
            let bridge_events = self.bridge_manager.tick();
            for be in bridge_events {
                events.push(RamifyEvent::Bridge(be));
            }
        }

        // 2. Resource exhaustion evaluation.
        if self.config.enable_resource_mgmt {
            let actions = self.resource_manager.evaluate();
            for action in actions {
                // Apply harvest actions automatically.
                if let ExhaustionAction::Harvest { ref agent_id, .. } = action {
                    self.resource_manager.apply_harvest(agent_id);
                }
                events.push(RamifyEvent::Exhaustion(action));
            }
        }

        // 3. SM rebalancing (less frequent — every 10 ticks).
        if self.config.enable_rebalancing && self.tick_count % 10 == 0 {
            let suggestions = self.rebalancer.suggest_rebalance(&self.resource_manager);
            for suggestion in suggestions {
                // Apply the migration.
                self.resource_manager
                    .update_usage(&suggestion.agent_id, |u| {
                        u.sm_index = suggestion.to_sm;
                        u.branch_count += 1;
                    });
                events.push(RamifyEvent::Rebalance {
                    agent_id: suggestion.agent_id,
                    from_sm: suggestion.from_sm,
                    to_sm: suggestion.to_sm,
                    improvement: suggestion.nutrient_improvement,
                });
            }
        }

        self.total_events += events.len() as u64;
        events
    }

    // --------------------------------------------------------
    // Query API
    // --------------------------------------------------------

    /// Get the currently detected access pattern.
    pub fn current_pattern(&self) -> Option<AccessPattern> {
        self.branch_registry.stats().current_pattern
    }

    /// Get the active compiled branch for a template.
    pub fn active_branch(&self, template_id: &str) -> Option<&CompiledBranch> {
        self.branch_registry.get_active_branch(template_id)
    }

    /// Get all compiled branches.
    pub fn all_branches(&self) -> Vec<&CompiledBranch> {
        self.branch_registry.all_branches()
    }

    /// Get the bridge for a specific agent pair.
    pub fn get_bridge(
        &self,
        a: &AgentEndpoint,
        b: &AgentEndpoint,
    ) -> Option<&shared_memory_bridge::SharedMemoryBridge> {
        let pair = AgentPair::new(a.clone(), b.clone());
        self.bridge_manager.get_bridge_for_pair(&pair)
    }

    /// Get all active bridges.
    pub fn active_bridges(&self) -> Vec<&shared_memory_bridge::SharedMemoryBridge> {
        self.bridge_manager.active_bridges()
    }

    /// Generate co-location hints for the CUDA kernel launcher.
    pub fn colocation_hints(&self) -> Vec<CoLocationHint> {
        shared_memory_bridge::generate_colocation_hints(&self.bridge_manager)
    }

    /// Get resource usage for a specific agent.
    pub fn agent_usage(
        &self,
        agent_id: &str,
    ) -> Option<&AgentResourceUsage> {
        self.resource_manager.get_agent_usage(agent_id)
    }

    /// Get the SM utilization snapshot.
    pub fn sm_snapshot(
        &self,
        sm_index: u32,
    ) -> Option<&resource_exhaustion::SmUtilizationSnapshot> {
        self.resource_manager.get_sm_snapshot(sm_index)
    }

    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    /// Get aggregated statistics from all subsystems.
    pub fn stats(&self) -> RamifyStats {
        RamifyStats {
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            tick_count: self.tick_count,
            total_events: self.total_events,
            branching: self.branch_registry.stats(),
            bridges: self.bridge_manager.stats(),
            resources: self.resource_manager.stats(),
        }
    }

    /// Get the engine configuration.
    pub fn config(&self) -> &RamifyConfig {
        &self.config
    }
}

/// Aggregated statistics from all Ramify subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamifyStats {
    pub uptime_seconds: f64,
    pub tick_count: u64,
    pub total_events: u64,
    pub branching: BranchRegistryStats,
    pub bridges: BridgeManagerStats,
    pub resources: ResourceExhaustionStats,
}

// ============================================================
// CLI Integration
// ============================================================

/// Print help text for the `ramify` CLI subcommand.
pub fn print_ramify_help() {
    println!("cudaclaw ramify — Adaptive GPU Kernel Specialization Engine");
    println!();
    println!("USAGE:");
    println!("  cudaclaw ramify [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --status              Show current Ramify engine status");
    println!("  --gpu <MODEL>         GPU model for config preset (e.g., 'rtx4090', 'a100')");
    println!("  --grid-cols <N>       Spreadsheet grid column count (default: 256)");
    println!("  --sm-count <N>        Number of SMs on the GPU (default: 128)");
    println!("  --max-bridges <N>     Maximum simultaneous bridges (default: 16)");
    println!("  --no-branching        Disable PTX branching");
    println!("  --no-bridges          Disable shared memory bridges");
    println!("  --no-resource-mgmt    Disable resource exhaustion management");
    println!("  --demo                Run a demonstration of all three subsystems");
    println!("  --help, -h            Show this help message");
}

/// Parse CLI arguments for the `ramify` subcommand.
pub fn parse_ramify_args(args: &[String]) -> Option<RamifyCliAction> {
    if args.is_empty() {
        return Some(RamifyCliAction::Demo);
    }

    let mut config = RamifyConfig::default();
    let mut action = RamifyCliAction::Demo;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--status" => action = RamifyCliAction::Status,
            "--demo" => action = RamifyCliAction::Demo,
            "--help" | "-h" => return None,
            "--gpu" => {
                i += 1;
                if i < args.len() {
                    config = RamifyConfig::for_gpu(&args[i]);
                }
            }
            "--grid-cols" => {
                i += 1;
                if i < args.len() {
                    config.grid_cols = args[i].parse().unwrap_or(256);
                }
            }
            "--sm-count" => {
                i += 1;
                if i < args.len() {
                    config.sm_count = args[i].parse().unwrap_or(128);
                }
            }
            "--max-bridges" => {
                i += 1;
                if i < args.len() {
                    config.max_bridges = args[i].parse().unwrap_or(16);
                }
            }
            "--no-branching" => config.enable_branching = false,
            "--no-bridges" => config.enable_bridges = false,
            "--no-resource-mgmt" => config.enable_resource_mgmt = false,
            _ => {}
        }
        i += 1;
    }

    Some(match action {
        RamifyCliAction::Status => RamifyCliAction::StatusWithConfig(config),
        _ => RamifyCliAction::DemoWithConfig(config),
    })
}

/// CLI actions for the ramify subcommand.
pub enum RamifyCliAction {
    Status,
    StatusWithConfig(RamifyConfig),
    Demo,
    DemoWithConfig(RamifyConfig),
}

/// Run the Ramify demonstration.
///
/// Simulates all three subsystems: pattern detection and PTX
/// branching, hot pair detection and bridge creation, and
/// resource exhaustion with pruning/branching.
pub fn run_demo(config: RamifyConfig) {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     cudaclaw Ramify Engine — Demonstration Mode         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let mut engine = RamifyEngine::new(config);
    engine.register_default_templates();

    // ── Phase 1: PTX Branching ──────────────────────────────
    println!("━━━ Phase 1: PTX Dynamic Branching ━━━");
    println!("Feeding 512 sequential cell accesses...");

    let mut branches_compiled = 0;
    for i in 0..512u32 {
        if let Some(branches) = engine.observe_cell_access(i) {
            branches_compiled += branches.len();
            for branch in &branches {
                println!(
                    "  ✦ Compiled branch: {} (pattern: {:?}, block_size: {})",
                    branch.branch_id,
                    branch.pattern,
                    branch.constants.get("BLOCK_SIZE").unwrap_or(&"?".to_string()),
                );
            }
        }
    }

    println!(
        "Pattern detected: {:?}",
        engine.current_pattern().unwrap_or(ptx_branching::AccessPattern::Random)
    );
    println!("Branches compiled: {}", branches_compiled);

    // Now switch to a strided pattern.
    println!("\nSwitching to strided pattern (stride=5)...");
    // Need to set cooldown to 0 for demo purposes.
    engine.branch_registry.set_cooldown(Duration::from_millis(0));
    for i in 0..512u32 {
        if let Some(branches) = engine.observe_cell_access(i * 5) {
            branches_compiled += branches.len();
            for branch in &branches {
                println!(
                    "  ✦ Compiled branch: {} (pattern: {:?})",
                    branch.branch_id, branch.pattern,
                );
            }
        }
    }

    let branch_stats = engine.branch_registry.stats();
    println!(
        "Total recompilations: {}, branches: {}\n",
        branch_stats.total_recompilations, branch_stats.branches_compiled
    );

    // ── Phase 2: Shared Memory Bridges ──────────────────────
    println!("━━━ Phase 2: Direct GPU Shared Memory Bridges ━━━");
    println!("Simulating frequent data sharing between agents...");

    let agent_a = AgentEndpoint::new("claw", "spreadsheet_cell_A1");
    let agent_b = AgentEndpoint::new("smpclaw", "geometric_twin_A1");

    // Pump transfers to make the pair hot.
    for _ in 0..500 {
        engine.record_agent_transfer(&agent_a, &agent_b, 128);
    }

    // Tick to detect hot pairs and create bridges.
    let events = engine.tick();
    for event in &events {
        match event {
            RamifyEvent::Bridge(BridgeEvent::Created {
                bridge_id,
                pair_key,
                sm_index,
            }) => {
                println!(
                    "  ✦ Bridge created: {} ({}) on SM {}",
                    bridge_id, pair_key, sm_index
                );
            }
            _ => {}
        }
    }

    let bridge_stats = engine.bridge_manager.stats();
    println!(
        "Active bridges: {}, hot pairs: {}\n",
        bridge_stats.active_bridges, bridge_stats.hot_pairs
    );

    // ── Phase 3: Resource Exhaustion ────────────────────────
    println!("━━━ Phase 3: Resource Exhaustion Logic ━━━");
    println!("Registering agents with varying resource demands...\n");

    // Heavy agent on SM 0.
    let mut heavy = AgentResourceUsage::new("greedy_formula_engine", 0);
    heavy.registers_used = 60000;
    heavy.shared_memory_used = 90000;
    heavy.warps_active = 44;
    heavy.threads_active = 1408;
    heavy.block_size = 256;
    heavy.priority = 30;
    engine.register_agent(heavy);

    // Medium agent on SM 0.
    let mut medium = AgentResourceUsage::new("crdt_merger", 0);
    medium.registers_used = 12000;
    medium.shared_memory_used = 16384;
    medium.warps_active = 8;
    medium.threads_active = 256;
    medium.block_size = 128;
    medium.priority = 50;
    engine.register_agent(medium);

    // Light agent on SM 1.
    let mut light = AgentResourceUsage::new("cell_formatter", 1);
    light.registers_used = 4096;
    light.shared_memory_used = 2048;
    light.warps_active = 2;
    light.threads_active = 64;
    light.block_size = 64;
    light.priority = 80;
    engine.register_agent(light);

    // Evaluate and show actions.
    let events = engine.tick();
    for event in &events {
        match event {
            RamifyEvent::Exhaustion(ExhaustionAction::Prune {
                agent_id,
                old_block_size,
                new_block_size,
                old_priority,
                new_priority,
            }) => {
                println!(
                    "  ✂ PRUNE: {} — block_size {} → {}, priority {} → {}",
                    agent_id, old_block_size, new_block_size, old_priority, new_priority
                );
            }
            RamifyEvent::Exhaustion(ExhaustionAction::Branch {
                agent_id,
                source_sm,
                target_sm,
                reason,
            }) => {
                println!(
                    "  🌱 BRANCH: {} — SM {} → SM {} ({})",
                    agent_id, source_sm, target_sm, reason
                );
            }
            RamifyEvent::Exhaustion(ExhaustionAction::Throttle {
                agent_id,
                sm_index,
                throttle_factor,
                ..
            }) => {
                println!(
                    "  ⏸ THROTTLE: {} on SM {} — factor {:.0}%",
                    agent_id, sm_index, throttle_factor * 100.0
                );
            }
            RamifyEvent::Exhaustion(ExhaustionAction::Harvest {
                agent_id,
                sm_index,
                registers_reclaimed,
                shared_memory_reclaimed,
            }) => {
                println!(
                    "  🌾 HARVEST: {} on SM {} — {} regs, {} bytes shmem reclaimed",
                    agent_id, sm_index, registers_reclaimed, shared_memory_reclaimed
                );
            }
            _ => {}
        }
    }

    // Final stats.
    let resource_stats = engine.resource_manager.stats();
    println!(
        "\nResource stats: {} agents, {} prunes, {} branches, avg nutrient: {:.2}",
        resource_stats.total_agents,
        resource_stats.total_prunes,
        resource_stats.total_branches,
        resource_stats.avg_nutrient_score
    );

    // ── Summary ─────────────────────────────────────────────
    println!("\n━━━ Engine Summary ━━━");
    let stats = engine.stats();
    println!("Uptime:             {:.2}s", stats.uptime_seconds);
    println!("Ticks:              {}", stats.tick_count);
    println!("Total events:       {}", stats.total_events);
    println!("Branches compiled:  {}", stats.branching.branches_compiled);
    println!("Active bridges:     {}", stats.bridges.active_bridges);
    println!("Shmem utilization:  {:.1}%", stats.bridges.shmem_utilization_percent);
    println!(
        "Critical SMs:       {}/{}",
        stats.resources.critical_sm_count, stats.resources.sm_count
    );
    println!();
}

/// Show current Ramify engine status.
pub fn show_status(config: RamifyConfig) {
    println!("cudaclaw Ramify Engine — Status");
    println!("────────────────────────────────");
    println!("GPU Architecture:   {:?}", config.gpu_arch);
    println!("SM Count:           {}", config.sm_count);
    println!("Shared Mem/SM:      {} KB", config.shmem_per_sm / 1024);
    println!("Reserved/SM:        {} KB", config.shmem_reserved_per_sm / 1024);
    println!("Max Bridges:        {}", config.max_bridges);
    println!("Grid Columns:       {}", config.grid_cols);
    println!("Access Window:      {}", config.access_window_size);
    println!("Branching:          {}", if config.enable_branching { "enabled" } else { "disabled" });
    println!("Bridges:            {}", if config.enable_bridges { "enabled" } else { "disabled" });
    println!("Resource Mgmt:      {}", if config.enable_resource_mgmt { "enabled" } else { "disabled" });
    println!("Rebalancing:        {}", if config.enable_rebalancing { "enabled" } else { "disabled" });
    println!("Tick Interval:      {:?}", config.tick_interval);
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = RamifyConfig::default();
        let engine = RamifyEngine::new(config);
        let stats = engine.stats();
        assert_eq!(stats.tick_count, 0);
        assert_eq!(stats.total_events, 0);
    }

    #[test]
    fn test_engine_with_gpu_preset() {
        let config = RamifyConfig::for_gpu("rtx4090");
        assert_eq!(config.gpu_arch, GpuArch::AdaLovelace);
        assert_eq!(config.sm_count, 128);

        let config = RamifyConfig::for_gpu("a100");
        assert_eq!(config.gpu_arch, GpuArch::Ampere);
    }

    #[test]
    fn test_branching_observation() {
        let mut config = RamifyConfig::default();
        config.access_window_size = 64;
        let mut engine = RamifyEngine::new(config);
        engine.register_default_templates();
        engine.branch_registry.set_cooldown(Duration::from_millis(0));

        // Feed sequential data.
        let mut compiled = false;
        for i in 0..128u32 {
            if engine.observe_cell_access(i).is_some() {
                compiled = true;
            }
        }
        assert!(compiled, "Should compile branches for sequential pattern");
    }

    #[test]
    fn test_bridge_observation() {
        let mut config = RamifyConfig::default();
        config.enable_bridges = true;
        let mut engine = RamifyEngine::new(config);

        let a = AgentEndpoint::new("claw", "a1");
        let b = AgentEndpoint::new("bot", "b1");

        for _ in 0..200 {
            engine.record_agent_transfer(&a, &b, 64);
        }

        let events = engine.tick();
        // May or may not create a bridge depending on rate thresholds,
        // but should not panic.
        let _ = events;
    }

    #[test]
    fn test_resource_management() {
        let config = RamifyConfig::default();
        let mut engine = RamifyEngine::new(config);

        let mut agent = AgentResourceUsage::new("test_agent", 0);
        agent.registers_used = 60000;
        agent.shared_memory_used = 90000;
        agent.warps_active = 44;
        agent.threads_active = 1408;
        agent.block_size = 256;
        agent.priority = 30;
        engine.register_agent(agent);

        let events = engine.tick();
        // Should generate some exhaustion actions.
        let has_exhaustion = events
            .iter()
            .any(|e| matches!(e, RamifyEvent::Exhaustion(_)));
        assert!(has_exhaustion, "Should detect resource exhaustion");
    }

    #[test]
    fn test_tick_cycle() {
        let config = RamifyConfig::default();
        let mut engine = RamifyEngine::new(config);

        // Multiple ticks should not panic.
        for _ in 0..20 {
            let _ = engine.tick();
        }

        let stats = engine.stats();
        assert_eq!(stats.tick_count, 20);
    }

    #[test]
    fn test_disabled_subsystems() {
        let mut config = RamifyConfig::default();
        config.enable_branching = false;
        config.enable_bridges = false;
        config.enable_resource_mgmt = false;
        config.enable_rebalancing = false;

        let mut engine = RamifyEngine::new(config);

        // Nothing should happen with everything disabled.
        assert!(engine.observe_cell_access(42).is_none());
        let a = AgentEndpoint::new("claw", "a1");
        let b = AgentEndpoint::new("bot", "b1");
        engine.record_agent_transfer(&a, &b, 64);
        let events = engine.tick();
        assert!(events.is_empty());
    }

    #[test]
    fn test_cli_parsing() {
        let args = vec!["--demo".to_string()];
        let action = parse_ramify_args(&args);
        assert!(action.is_some());

        let args = vec!["--status".to_string()];
        let action = parse_ramify_args(&args);
        assert!(action.is_some());

        let args = vec!["--help".to_string()];
        let action = parse_ramify_args(&args);
        assert!(action.is_none());
    }

    #[test]
    fn test_stats_serialization() {
        let config = RamifyConfig::default();
        let engine = RamifyEngine::new(config);
        let stats = engine.stats();

        // Should serialize to JSON without panicking.
        let json = serde_json::to_string_pretty(&stats).unwrap();
        assert!(json.contains("tick_count"));
        assert!(json.contains("branching"));
        assert!(json.contains("bridges"));
        assert!(json.contains("resources"));
    }
}
