// ============================================================
// Resource Exhaustion Logic — GPU "Soil Nutrient" Management
// ============================================================
//
// GPU registers and shared memory are finite resources analogous
// to soil nutrients in a biological system. When one agent
// "exhausts" a resource on an SM (Streaming Multiprocessor), the
// system must intelligently respond:
//
// 1. PRUNE — Reduce the agent's priority or resource allocation
//    so other agents can thrive.
// 2. BRANCH — Migrate the agent to a different SM where resources
//    are more plentiful ("harvested more effectively").
// 3. THROTTLE — Temporarily limit the agent's execution rate
//    to prevent starvation of co-located agents.
//
// RESOURCE MODEL:
// Each SM has a fixed pool of:
// - Registers:     65,536 per SM (Ampere/Ada) shared across all
//                  threads on the SM. A 128-reg kernel with 256
//                  threads consumes 32,768 regs = 50% of budget.
// - Shared Memory: 48-164 KB per SM (configurable). Used by
//                  working sets, bridges, and scratch space.
// - Warp Slots:    32-64 per SM. Each running warp occupies one
//                  slot; exhaustion means new warps queue.
// - Thread Slots:  2048 per SM (max concurrent threads).
//
// The system monitors per-agent resource consumption and applies
// biologically-inspired policies:
// - "Nutrient depletion" → resource utilization > threshold
// - "Pruning"           → reduce block size or register count
// - "Branching"         → migrate to a different SM
// - "Harvesting"        → reclaim resources from idle agents
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn instant_now() -> Instant { Instant::now() }

// ============================================================
// GPU Resource Model
// ============================================================

/// Resources available on a single Streaming Multiprocessor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmResourcePool {
    /// Total 32-bit registers available.
    pub total_registers: u32,
    /// Total shared memory in bytes.
    pub total_shared_memory: u32,
    /// Maximum concurrent warps.
    pub max_warps: u32,
    /// Maximum concurrent threads.
    pub max_threads: u32,
    /// L1 cache size in bytes (shared with shared memory on some archs).
    pub l1_cache_bytes: u32,
}

impl SmResourcePool {
    /// Default resource pool for an Ada Lovelace SM (sm_89).
    pub fn ada_lovelace() -> Self {
        SmResourcePool {
            total_registers: 65536,
            total_shared_memory: 102400, // 100 KB configurable
            max_warps: 48,
            max_threads: 1536,
            l1_cache_bytes: 131072, // 128 KB
        }
    }

    /// Default resource pool for an Ampere SM (sm_80/86).
    pub fn ampere() -> Self {
        SmResourcePool {
            total_registers: 65536,
            total_shared_memory: 167936, // 164 KB max
            max_warps: 64,
            max_threads: 2048,
            l1_cache_bytes: 196608, // 192 KB
        }
    }

    /// Default resource pool for a Volta SM (sm_70).
    pub fn volta() -> Self {
        SmResourcePool {
            total_registers: 65536,
            total_shared_memory: 98304, // 96 KB
            max_warps: 64,
            max_threads: 2048,
            l1_cache_bytes: 131072, // 128 KB
        }
    }
}

/// Resource consumption by a single agent on an SM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResourceUsage {
    /// Agent identifier.
    pub agent_id: String,
    /// SM index where the agent is currently running.
    pub sm_index: u32,
    /// Registers consumed by this agent's kernel.
    pub registers_used: u32,
    /// Shared memory consumed (bytes).
    pub shared_memory_used: u32,
    /// Warps actively running.
    pub warps_active: u32,
    /// Threads actively running.
    pub threads_active: u32,
    /// Block size (threads per block) used by this agent.
    pub block_size: u32,
    /// Number of blocks launched on this SM.
    pub blocks_on_sm: u32,
    /// Agent's current priority (0 = lowest, 100 = highest).
    pub priority: u32,
    /// When the agent was last active.
    #[serde(skip, default = "instant_now")]
    pub last_active: Instant,
    /// Cumulative execution time in nanoseconds.
    pub cumulative_exec_ns: u64,
    /// Number of times this agent has been pruned.
    pub prune_count: u32,
    /// Number of times this agent has been branched to another SM.
    pub branch_count: u32,
}

impl AgentResourceUsage {
    /// Create a new usage record.
    pub fn new(agent_id: &str, sm_index: u32) -> Self {
        AgentResourceUsage {
            agent_id: agent_id.to_string(),
            sm_index,
            registers_used: 0,
            shared_memory_used: 0,
            warps_active: 0,
            threads_active: 0,
            block_size: 128,
            blocks_on_sm: 1,
            priority: 50,
            last_active: Instant::now(),
            cumulative_exec_ns: 0,
            prune_count: 0,
            branch_count: 0,
        }
    }

    /// Register count per thread.
    pub fn regs_per_thread(&self) -> u32 {
        if self.threads_active > 0 {
            self.registers_used / self.threads_active
        } else {
            0
        }
    }
}

// ============================================================
// Resource Utilization Snapshot
// ============================================================

/// A point-in-time snapshot of resource utilization on one SM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmUtilizationSnapshot {
    /// SM index.
    pub sm_index: u32,
    /// Register utilization (0.0 to 1.0).
    pub register_utilization: f64,
    /// Shared memory utilization (0.0 to 1.0).
    pub shared_memory_utilization: f64,
    /// Warp occupancy (0.0 to 1.0).
    pub warp_occupancy: f64,
    /// Thread occupancy (0.0 to 1.0).
    pub thread_occupancy: f64,
    /// Number of agents on this SM.
    pub agent_count: usize,
    /// Overall "nutrient" health score (0.0 = depleted, 1.0 = abundant).
    pub nutrient_score: f64,
    /// When this snapshot was taken.
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
}

impl SmUtilizationSnapshot {
    /// Compute the nutrient score from utilization metrics.
    ///
    /// The nutrient score is the complement of the maximum
    /// utilization across all resource types. A score near 0
    /// means the SM is nearly exhausted; near 1 means abundant.
    pub fn compute_nutrient_score(&mut self) {
        let max_util = self
            .register_utilization
            .max(self.shared_memory_utilization)
            .max(self.warp_occupancy)
            .max(self.thread_occupancy);
        self.nutrient_score = 1.0 - max_util;
    }
}

// ============================================================
// Exhaustion Policy Actions
// ============================================================

/// Actions the system can take when resources are exhausted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExhaustionAction {
    /// Reduce the agent's block size to free registers/warps.
    Prune {
        agent_id: String,
        old_block_size: u32,
        new_block_size: u32,
        old_priority: u32,
        new_priority: u32,
    },
    /// Migrate the agent to a different SM with more resources.
    Branch {
        agent_id: String,
        source_sm: u32,
        target_sm: u32,
        reason: String,
    },
    /// Temporarily throttle the agent's execution rate.
    Throttle {
        agent_id: String,
        sm_index: u32,
        throttle_factor: f64, // 0.0 = fully throttled, 1.0 = no throttle
        duration: Duration,
    },
    /// Reclaim resources from an idle agent.
    Harvest {
        agent_id: String,
        sm_index: u32,
        registers_reclaimed: u32,
        shared_memory_reclaimed: u32,
    },
    /// No action needed — resources are healthy.
    NoAction,
}

// ============================================================
// Resource Exhaustion Manager
// ============================================================

/// Thresholds for triggering exhaustion actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExhaustionThresholds {
    /// Register utilization above this triggers pruning.
    pub register_prune_threshold: f64,
    /// Shared memory utilization above this triggers pruning.
    pub shmem_prune_threshold: f64,
    /// Warp occupancy above this triggers branching.
    pub warp_branch_threshold: f64,
    /// Overall nutrient score below this triggers action.
    pub nutrient_critical_threshold: f64,
    /// Idle time (seconds) before an agent can be harvested.
    pub idle_harvest_seconds: f64,
    /// Minimum block size after pruning (don't go below one warp).
    pub min_block_size: u32,
    /// Maximum number of prune actions per agent per epoch.
    pub max_prunes_per_epoch: u32,
    /// Throttle factor when throttling is applied.
    pub throttle_factor: f64,
    /// Throttle duration.
    pub throttle_duration: Duration,
}

impl Default for ExhaustionThresholds {
    fn default() -> Self {
        ExhaustionThresholds {
            register_prune_threshold: 0.85,
            shmem_prune_threshold: 0.90,
            warp_branch_threshold: 0.90,
            nutrient_critical_threshold: 0.15,
            idle_harvest_seconds: 5.0,
            min_block_size: 32,
            max_prunes_per_epoch: 3,
            throttle_factor: 0.5,
            throttle_duration: Duration::from_secs(2),
        }
    }
}

/// Manages resource exhaustion detection and response across all SMs.
pub struct ResourceExhaustionManager {
    /// Resource pool specification per SM.
    resource_pool: SmResourcePool,
    /// Number of SMs on the GPU.
    sm_count: u32,
    /// Per-agent resource usage, keyed by agent_id.
    agent_usage: HashMap<String, AgentResourceUsage>,
    /// Latest utilization snapshot per SM.
    sm_snapshots: HashMap<u32, SmUtilizationSnapshot>,
    /// Exhaustion policy thresholds.
    thresholds: ExhaustionThresholds,
    /// History of actions taken (for auditing).
    action_history: Vec<(u64, ExhaustionAction)>,
    /// Prune count per agent in the current epoch.
    epoch_prune_counts: HashMap<String, u32>,
    /// When the current epoch started.
    epoch_start: Instant,
    /// Epoch duration (prune counts reset).
    epoch_duration: Duration,
}

impl ResourceExhaustionManager {
    /// Create a new manager.
    ///
    /// # Arguments
    /// * `resource_pool` — resource specification for each SM
    /// * `sm_count` — number of SMs on the GPU
    /// * `thresholds` — exhaustion policy thresholds
    pub fn new(
        resource_pool: SmResourcePool,
        sm_count: u32,
        thresholds: ExhaustionThresholds,
    ) -> Self {
        ResourceExhaustionManager {
            resource_pool,
            sm_count,
            agent_usage: HashMap::new(),
            sm_snapshots: HashMap::new(),
            thresholds,
            action_history: Vec::new(),
            epoch_prune_counts: HashMap::new(),
            epoch_start: Instant::now(),
            epoch_duration: Duration::from_secs(30),
        }
    }

    /// Register an agent and its initial resource usage.
    pub fn register_agent(&mut self, usage: AgentResourceUsage) {
        self.agent_usage.insert(usage.agent_id.clone(), usage);
    }

    /// Update resource usage for an existing agent.
    pub fn update_usage(&mut self, agent_id: &str, updater: impl FnOnce(&mut AgentResourceUsage)) {
        if let Some(usage) = self.agent_usage.get_mut(agent_id) {
            updater(usage);
            usage.last_active = Instant::now();
        }
    }

    /// Take a utilization snapshot of all SMs.
    ///
    /// This aggregates per-agent usage into per-SM utilization
    /// metrics and computes nutrient scores.
    pub fn snapshot_all_sms(&mut self) {
        let now = Instant::now();

        for sm in 0..self.sm_count {
            let agents_on_sm: Vec<&AgentResourceUsage> = self
                .agent_usage
                .values()
                .filter(|a| a.sm_index == sm)
                .collect();

            let total_regs: u32 = agents_on_sm.iter().map(|a| a.registers_used).sum();
            let total_shmem: u32 = agents_on_sm.iter().map(|a| a.shared_memory_used).sum();
            let total_warps: u32 = agents_on_sm.iter().map(|a| a.warps_active).sum();
            let total_threads: u32 = agents_on_sm.iter().map(|a| a.threads_active).sum();

            let mut snapshot = SmUtilizationSnapshot {
                sm_index: sm,
                register_utilization: total_regs as f64 / self.resource_pool.total_registers as f64,
                shared_memory_utilization: total_shmem as f64
                    / self.resource_pool.total_shared_memory as f64,
                warp_occupancy: total_warps as f64 / self.resource_pool.max_warps as f64,
                thread_occupancy: total_threads as f64 / self.resource_pool.max_threads as f64,
                agent_count: agents_on_sm.len(),
                nutrient_score: 0.0,
                timestamp: now,
            };
            snapshot.compute_nutrient_score();

            self.sm_snapshots.insert(sm, snapshot);
        }
    }

    /// Evaluate all SMs and agents, returning recommended actions.
    ///
    /// This is the main decision function. Call it periodically
    /// (e.g., every 100ms) to detect exhaustion and respond.
    pub fn evaluate(&mut self) -> Vec<ExhaustionAction> {
        // Reset epoch if needed.
        if self.epoch_start.elapsed() >= self.epoch_duration {
            self.epoch_prune_counts.clear();
            self.epoch_start = Instant::now();
        }

        // Take fresh snapshots.
        self.snapshot_all_sms();

        let mut actions = Vec::new();

        // Collect SM snapshots to avoid borrow issues.
        let sm_snapshots: Vec<SmUtilizationSnapshot> =
            self.sm_snapshots.values().cloned().collect();

        for snapshot in &sm_snapshots {
            // Check if this SM is in critical state.
            if snapshot.nutrient_score >= self.thresholds.nutrient_critical_threshold {
                continue; // SM is healthy
            }

            // Find agents on this SM, sorted by priority (lowest first).
            let mut agents_on_sm: Vec<String> = self
                .agent_usage
                .values()
                .filter(|a| a.sm_index == snapshot.sm_index)
                .map(|a| a.agent_id.clone())
                .collect();

            // Sort by priority ascending — prune low-priority agents first.
            agents_on_sm.sort_by_key(|id| {
                self.agent_usage
                    .get(id)
                    .map(|a| a.priority)
                    .unwrap_or(50)
            });

            for agent_id in &agents_on_sm {
                let action = self.decide_action(agent_id, snapshot);
                if !matches!(action, ExhaustionAction::NoAction) {
                    actions.push(action);
                }
            }
        }

        // Harvest idle agents regardless of SM health.
        let idle_agents = self.find_idle_agents();
        for (agent_id, sm_index) in idle_agents {
            if let Some(usage) = self.agent_usage.get(&agent_id) {
                actions.push(ExhaustionAction::Harvest {
                    agent_id: agent_id.clone(),
                    sm_index,
                    registers_reclaimed: usage.registers_used,
                    shared_memory_reclaimed: usage.shared_memory_used,
                });
            }
        }

        // Record actions.
        let now_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        for action in &actions {
            self.action_history.push((now_epoch, action.clone()));
        }

        actions
    }

    /// Decide what action to take for a specific agent on an exhausted SM.
    fn decide_action(
        &mut self,
        agent_id: &str,
        snapshot: &SmUtilizationSnapshot,
    ) -> ExhaustionAction {
        // Copy values from immutable borrow up front to avoid borrow conflicts.
        let (block_size, priority, registers_used, shared_memory_used, blocks_on_sm) =
            match self.agent_usage.get(agent_id) {
                Some(u) => (u.block_size, u.priority, u.registers_used, u.shared_memory_used, u.blocks_on_sm),
                None => return ExhaustionAction::NoAction,
            };

        // Check if agent has been pruned too many times this epoch.
        let prune_count = self
            .epoch_prune_counts
            .get(agent_id)
            .copied()
            .unwrap_or(0);

        // STRATEGY 1: Prune — reduce block size.
        // Applicable when register or shared memory utilization is high.
        if (snapshot.register_utilization > self.thresholds.register_prune_threshold
            || snapshot.shared_memory_utilization > self.thresholds.shmem_prune_threshold)
            && prune_count < self.thresholds.max_prunes_per_epoch
            && block_size > self.thresholds.min_block_size
        {
            let new_block_size = (block_size / 2).max(self.thresholds.min_block_size);
            let new_priority = priority.saturating_sub(10);

            *self.epoch_prune_counts.entry(agent_id.to_string()).or_insert(0) += 1;

            // Apply the prune.
            if let Some(u) = self.agent_usage.get_mut(agent_id) {
                u.block_size = new_block_size;
                u.priority = new_priority;
                u.prune_count += 1;
                // Recalculate resource usage based on new block size.
                let ratio = new_block_size as f64 / block_size as f64;
                u.registers_used = (registers_used as f64 * ratio) as u32;
                u.shared_memory_used = (shared_memory_used as f64 * ratio) as u32;
                u.threads_active = new_block_size * blocks_on_sm;
                u.warps_active = u.threads_active / 32;
            }

            return ExhaustionAction::Prune {
                agent_id: agent_id.to_string(),
                old_block_size: block_size,
                new_block_size,
                old_priority: priority,
                new_priority,
            };
        }

        // STRATEGY 2: Branch — migrate to a different SM.
        // Applicable when warp occupancy is saturated.
        if snapshot.warp_occupancy > self.thresholds.warp_branch_threshold {
            if let Some(target_sm) = self.find_best_sm_for_branch(snapshot.sm_index) {
                // Apply the branch.
                if let Some(u) = self.agent_usage.get_mut(agent_id) {
                    u.sm_index = target_sm;
                    u.branch_count += 1;
                }

                return ExhaustionAction::Branch {
                    agent_id: agent_id.to_string(),
                    source_sm: snapshot.sm_index,
                    target_sm,
                    reason: format!(
                        "Warp occupancy {:.0}% exceeds threshold {:.0}%",
                        snapshot.warp_occupancy * 100.0,
                        self.thresholds.warp_branch_threshold * 100.0
                    ),
                };
            }
        }

        // STRATEGY 3: Throttle — slow down the agent.
        // Last resort when pruning and branching aren't possible.
        if snapshot.nutrient_score < self.thresholds.nutrient_critical_threshold * 0.5 {
            return ExhaustionAction::Throttle {
                agent_id: agent_id.to_string(),
                sm_index: snapshot.sm_index,
                throttle_factor: self.thresholds.throttle_factor,
                duration: self.thresholds.throttle_duration,
            };
        }

        ExhaustionAction::NoAction
    }

    /// Find the SM with the highest nutrient score for branching.
    fn find_best_sm_for_branch(&self, exclude_sm: u32) -> Option<u32> {
        self.sm_snapshots
            .values()
            .filter(|s| s.sm_index != exclude_sm)
            .max_by(|a, b| {
                a.nutrient_score
                    .partial_cmp(&b.nutrient_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|s| s.nutrient_score > self.thresholds.nutrient_critical_threshold)
            .map(|s| s.sm_index)
    }

    /// Find agents that have been idle for longer than the harvest threshold.
    fn find_idle_agents(&self) -> Vec<(String, u32)> {
        let idle_threshold = Duration::from_secs_f64(self.thresholds.idle_harvest_seconds);
        self.agent_usage
            .values()
            .filter(|a| a.last_active.elapsed() > idle_threshold)
            .map(|a| (a.agent_id.clone(), a.sm_index))
            .collect()
    }

    /// Apply a harvest action: remove the agent and reclaim resources.
    pub fn apply_harvest(&mut self, agent_id: &str) {
        self.agent_usage.remove(agent_id);
    }

    /// Get the utilization snapshot for a specific SM.
    pub fn get_sm_snapshot(&self, sm_index: u32) -> Option<&SmUtilizationSnapshot> {
        self.sm_snapshots.get(&sm_index)
    }

    /// Get all SM snapshots.
    pub fn all_sm_snapshots(&self) -> Vec<&SmUtilizationSnapshot> {
        self.sm_snapshots.values().collect()
    }

    /// Get resource usage for a specific agent.
    pub fn get_agent_usage(&self, agent_id: &str) -> Option<&AgentResourceUsage> {
        self.agent_usage.get(agent_id)
    }

    /// Get all agents on a specific SM.
    pub fn agents_on_sm(&self, sm_index: u32) -> Vec<&AgentResourceUsage> {
        self.agent_usage
            .values()
            .filter(|a| a.sm_index == sm_index)
            .collect()
    }

    /// Get overall system statistics.
    pub fn stats(&self) -> ResourceExhaustionStats {
        let total_agents = self.agent_usage.len();
        let total_prunes: u32 = self.agent_usage.values().map(|a| a.prune_count).sum();
        let total_branches: u32 = self.agent_usage.values().map(|a| a.branch_count).sum();

        let avg_nutrient = if self.sm_snapshots.is_empty() {
            1.0
        } else {
            self.sm_snapshots.values().map(|s| s.nutrient_score).sum::<f64>()
                / self.sm_snapshots.len() as f64
        };

        let critical_sms = self
            .sm_snapshots
            .values()
            .filter(|s| s.nutrient_score < self.thresholds.nutrient_critical_threshold)
            .count();

        ResourceExhaustionStats {
            total_agents,
            total_prunes,
            total_branches,
            total_actions: self.action_history.len(),
            avg_nutrient_score: avg_nutrient,
            critical_sm_count: critical_sms,
            sm_count: self.sm_count,
        }
    }

    /// Get the action history.
    pub fn action_history(&self) -> &[(u64, ExhaustionAction)] {
        &self.action_history
    }
}

/// Aggregated statistics from the exhaustion manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceExhaustionStats {
    pub total_agents: usize,
    pub total_prunes: u32,
    pub total_branches: u32,
    pub total_actions: usize,
    pub avg_nutrient_score: f64,
    pub critical_sm_count: usize,
    pub sm_count: u32,
}

// ============================================================
// SM Rebalancer — Periodic Load Balancing
// ============================================================

/// Periodically rebalances agents across SMs to equalize
/// nutrient scores (resource availability).
///
/// This is a higher-level optimization that runs less frequently
/// than the exhaustion manager's evaluate() function. It considers
/// the global distribution of agents across all SMs and suggests
/// migrations to improve overall balance.
pub struct SmRebalancer {
    /// Target nutrient score imbalance (max difference between SMs).
    pub max_imbalance: f64,
    /// Minimum improvement required to justify a migration.
    pub min_improvement: f64,
    /// History of rebalancing actions.
    rebalance_history: Vec<RebalanceAction>,
}

/// A single rebalancing action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceAction {
    pub agent_id: String,
    pub from_sm: u32,
    pub to_sm: u32,
    pub nutrient_improvement: f64,
    pub timestamp: u64,
}

impl SmRebalancer {
    /// Create a new rebalancer.
    pub fn new(max_imbalance: f64, min_improvement: f64) -> Self {
        SmRebalancer {
            max_imbalance,
            min_improvement,
            rebalance_history: Vec::new(),
        }
    }

    /// Suggest rebalancing actions based on current SM state.
    ///
    /// Returns a list of suggested migrations. The caller is
    /// responsible for applying them.
    pub fn suggest_rebalance(
        &mut self,
        manager: &ResourceExhaustionManager,
    ) -> Vec<RebalanceAction> {
        let snapshots = manager.all_sm_snapshots();
        if snapshots.len() < 2 {
            return Vec::new();
        }

        // Find the most and least loaded SMs.
        let mut sorted: Vec<&SmUtilizationSnapshot> = snapshots;
        sorted.sort_by(|a, b| {
            a.nutrient_score
                .partial_cmp(&b.nutrient_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let most_loaded = sorted.first().unwrap();
        let least_loaded = sorted.last().unwrap();

        let imbalance = least_loaded.nutrient_score - most_loaded.nutrient_score;

        if imbalance < self.max_imbalance {
            return Vec::new(); // Already balanced enough.
        }

        // Find the lowest-priority agent on the most loaded SM.
        let agents = manager.agents_on_sm(most_loaded.sm_index);
        let candidate = agents.iter().min_by_key(|a| a.priority);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut actions = Vec::new();

        if let Some(agent) = candidate {
            let improvement = imbalance * 0.5; // Conservative estimate
            if improvement >= self.min_improvement {
                let action = RebalanceAction {
                    agent_id: agent.agent_id.clone(),
                    from_sm: most_loaded.sm_index,
                    to_sm: least_loaded.sm_index,
                    nutrient_improvement: improvement,
                    timestamp: now,
                };
                self.rebalance_history.push(action.clone());
                actions.push(action);
            }
        }

        actions
    }

    /// Get the rebalance history.
    pub fn history(&self) -> &[RebalanceAction] {
        &self.rebalance_history
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> ResourceExhaustionManager {
        ResourceExhaustionManager::new(
            SmResourcePool::ada_lovelace(),
            4,
            ExhaustionThresholds::default(),
        )
    }

    #[test]
    fn test_sm_resource_pool_presets() {
        let ada = SmResourcePool::ada_lovelace();
        assert_eq!(ada.total_registers, 65536);
        assert!(ada.total_shared_memory > 0);

        let ampere = SmResourcePool::ampere();
        assert_eq!(ampere.max_threads, 2048);

        let volta = SmResourcePool::volta();
        assert_eq!(volta.total_registers, 65536);
    }

    #[test]
    fn test_agent_registration() {
        let mut manager = create_test_manager();

        let agent = AgentResourceUsage::new("agent_1", 0);
        manager.register_agent(agent);

        let usage = manager.get_agent_usage("agent_1");
        assert!(usage.is_some());
        assert_eq!(usage.unwrap().sm_index, 0);
    }

    #[test]
    fn test_usage_update() {
        let mut manager = create_test_manager();
        manager.register_agent(AgentResourceUsage::new("agent_1", 0));

        manager.update_usage("agent_1", |u| {
            u.registers_used = 16384;
            u.shared_memory_used = 8192;
            u.warps_active = 8;
            u.threads_active = 256;
        });

        let usage = manager.get_agent_usage("agent_1").unwrap();
        assert_eq!(usage.registers_used, 16384);
        assert_eq!(usage.shared_memory_used, 8192);
    }

    #[test]
    fn test_snapshot_computation() {
        let mut manager = create_test_manager();

        // Place a heavy agent on SM 0.
        let mut agent = AgentResourceUsage::new("heavy_agent", 0);
        agent.registers_used = 60000; // ~91% of 65536
        agent.shared_memory_used = 95000; // ~93% of 102400
        agent.warps_active = 44; // ~92% of 48
        agent.threads_active = 1408; // ~92% of 1536
        manager.register_agent(agent);

        manager.snapshot_all_sms();

        let snap = manager.get_sm_snapshot(0).unwrap();
        assert!(snap.register_utilization > 0.90);
        assert!(snap.shared_memory_utilization > 0.90);
        assert!(snap.nutrient_score < 0.10, "SM should be nearly depleted");
    }

    #[test]
    fn test_prune_action() {
        let mut manager = create_test_manager();

        let mut agent = AgentResourceUsage::new("greedy_agent", 0);
        agent.registers_used = 60000;
        agent.shared_memory_used = 95000;
        agent.warps_active = 44;
        agent.threads_active = 1408;
        agent.block_size = 256;
        agent.priority = 30;
        manager.register_agent(agent);

        let actions = manager.evaluate();

        // Should recommend pruning the greedy agent.
        let has_prune = actions.iter().any(|a| matches!(a, ExhaustionAction::Prune { .. }));
        assert!(has_prune, "Should recommend pruning: {:?}", actions);
    }

    #[test]
    fn test_branch_action() {
        let mut manager = create_test_manager();

        // Overload SM 0 with warps.
        let mut agent = AgentResourceUsage::new("warp_hog", 0);
        agent.warps_active = 46; // 96% occupancy
        agent.threads_active = 1472;
        agent.registers_used = 30000; // moderate registers (won't trigger prune first)
        agent.shared_memory_used = 40000; // moderate shmem
        agent.block_size = 32; // already minimum, can't prune
        agent.priority = 20;
        manager.register_agent(agent);

        let actions = manager.evaluate();

        // Should recommend branching to another SM.
        let has_branch = actions.iter().any(|a| matches!(a, ExhaustionAction::Branch { .. }));
        assert!(has_branch, "Should recommend branching: {:?}", actions);
    }

    #[test]
    fn test_healthy_sm_no_action() {
        let mut manager = create_test_manager();

        // Light agent on SM 0.
        let mut agent = AgentResourceUsage::new("light_agent", 0);
        agent.registers_used = 8192;
        agent.shared_memory_used = 4096;
        agent.warps_active = 4;
        agent.threads_active = 128;
        manager.register_agent(agent);

        let actions = manager.evaluate();

        // Filter out harvest actions (agent might be "idle" due to timing).
        let non_harvest: Vec<_> = actions
            .iter()
            .filter(|a| !matches!(a, ExhaustionAction::Harvest { .. }))
            .collect();

        assert!(
            non_harvest.is_empty() || non_harvest.iter().all(|a| matches!(a, ExhaustionAction::NoAction)),
            "Healthy SM should not trigger prune/branch/throttle: {:?}",
            non_harvest
        );
    }

    #[test]
    fn test_nutrient_score_computation() {
        let mut snap = SmUtilizationSnapshot {
            sm_index: 0,
            register_utilization: 0.5,
            shared_memory_utilization: 0.3,
            warp_occupancy: 0.7,
            thread_occupancy: 0.6,
            agent_count: 2,
            nutrient_score: 0.0,
            timestamp: Instant::now(),
        };

        snap.compute_nutrient_score();
        // Max utilization is 0.7 (warps), so nutrient = 0.3.
        assert!((snap.nutrient_score - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_rebalancer() {
        let mut manager = create_test_manager();

        // Heavy agent on SM 0.
        let mut heavy = AgentResourceUsage::new("heavy", 0);
        heavy.registers_used = 60000;
        heavy.shared_memory_used = 90000;
        heavy.warps_active = 44;
        heavy.threads_active = 1408;
        heavy.priority = 20;
        manager.register_agent(heavy);

        // Light agent on SM 1.
        let mut light = AgentResourceUsage::new("light", 1);
        light.registers_used = 4096;
        light.shared_memory_used = 2048;
        light.warps_active = 2;
        light.threads_active = 64;
        light.priority = 80;
        manager.register_agent(light);

        manager.snapshot_all_sms();

        let mut rebalancer = SmRebalancer::new(0.30, 0.10);
        let suggestions = rebalancer.suggest_rebalance(&manager);

        // Should suggest moving the heavy agent from SM 0 to SM 2 or 3.
        if !suggestions.is_empty() {
            assert_eq!(suggestions[0].from_sm, 0);
            assert_ne!(suggestions[0].to_sm, 0);
        }
    }

    #[test]
    fn test_stats() {
        let mut manager = create_test_manager();
        manager.register_agent(AgentResourceUsage::new("a1", 0));
        manager.register_agent(AgentResourceUsage::new("a2", 1));

        let stats = manager.stats();
        assert_eq!(stats.total_agents, 2);
        assert_eq!(stats.sm_count, 4);
    }
}
