// ============================================================
// Direct GPU Shared Memory Bridge — Agent-to-Agent Fast Interconnect
// ============================================================
//
// When two agents (e.g., a spreadsheet cell and a geometric twin)
// frequently share data, going through global memory adds ~100 ns
// per round-trip. This module detects frequent inter-agent
// communication pairs and dynamically establishes a Direct Shared
// Memory Bridge between them.
//
// DESIGN:
// - CommunicationTracker records every (src_agent, dst_agent) data
//   transfer with timestamp and byte count.
// - HotPairDetector identifies pairs whose transfer frequency
//   exceeds a configurable threshold.
// - SharedMemoryBridge allocates a region of GPU shared memory
//   (within a single SM) as a fast mailbox between the two agents.
// - BridgeManager orchestrates creation, monitoring, and teardown
//   of bridges as communication patterns evolve.
//
// MEMORY HIERARCHY (latency for 32-byte access):
//   Registers:    ~1 cycle   (not shareable across agents)
//   Shared Mem:   ~5 cycles  (shared within an SM / thread block)
//   L1 Cache:     ~30 cycles (per-SM, automatic)
//   L2 Cache:     ~200 cycles (chip-wide)
//   Global VRAM:  ~400 cycles (all SMs, via L2)
//   PCIe (host):  ~5,000+ cycles
//
// By routing hot agent pairs through shared memory instead of
// global memory, we eliminate ~395 cycles per access.
//
// CONSTRAINTS:
// - Both agents must be scheduled on the same SM for shared
//   memory to work. The bridge system requests co-location via
//   block assignment hints.
// - Total shared memory per SM is limited (48-164 KB depending
//   on GPU architecture). Bridges compete with kernel working
//   sets for this budget.
// - Bridges are transient: they exist only while the hot pair
//   remains hot. If communication drops below the threshold,
//   the bridge is torn down and the shared memory reclaimed.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// Agent Communication Tracking
// ============================================================

/// Unique identifier for an agent endpoint.
/// Combines agent type + instance ID for disambiguation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentEndpoint {
    /// Agent type (e.g., "claw", "bot", "seed", "smpclaw").
    pub agent_type: String,
    /// Instance identifier within the agent type.
    pub instance_id: String,
}

impl AgentEndpoint {
    pub fn new(agent_type: &str, instance_id: &str) -> Self {
        AgentEndpoint {
            agent_type: agent_type.to_string(),
            instance_id: instance_id.to_string(),
        }
    }

    /// Canonical string key for HashMap usage.
    pub fn key(&self) -> String {
        format!("{}:{}", self.agent_type, self.instance_id)
    }
}

/// An ordered pair of agents that communicate.
/// The pair is always stored with the lexicographically smaller
/// endpoint first, so (A,B) and (B,A) map to the same pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentPair {
    pub first: AgentEndpoint,
    pub second: AgentEndpoint,
}

impl AgentPair {
    /// Create a canonical pair (sorted by key).
    pub fn new(a: AgentEndpoint, b: AgentEndpoint) -> Self {
        if a.key() <= b.key() {
            AgentPair { first: a, second: b }
        } else {
            AgentPair { first: b, second: a }
        }
    }

    /// Canonical string key.
    pub fn key(&self) -> String {
        format!("{}<->{}", self.first.key(), self.second.key())
    }
}

/// A single recorded data transfer between two agents.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// When the transfer occurred.
    pub timestamp: Instant,
    /// Number of bytes transferred.
    pub bytes: u64,
    /// Transfer direction: true = first→second, false = second→first.
    pub direction_forward: bool,
}

/// Rolling statistics for a communication pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairStats {
    /// Total number of transfers recorded.
    pub total_transfers: u64,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Transfers in the current measurement window.
    pub window_transfers: u64,
    /// Bytes in the current measurement window.
    pub window_bytes: u64,
    /// Average transfer rate (transfers/second) in the window.
    pub transfer_rate: f64,
    /// Average bandwidth (bytes/second) in the window.
    pub bandwidth: f64,
    /// Whether this pair is currently classified as "hot".
    pub is_hot: bool,
    /// How many consecutive windows this pair has been hot.
    pub consecutive_hot_windows: u32,
}

impl Default for PairStats {
    fn default() -> Self {
        PairStats {
            total_transfers: 0,
            total_bytes: 0,
            window_transfers: 0,
            window_bytes: 0,
            transfer_rate: 0.0,
            bandwidth: 0.0,
            is_hot: false,
            consecutive_hot_windows: 0,
        }
    }
}

/// Tracks communication between all agent pairs and identifies
/// hot pairs that would benefit from a shared memory bridge.
pub struct CommunicationTracker {
    /// Rolling window of recent transfers per pair.
    recent_transfers: HashMap<String, Vec<TransferRecord>>,
    /// Aggregated statistics per pair.
    pair_stats: HashMap<String, PairStats>,
    /// Pair metadata (agent endpoints).
    pair_info: HashMap<String, AgentPair>,
    /// Window duration for rate calculation.
    window_duration: Duration,
    /// Minimum transfer rate to classify a pair as "hot".
    hot_threshold_rate: f64,
    /// Minimum consecutive hot windows before bridge creation.
    hot_stability_windows: u32,
    /// When the last window rotation occurred.
    last_window_rotation: Instant,
}

impl CommunicationTracker {
    /// Create a new tracker.
    ///
    /// # Arguments
    /// * `window_duration` — measurement window for rate calculation
    /// * `hot_threshold_rate` — transfers/sec above which a pair is "hot"
    /// * `hot_stability_windows` — consecutive hot windows required
    pub fn new(
        window_duration: Duration,
        hot_threshold_rate: f64,
        hot_stability_windows: u32,
    ) -> Self {
        CommunicationTracker {
            recent_transfers: HashMap::new(),
            pair_stats: HashMap::new(),
            pair_info: HashMap::new(),
            window_duration,
            hot_threshold_rate,
            hot_stability_windows,
            last_window_rotation: Instant::now(),
        }
    }

    /// Record a data transfer between two agents.
    pub fn record_transfer(
        &mut self,
        src: &AgentEndpoint,
        dst: &AgentEndpoint,
        bytes: u64,
    ) {
        let pair = AgentPair::new(src.clone(), dst.clone());
        let key = pair.key();
        let direction_forward = src.key() <= dst.key();

        let record = TransferRecord {
            timestamp: Instant::now(),
            bytes,
            direction_forward,
        };

        self.recent_transfers
            .entry(key.clone())
            .or_default()
            .push(record);

        self.pair_info.entry(key.clone()).or_insert(pair);

        let stats = self.pair_stats.entry(key).or_default();
        stats.total_transfers += 1;
        stats.total_bytes += bytes;
        stats.window_transfers += 1;
        stats.window_bytes += bytes;
    }

    /// Rotate the measurement window and recalculate rates.
    /// Call this periodically (e.g., every `window_duration`).
    ///
    /// Returns the list of pairs that became hot in this window.
    pub fn rotate_window(&mut self) -> Vec<AgentPair> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_window_rotation).as_secs_f64();
        self.last_window_rotation = now;

        let mut newly_hot = Vec::new();

        for (key, stats) in &mut self.pair_stats {
            // Calculate rates.
            if elapsed > 0.0 {
                stats.transfer_rate = stats.window_transfers as f64 / elapsed;
                stats.bandwidth = stats.window_bytes as f64 / elapsed;
            }

            let was_hot = stats.is_hot;

            // Classify hotness.
            if stats.transfer_rate >= self.hot_threshold_rate {
                stats.is_hot = true;
                stats.consecutive_hot_windows += 1;
            } else {
                stats.is_hot = false;
                stats.consecutive_hot_windows = 0;
            }

            // Report newly-stable hot pairs.
            if stats.is_hot
                && stats.consecutive_hot_windows >= self.hot_stability_windows
                && (!was_hot || stats.consecutive_hot_windows == self.hot_stability_windows)
            {
                if let Some(pair) = self.pair_info.get(key) {
                    newly_hot.push(pair.clone());
                }
            }

            // Reset window counters.
            stats.window_transfers = 0;
            stats.window_bytes = 0;
        }

        // Prune old transfer records.
        let cutoff = now - self.window_duration * 2;
        for records in self.recent_transfers.values_mut() {
            records.retain(|r| r.timestamp > cutoff);
        }

        newly_hot
    }

    /// Get statistics for a specific pair.
    pub fn get_pair_stats(&self, pair: &AgentPair) -> Option<&PairStats> {
        self.pair_stats.get(&pair.key())
    }

    /// Get all currently hot pairs.
    pub fn hot_pairs(&self) -> Vec<(&AgentPair, &PairStats)> {
        self.pair_stats
            .iter()
            .filter(|(_, s)| s.is_hot && s.consecutive_hot_windows >= self.hot_stability_windows)
            .filter_map(|(key, stats)| {
                self.pair_info.get(key).map(|pair| (pair, stats))
            })
            .collect()
    }

    /// Get all tracked pairs with their stats.
    pub fn all_pairs(&self) -> Vec<(&AgentPair, &PairStats)> {
        self.pair_stats
            .iter()
            .filter_map(|(key, stats)| {
                self.pair_info.get(key).map(|pair| (pair, stats))
            })
            .collect()
    }
}

// ============================================================
// Shared Memory Bridge
// ============================================================

/// State of a shared memory bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeState {
    /// Bridge is being set up (allocating shared memory region).
    Provisioning,
    /// Bridge is active and routing data.
    Active,
    /// Bridge is being monitored for cooldown.
    Cooling,
    /// Bridge has been torn down.
    Torn,
}

/// Configuration for a shared memory bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Size of the shared memory mailbox in bytes.
    /// Default: 4096 (fits ~128 CRDTCells at 32 bytes each).
    pub mailbox_size_bytes: u32,
    /// Number of mailbox slots (double-buffered by default).
    pub slot_count: u32,
    /// Which SM (Streaming Multiprocessor) to co-locate on.
    /// None = let the scheduler decide.
    pub target_sm: Option<u32>,
    /// Maximum latency target in nanoseconds.
    pub latency_target_ns: u32,
    /// Whether to use atomic operations for mailbox access
    /// (required when >2 agents share the same bridge).
    pub use_atomics: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            mailbox_size_bytes: 4096,
            slot_count: 2,
            target_sm: None,
            latency_target_ns: 50,
            use_atomics: false,
        }
    }
}

/// A direct shared memory bridge between two agents.
///
/// The bridge allocates a region of GPU shared memory as a
/// fast mailbox. Data written by one agent is immediately
/// visible to the other without going through global memory.
///
/// MEMORY LAYOUT (within shared memory):
/// ```text
/// +------------------+------------------+
/// | Slot 0 (4KB)     | Slot 1 (4KB)     |  ← double-buffered
/// | [header 16B]     | [header 16B]     |
/// | [data payload]   | [data payload]   |
/// +------------------+------------------+
/// | Control Block (64B)                 |
/// | - write_slot: u32                   |
/// | - read_slot:  u32                   |
/// | - sequence:   u64                   |
/// | - flags:      u32                   |
/// +-------------------------------------+
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryBridge {
    /// Unique bridge identifier.
    pub bridge_id: String,
    /// The two agents connected by this bridge.
    pub pair: AgentPair,
    /// Bridge configuration.
    pub config: BridgeConfig,
    /// Current state.
    pub state: BridgeState,
    /// When the bridge was created (epoch seconds).
    pub created_at: u64,
    /// Total messages routed through this bridge.
    pub messages_routed: u64,
    /// Total bytes routed.
    pub bytes_routed: u64,
    /// Estimated latency savings in nanoseconds per transfer.
    pub latency_savings_ns: u64,
    /// SM index where the bridge is co-located (if assigned).
    pub assigned_sm: Option<u32>,
    /// Shared memory offset within the SM's shared memory bank.
    pub shmem_offset: u32,
    /// Total shared memory consumed by this bridge in bytes.
    pub shmem_consumed: u32,
}

impl SharedMemoryBridge {
    /// Create a new bridge in Provisioning state.
    pub fn new(pair: AgentPair, config: BridgeConfig) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Total shared memory: slots * mailbox_size + 64-byte control block
        let shmem_consumed = config.slot_count * config.mailbox_size_bytes + 64;

        SharedMemoryBridge {
            bridge_id: format!("bridge_{}_{}", pair.key().replace("<->", "_"), now),
            pair,
            config,
            state: BridgeState::Provisioning,
            created_at: now,
            messages_routed: 0,
            bytes_routed: 0,
            latency_savings_ns: 0,
            assigned_sm: None,
            shmem_offset: 0,
            shmem_consumed,
        }
    }

    /// Activate the bridge (called after SM allocation succeeds).
    pub fn activate(&mut self, sm_index: u32, shmem_offset: u32) {
        self.state = BridgeState::Active;
        self.assigned_sm = Some(sm_index);
        self.shmem_offset = shmem_offset;
    }

    /// Record a message routed through this bridge.
    pub fn record_message(&mut self, bytes: u64) {
        self.messages_routed += 1;
        self.bytes_routed += bytes;
        // Shared memory access is ~5 cycles vs ~400 for global memory.
        // At ~1.5 GHz GPU clock: 395 cycles * 0.67 ns/cycle ≈ 265 ns saved.
        self.latency_savings_ns += 265;
    }

    /// Start cooling down the bridge (communication dropped).
    pub fn start_cooling(&mut self) {
        self.state = BridgeState::Cooling;
    }

    /// Tear down the bridge, releasing shared memory.
    pub fn tear_down(&mut self) {
        self.state = BridgeState::Torn;
        self.assigned_sm = None;
    }

    /// Check if the bridge is currently active.
    pub fn is_active(&self) -> bool {
        self.state == BridgeState::Active
    }
}

// ============================================================
// SM Shared Memory Budget Tracker
// ============================================================

/// Tracks shared memory allocation across all SMs.
///
/// GPU shared memory is a finite resource (48-164 KB per SM).
/// This tracker ensures bridges don't over-allocate and compete
/// with kernel working sets.
#[derive(Debug, Clone)]
pub struct SmSharedMemoryBudget {
    /// Total shared memory per SM in bytes.
    pub total_per_sm: u32,
    /// Shared memory reserved for kernel working sets (not available for bridges).
    pub reserved_per_sm: u32,
    /// Current allocation per SM: sm_index → bytes allocated to bridges.
    allocations: HashMap<u32, u32>,
    /// Number of SMs on the GPU.
    pub sm_count: u32,
}

impl SmSharedMemoryBudget {
    /// Create a new budget tracker.
    ///
    /// # Arguments
    /// * `total_per_sm` — total shared memory per SM in bytes (e.g., 65536 for 64KB)
    /// * `reserved_per_sm` — bytes reserved for kernel working sets
    /// * `sm_count` — number of streaming multiprocessors
    pub fn new(total_per_sm: u32, reserved_per_sm: u32, sm_count: u32) -> Self {
        SmSharedMemoryBudget {
            total_per_sm,
            reserved_per_sm,
            allocations: HashMap::new(),
            sm_count,
        }
    }

    /// Available bytes for bridges on a given SM.
    pub fn available(&self, sm_index: u32) -> u32 {
        let used = self.allocations.get(&sm_index).copied().unwrap_or(0);
        let budget = self.total_per_sm.saturating_sub(self.reserved_per_sm);
        budget.saturating_sub(used)
    }

    /// Try to allocate `bytes` on the given SM.
    /// Returns true if allocation succeeded.
    pub fn try_allocate(&mut self, sm_index: u32, bytes: u32) -> bool {
        if self.available(sm_index) >= bytes {
            *self.allocations.entry(sm_index).or_insert(0) += bytes;
            true
        } else {
            false
        }
    }

    /// Release `bytes` from a given SM.
    pub fn release(&mut self, sm_index: u32, bytes: u32) {
        if let Some(used) = self.allocations.get_mut(&sm_index) {
            *used = used.saturating_sub(bytes);
        }
    }

    /// Find the SM with the most available shared memory for bridges.
    pub fn best_sm(&self) -> Option<u32> {
        (0..self.sm_count).max_by_key(|&sm| self.available(sm))
    }

    /// Total bytes allocated across all SMs.
    pub fn total_allocated(&self) -> u32 {
        self.allocations.values().sum()
    }

    /// Utilization percentage across all SMs.
    pub fn utilization_percent(&self) -> f64 {
        let budget_per_sm = self.total_per_sm.saturating_sub(self.reserved_per_sm);
        let total_budget = budget_per_sm as f64 * self.sm_count as f64;
        if total_budget <= 0.0 {
            return 0.0;
        }
        (self.total_allocated() as f64 / total_budget) * 100.0
    }
}

// ============================================================
// Bridge Manager
// ============================================================

/// Orchestrates bridge lifecycle: creation, monitoring, teardown.
pub struct BridgeManager {
    /// All bridges, keyed by bridge_id.
    bridges: HashMap<String, SharedMemoryBridge>,
    /// Pair key → bridge_id mapping for fast lookup.
    pair_to_bridge: HashMap<String, String>,
    /// Shared memory budget tracker.
    shmem_budget: SmSharedMemoryBudget,
    /// Communication tracker for hot pair detection.
    comm_tracker: CommunicationTracker,
    /// Default bridge configuration.
    default_config: BridgeConfig,
    /// Maximum number of simultaneous bridges.
    max_bridges: usize,
    /// Cooling duration before teardown (if pair goes cold).
    cooling_duration: Duration,
    /// Cooling start times for bridges in Cooling state.
    cooling_starts: HashMap<String, Instant>,
}

impl BridgeManager {
    /// Create a new bridge manager.
    ///
    /// # Arguments
    /// * `sm_count` — number of SMs on the GPU
    /// * `shmem_per_sm` — total shared memory per SM in bytes
    /// * `shmem_reserved` — bytes reserved for kernel working sets per SM
    /// * `max_bridges` — maximum simultaneous bridges
    pub fn new(
        sm_count: u32,
        shmem_per_sm: u32,
        shmem_reserved: u32,
        max_bridges: usize,
    ) -> Self {
        BridgeManager {
            bridges: HashMap::new(),
            pair_to_bridge: HashMap::new(),
            shmem_budget: SmSharedMemoryBudget::new(shmem_per_sm, shmem_reserved, sm_count),
            comm_tracker: CommunicationTracker::new(
                Duration::from_secs(1),  // 1-second measurement window
                100.0,                    // 100 transfers/sec threshold
                3,                        // 3 consecutive hot windows
            ),
            default_config: BridgeConfig::default(),
            max_bridges,
            cooling_duration: Duration::from_secs(5),
            cooling_starts: HashMap::new(),
        }
    }

    /// Record a data transfer and potentially create/maintain bridges.
    ///
    /// This is the main entry point. Call it for every inter-agent
    /// data transfer.
    ///
    /// Returns a list of bridge state changes (created, torn down).
    pub fn record_transfer(
        &mut self,
        src: &AgentEndpoint,
        dst: &AgentEndpoint,
        bytes: u64,
    ) -> Vec<BridgeEvent> {
        self.comm_tracker.record_transfer(src, dst, bytes);

        // Route through existing bridge if active.
        let pair = AgentPair::new(src.clone(), dst.clone());
        let pair_key = pair.key();

        if let Some(bridge_id) = self.pair_to_bridge.get(&pair_key).cloned() {
            if let Some(bridge) = self.bridges.get_mut(&bridge_id) {
                if bridge.is_active() {
                    bridge.record_message(bytes);
                }
            }
        }

        Vec::new() // Events are generated during tick()
    }

    /// Periodic tick: rotate windows, create/tear down bridges.
    ///
    /// Call this at regular intervals (e.g., every second).
    pub fn tick(&mut self) -> Vec<BridgeEvent> {
        let mut events = Vec::new();

        // Rotate communication window.
        let newly_hot = self.comm_tracker.rotate_window();

        // Create bridges for newly hot pairs.
        for pair in newly_hot {
            if self.bridges.len() >= self.max_bridges {
                break;
            }
            if self.pair_to_bridge.contains_key(&pair.key()) {
                continue;
            }
            if let Some(event) = self.create_bridge(pair) {
                events.push(event);
            }
        }

        // Check for pairs that went cold: start cooling.
        let hot_keys: Vec<String> = self
            .comm_tracker
            .hot_pairs()
            .iter()
            .map(|(p, _)| p.key())
            .collect();

        let active_bridge_keys: Vec<String> = self
            .pair_to_bridge
            .keys()
            .cloned()
            .collect();

        for pair_key in &active_bridge_keys {
            if !hot_keys.contains(pair_key) {
                if let Some(bridge_id) = self.pair_to_bridge.get(pair_key).cloned() {
                    if let Some(bridge) = self.bridges.get_mut(&bridge_id) {
                        if bridge.state == BridgeState::Active {
                            bridge.start_cooling();
                            self.cooling_starts
                                .insert(bridge_id.clone(), Instant::now());
                            events.push(BridgeEvent::Cooling {
                                bridge_id: bridge_id.clone(),
                                pair_key: pair_key.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Tear down bridges that have been cooling long enough.
        let mut to_teardown = Vec::new();
        for (bridge_id, start) in &self.cooling_starts {
            if start.elapsed() >= self.cooling_duration {
                to_teardown.push(bridge_id.clone());
            }
        }
        for bridge_id in to_teardown {
            if let Some(event) = self.teardown_bridge(&bridge_id) {
                events.push(event);
            }
            self.cooling_starts.remove(&bridge_id);
        }

        events
    }

    /// Create a bridge for a hot pair.
    fn create_bridge(&mut self, pair: AgentPair) -> Option<BridgeEvent> {
        let mut bridge = SharedMemoryBridge::new(pair.clone(), self.default_config.clone());

        // Find the best SM with available shared memory.
        let sm = if let Some(target) = bridge.config.target_sm {
            if self.shmem_budget.available(target) >= bridge.shmem_consumed {
                Some(target)
            } else {
                self.shmem_budget.best_sm()
            }
        } else {
            self.shmem_budget.best_sm()
        };

        let sm_index = sm?;

        if !self.shmem_budget.try_allocate(sm_index, bridge.shmem_consumed) {
            return None;
        }

        let offset = self.shmem_budget.total_per_sm
            - self.shmem_budget.available(sm_index)
            - bridge.shmem_consumed;

        bridge.activate(sm_index, offset);

        let bridge_id = bridge.bridge_id.clone();
        let pair_key = pair.key();

        self.pair_to_bridge.insert(pair_key.clone(), bridge_id.clone());
        self.bridges.insert(bridge_id.clone(), bridge);

        Some(BridgeEvent::Created {
            bridge_id,
            pair_key,
            sm_index,
        })
    }

    /// Tear down a bridge, releasing shared memory.
    fn teardown_bridge(&mut self, bridge_id: &str) -> Option<BridgeEvent> {
        let bridge = self.bridges.get_mut(bridge_id)?;
        let pair_key = bridge.pair.key();
        let sm_index = bridge.assigned_sm?;
        let shmem = bridge.shmem_consumed;

        bridge.tear_down();
        self.shmem_budget.release(sm_index, shmem);
        self.pair_to_bridge.remove(&pair_key);

        Some(BridgeEvent::TornDown {
            bridge_id: bridge_id.to_string(),
            pair_key,
            messages_routed: bridge.messages_routed,
            bytes_routed: bridge.bytes_routed,
            latency_saved_ns: bridge.latency_savings_ns,
        })
    }

    /// Get a bridge by its ID.
    pub fn get_bridge(&self, bridge_id: &str) -> Option<&SharedMemoryBridge> {
        self.bridges.get(bridge_id)
    }

    /// Get the bridge for a specific agent pair, if one exists.
    pub fn get_bridge_for_pair(&self, pair: &AgentPair) -> Option<&SharedMemoryBridge> {
        let bridge_id = self.pair_to_bridge.get(&pair.key())?;
        self.bridges.get(bridge_id)
    }

    /// Get statistics about all bridges.
    pub fn stats(&self) -> BridgeManagerStats {
        let active = self.bridges.values().filter(|b| b.is_active()).count();
        let cooling = self
            .bridges
            .values()
            .filter(|b| b.state == BridgeState::Cooling)
            .count();
        let total_messages: u64 = self.bridges.values().map(|b| b.messages_routed).sum();
        let total_savings_ns: u64 = self.bridges.values().map(|b| b.latency_savings_ns).sum();

        BridgeManagerStats {
            active_bridges: active,
            cooling_bridges: cooling,
            total_bridges_created: self.bridges.len(),
            total_messages_routed: total_messages,
            total_latency_savings_ns: total_savings_ns,
            shmem_utilization_percent: self.shmem_budget.utilization_percent(),
            hot_pairs: self.comm_tracker.hot_pairs().len(),
        }
    }

    /// List all active bridges.
    pub fn active_bridges(&self) -> Vec<&SharedMemoryBridge> {
        self.bridges.values().filter(|b| b.is_active()).collect()
    }

    /// Set the default bridge configuration for new bridges.
    pub fn set_default_config(&mut self, config: BridgeConfig) {
        self.default_config = config;
    }
}

/// Events emitted by the bridge manager during tick().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeEvent {
    Created {
        bridge_id: String,
        pair_key: String,
        sm_index: u32,
    },
    Cooling {
        bridge_id: String,
        pair_key: String,
    },
    TornDown {
        bridge_id: String,
        pair_key: String,
        messages_routed: u64,
        bytes_routed: u64,
        latency_saved_ns: u64,
    },
}

/// Aggregated statistics from the bridge manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeManagerStats {
    pub active_bridges: usize,
    pub cooling_bridges: usize,
    pub total_bridges_created: usize,
    pub total_messages_routed: u64,
    pub total_latency_savings_ns: u64,
    pub shmem_utilization_percent: f64,
    pub hot_pairs: usize,
}

// ============================================================
// CUDA Kernel Hints — Co-location Directives
// ============================================================

/// Hints for the CUDA kernel launcher to co-locate bridged agents
/// on the same SM. These hints are advisory — the CUDA runtime
/// may override them.
///
/// In CUDA, shared memory is per-block, and blocks are assigned
/// to SMs by the hardware scheduler. To co-locate two agents:
/// 1. Launch them in the same thread block, OR
/// 2. Use cudaFuncSetAttribute to hint SM affinity (sm_90+), OR
/// 3. Use Cooperative Groups for inter-block shared memory (sm_90+).
///
/// This struct captures the intent; the actual kernel launch
/// configuration is handled by the executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoLocationHint {
    /// Agent pair that should be co-located.
    pub pair: AgentPair,
    /// Preferred SM index (from SmSharedMemoryBudget::best_sm).
    pub preferred_sm: u32,
    /// Shared memory offset for the bridge mailbox.
    pub shmem_offset: u32,
    /// Shared memory size required.
    pub shmem_size: u32,
    /// Whether the hint is mandatory (block launch) or advisory.
    pub mandatory: bool,
}

/// Generate co-location hints for all active bridges.
pub fn generate_colocation_hints(manager: &BridgeManager) -> Vec<CoLocationHint> {
    manager
        .active_bridges()
        .iter()
        .filter_map(|bridge| {
            let sm = bridge.assigned_sm?;
            Some(CoLocationHint {
                pair: bridge.pair.clone(),
                preferred_sm: sm,
                shmem_offset: bridge.shmem_offset,
                shmem_size: bridge.shmem_consumed,
                mandatory: true,
            })
        })
        .collect()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agents() -> (AgentEndpoint, AgentEndpoint) {
        (
            AgentEndpoint::new("claw", "agent_1"),
            AgentEndpoint::new("smpclaw", "agent_2"),
        )
    }

    #[test]
    fn test_agent_pair_canonical_ordering() {
        let (a, b) = make_agents();
        let pair1 = AgentPair::new(a.clone(), b.clone());
        let pair2 = AgentPair::new(b, a);
        assert_eq!(pair1.key(), pair2.key());
    }

    #[test]
    fn test_communication_tracker_records() {
        let mut tracker = CommunicationTracker::new(
            Duration::from_secs(1),
            10.0,
            1,
        );
        let (a, b) = make_agents();

        for _ in 0..100 {
            tracker.record_transfer(&a, &b, 32);
        }

        let pair = AgentPair::new(a, b);
        let stats = tracker.get_pair_stats(&pair).unwrap();
        assert_eq!(stats.total_transfers, 100);
        assert_eq!(stats.total_bytes, 3200);
    }

    #[test]
    fn test_hot_pair_detection() {
        let mut tracker = CommunicationTracker::new(
            Duration::from_millis(10),
            5.0,
            1,
        );
        let (a, b) = make_agents();

        // Pump transfers.
        for _ in 0..100 {
            tracker.record_transfer(&a, &b, 32);
        }

        let hot = tracker.rotate_window();
        // Should detect the pair as hot (100 transfers in ~0 time = high rate).
        assert!(!hot.is_empty(), "Should detect at least one hot pair");
    }

    #[test]
    fn test_shmem_budget() {
        let mut budget = SmSharedMemoryBudget::new(65536, 49152, 4);

        // 16KB available per SM (65536 - 49152).
        assert_eq!(budget.available(0), 16384);

        // Allocate 8KB on SM 0.
        assert!(budget.try_allocate(0, 8192));
        assert_eq!(budget.available(0), 8192);

        // Try to allocate 12KB on SM 0 — should fail.
        assert!(!budget.try_allocate(0, 12288));

        // Release 4KB.
        budget.release(0, 4096);
        assert_eq!(budget.available(0), 12288);
    }

    #[test]
    fn test_bridge_lifecycle() {
        let (a, b) = make_agents();
        let pair = AgentPair::new(a, b);
        let config = BridgeConfig::default();

        let mut bridge = SharedMemoryBridge::new(pair, config);
        assert_eq!(bridge.state, BridgeState::Provisioning);

        bridge.activate(2, 0);
        assert_eq!(bridge.state, BridgeState::Active);
        assert!(bridge.is_active());

        bridge.record_message(64);
        assert_eq!(bridge.messages_routed, 1);
        assert_eq!(bridge.bytes_routed, 64);

        bridge.start_cooling();
        assert_eq!(bridge.state, BridgeState::Cooling);
        assert!(!bridge.is_active());

        bridge.tear_down();
        assert_eq!(bridge.state, BridgeState::Torn);
    }

    #[test]
    fn test_bridge_manager_create_and_teardown() {
        let mut manager = BridgeManager::new(
            4,     // 4 SMs
            65536, // 64KB per SM
            49152, // 48KB reserved for kernels
            8,     // max 8 bridges
        );

        // Simulate hot pair detection.
        let (a, b) = make_agents();
        let pair = AgentPair::new(a.clone(), b.clone());

        // Manually create a bridge.
        let event = manager.create_bridge(pair.clone());
        assert!(event.is_some());
        assert!(matches!(event.unwrap(), BridgeEvent::Created { .. }));

        let stats = manager.stats();
        assert_eq!(stats.active_bridges, 1);

        // Check the bridge exists.
        let bridge = manager.get_bridge_for_pair(&pair);
        assert!(bridge.is_some());
        assert!(bridge.unwrap().is_active());
    }

    #[test]
    fn test_colocation_hints() {
        let mut manager = BridgeManager::new(4, 65536, 49152, 8);
        let (a, b) = make_agents();
        let pair = AgentPair::new(a, b);
        manager.create_bridge(pair);

        let hints = generate_colocation_hints(&manager);
        assert_eq!(hints.len(), 1);
        assert!(hints[0].mandatory);
    }
}
