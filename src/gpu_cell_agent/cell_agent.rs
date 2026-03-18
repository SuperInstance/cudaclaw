// ============================================================
// Cell Agent — GPU-Compatible Agent Structure
// ============================================================
//
// Each spreadsheet cell is a CellAgent that runs inside a GPU
// kernel. The agent carries its value, CRDT metadata, constraint
// mask, fiber affinity, and execution history.
//
// GPU COMPATIBILITY:
//   CellAgent is #[repr(C)] so it can be copied to GPU memory
//   directly. For maximum coalescing performance, the host
//   converts the grid to SoA format (CellAgentSoA) before
//   launching kernels.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Cell Agent State
// ============================================================

/// Runtime state of a cell agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum CellAgentState {
    /// Agent is idle — no pending work.
    Idle = 0,
    /// Agent is executing its Muscle Fiber kernel.
    Executing = 1,
    /// Agent is waiting for a dependency (formula input).
    Blocked = 2,
    /// Agent completed its last task successfully.
    Completed = 3,
    /// Agent encountered an error.
    Error = 4,
    /// Agent is being migrated to a different SM.
    Migrating = 5,
}

impl Default for CellAgentState {
    fn default() -> Self {
        CellAgentState::Idle
    }
}

// ============================================================
// Cell Agent (GPU-compatible)
// ============================================================

/// A single cell agent, stored in GPU-compatible layout.
///
/// On the GPU side, this maps to a CUDA struct with identical
/// field ordering and alignment. The host uses `CellAgentSoA`
/// for coalesced transfers; this AoS layout is for per-agent
/// manipulation on the host.
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAgent {
    // ── Cell data ──
    /// Current cell value.
    pub value: f64,
    /// CRDT timestamp (Lamport clock).
    pub timestamp: u64,
    /// Node ID of the last writer.
    pub node_id: u32,
    /// Cell state flags.
    pub cell_state: u32,

    // ── Grid position ──
    /// Row in the spreadsheet grid.
    pub row: u32,
    /// Column in the spreadsheet grid.
    pub col: u32,

    // ── Agent metadata ──
    /// Current agent state.
    pub state: CellAgentState,
    /// Muscle Fiber affinity tag.
    pub fiber_affinity: String,
    /// Constraint mask: bit flags for which constraints apply.
    /// Bit 0 = register_budget, 1 = shmem_ceiling, etc.
    pub constraint_mask: u32,
    /// SM index where this agent is currently located.
    pub sm_index: u32,

    // ── Execution metrics ──
    /// Total number of executions.
    pub execution_count: u64,
    /// Cumulative execution time in microseconds.
    pub total_execution_time_us: f64,
    /// Last execution time in microseconds.
    pub last_execution_time_us: f64,
    /// Number of successful executions.
    pub success_count: u64,
    /// Number of failed executions.
    pub error_count: u64,
}

impl CellAgent {
    /// Create a new idle cell agent.
    pub fn new(row: u32, col: u32) -> Self {
        CellAgent {
            value: 0.0,
            timestamp: 0,
            node_id: 0,
            cell_state: 0,
            row,
            col,
            state: CellAgentState::Idle,
            fiber_affinity: "default".into(),
            constraint_mask: 0x1F, // All 5 default constraints enabled.
            sm_index: 0,
            execution_count: 0,
            total_execution_time_us: 0.0,
            last_execution_time_us: 0.0,
            success_count: 0,
            error_count: 0,
        }
    }

    /// Record an execution result.
    pub fn record_execution(&mut self, time_us: f64, success: bool) {
        self.execution_count += 1;
        self.total_execution_time_us += time_us;
        self.last_execution_time_us = time_us;
        if success {
            self.success_count += 1;
            self.state = CellAgentState::Completed;
        } else {
            self.error_count += 1;
            self.state = CellAgentState::Error;
        }
    }

    /// Average execution time.
    pub fn avg_execution_time_us(&self) -> f64 {
        if self.execution_count == 0 {
            0.0
        } else {
            self.total_execution_time_us / self.execution_count as f64
        }
    }

    /// Success rate (0.0-1.0).
    pub fn success_rate(&self) -> f64 {
        if self.execution_count == 0 {
            1.0
        } else {
            self.success_count as f64 / self.execution_count as f64
        }
    }

    /// Linear index in the grid.
    pub fn linear_index(&self, cols: u32) -> u32 {
        self.row * cols + self.col
    }
}

// ============================================================
// Execution Record
// ============================================================

/// A single execution record for ML feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExecutionRecord {
    /// Row of the agent.
    pub agent_row: u32,
    /// Column of the agent.
    pub agent_col: u32,
    /// Muscle Fiber type used.
    pub fiber_type: String,
    /// Execution time in microseconds.
    pub execution_time_us: f64,
    /// Registers consumed.
    pub registers_used: u32,
    /// Shared memory consumed (bytes).
    pub shared_memory_used: u32,
    /// Whether the execution succeeded.
    pub success: bool,
    /// Epoch timestamp.
    pub timestamp_epoch: u64,
}

// ============================================================
// SoA Layout for GPU Transfer
// ============================================================

/// Structure-of-Arrays layout for coalesced GPU memory access.
///
/// Adjacent threads in a warp access adjacent elements in each
/// array, maximizing VRAM throughput.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAgentSoA {
    /// Number of agents.
    pub count: usize,
    /// Values (f64 per cell).
    pub values: Vec<f64>,
    /// Timestamps (u64 per cell).
    pub timestamps: Vec<u64>,
    /// Node IDs (u32 per cell).
    pub node_ids: Vec<u32>,
    /// Cell states (u32 per cell).
    pub cell_states: Vec<u32>,
    /// Agent states (u32 per cell, cast from CellAgentState).
    pub agent_states: Vec<u32>,
    /// Fiber IDs (u32 per cell, mapped from fiber_affinity string).
    pub fiber_ids: Vec<u32>,
    /// Constraint masks (u32 per cell).
    pub constraint_masks: Vec<u32>,
    /// SM indices (u32 per cell).
    pub sm_indices: Vec<u32>,
    /// Execution counts (u64 per cell).
    pub execution_counts: Vec<u64>,
}

// ============================================================
// Kernel Launch Config
// ============================================================

/// GPU kernel launch configuration derived from the agent grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentKernelConfig {
    /// Grid dimension (number of blocks).
    pub grid_dim: u32,
    /// Block dimension (threads per block).
    pub block_dim: u32,
    /// Shared memory per block (bytes).
    pub shared_memory_bytes: u32,
    /// Registers per thread.
    pub registers_per_thread: u32,
    /// Total agents to process.
    pub total_agents: u32,
}

// ============================================================
// Cell Agent Grid
// ============================================================

/// Host-side manager for a grid of cell agents.
pub struct CellAgentGrid {
    rows: u32,
    cols: u32,
    agents: Vec<CellAgent>,
    /// Execution log for ML feedback.
    execution_log: Vec<AgentExecutionRecord>,
    /// Fiber name -> numeric ID mapping for SoA.
    fiber_id_map: HashMap<String, u32>,
    next_fiber_id: u32,
}

impl CellAgentGrid {
    /// Create a new grid of cell agents.
    pub fn new(rows: u32, cols: u32) -> Self {
        let total = (rows * cols) as usize;
        let mut agents = Vec::with_capacity(total);
        for r in 0..rows {
            for c in 0..cols {
                agents.push(CellAgent::new(r, c));
            }
        }

        let mut fiber_id_map = HashMap::new();
        fiber_id_map.insert("default".into(), 0);

        CellAgentGrid {
            rows,
            cols,
            agents,
            execution_log: Vec::new(),
            fiber_id_map,
            next_fiber_id: 1,
        }
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }

    /// Get total cell count.
    pub fn total_cells(&self) -> u32 {
        self.rows * self.cols
    }

    /// Get an agent by coordinates.
    pub fn get_agent(&self, row: u32, col: u32) -> Option<&CellAgent> {
        if row < self.rows && col < self.cols {
            Some(&self.agents[(row * self.cols + col) as usize])
        } else {
            None
        }
    }

    /// Get a mutable agent by coordinates.
    pub fn get_agent_mut(&mut self, row: u32, col: u32) -> Option<&mut CellAgent> {
        if row < self.rows && col < self.cols {
            Some(&mut self.agents[(row * self.cols + col) as usize])
        } else {
            None
        }
    }

    /// Set a cell's value.
    pub fn set_cell_value(&mut self, row: u32, col: u32, value: f64) {
        if let Some(agent) = self.get_agent_mut(row, col) {
            agent.value = value;
            agent.timestamp += 1;
            agent.state = CellAgentState::Completed;
        }
    }

    /// Assign a Muscle Fiber to a cell.
    pub fn assign_fiber(&mut self, row: u32, col: u32, fiber: String) {
        // Register fiber ID if new.
        if !self.fiber_id_map.contains_key(&fiber) {
            self.fiber_id_map.insert(fiber.clone(), self.next_fiber_id);
            self.next_fiber_id += 1;
        }

        if let Some(agent) = self.get_agent_mut(row, col) {
            agent.fiber_affinity = fiber;
        }
    }

    /// Record an execution for ML feedback.
    pub fn record_execution(&mut self, record: AgentExecutionRecord) {
        let row = record.agent_row;
        let col = record.agent_col;
        let time = record.execution_time_us;
        let success = record.success;

        self.execution_log.push(record);

        if let Some(agent) = self.get_agent_mut(row, col) {
            agent.record_execution(time, success);
        }
    }

    /// Get the execution log for ML feedback.
    pub fn execution_log(&self) -> &[AgentExecutionRecord] {
        &self.execution_log
    }

    /// Convert to SoA layout for GPU transfer.
    pub fn to_soa(&self) -> CellAgentSoA {
        let count = self.agents.len();
        let mut soa = CellAgentSoA {
            count,
            values: Vec::with_capacity(count),
            timestamps: Vec::with_capacity(count),
            node_ids: Vec::with_capacity(count),
            cell_states: Vec::with_capacity(count),
            agent_states: Vec::with_capacity(count),
            fiber_ids: Vec::with_capacity(count),
            constraint_masks: Vec::with_capacity(count),
            sm_indices: Vec::with_capacity(count),
            execution_counts: Vec::with_capacity(count),
        };

        for agent in &self.agents {
            soa.values.push(agent.value);
            soa.timestamps.push(agent.timestamp);
            soa.node_ids.push(agent.node_id);
            soa.cell_states.push(agent.cell_state);
            soa.agent_states.push(agent.state as u32);
            soa.fiber_ids.push(
                *self.fiber_id_map.get(&agent.fiber_affinity).unwrap_or(&0),
            );
            soa.constraint_masks.push(agent.constraint_mask);
            soa.sm_indices.push(agent.sm_index);
            soa.execution_counts.push(agent.execution_count);
        }

        soa
    }

    /// Compute kernel launch config based on the grid and fiber registry.
    pub fn kernel_config(
        &self,
        fiber_registry: &super::muscle_fiber::FiberRegistry,
    ) -> AgentKernelConfig {
        let total = self.total_cells();

        // Find the most common fiber and use its config.
        let mut fiber_counts: HashMap<&str, usize> = HashMap::new();
        for agent in &self.agents {
            *fiber_counts.entry(&agent.fiber_affinity).or_insert(0) += 1;
        }
        let dominant_fiber = fiber_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(f, _)| *f)
            .unwrap_or("default");

        let (block_dim, regs, shmem) = fiber_registry
            .get_fiber_by_name(dominant_fiber)
            .map(|f| (f.block_size, f.registers_per_thread, f.shared_memory_bytes))
            .unwrap_or((256, 32, 4096));

        let grid_dim = (total + block_dim - 1) / block_dim;

        AgentKernelConfig {
            grid_dim,
            block_dim,
            shared_memory_bytes: shmem,
            registers_per_thread: regs,
            total_agents: total,
        }
    }

    /// Get grid statistics.
    pub fn stats(&self) -> CellAgentGridStats {
        let active = self.agents.iter().filter(|a| a.execution_count > 0).count();

        let mut fiber_dist: HashMap<String, usize> = HashMap::new();
        for agent in &self.agents {
            *fiber_dist.entry(agent.fiber_affinity.clone()).or_insert(0) += 1;
        }

        CellAgentGridStats {
            total_cells: self.agents.len(),
            active_cells: active,
            execution_records: self.execution_log.len(),
            fiber_distribution: fiber_dist,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Get all agents (for iteration).
    pub fn all_agents(&self) -> &[CellAgent] {
        &self.agents
    }

    /// Drain the execution log (consumed by ML feedback).
    pub fn drain_execution_log(&mut self) -> Vec<AgentExecutionRecord> {
        std::mem::take(&mut self.execution_log)
    }
}

/// Grid statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAgentGridStats {
    pub total_cells: usize,
    pub active_cells: usize,
    pub execution_records: usize,
    pub fiber_distribution: HashMap<String, usize>,
    pub rows: u32,
    pub cols: u32,
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_agent_creation() {
        let agent = CellAgent::new(5, 10);
        assert_eq!(agent.row, 5);
        assert_eq!(agent.col, 10);
        assert_eq!(agent.state, CellAgentState::Idle);
        assert_eq!(agent.value, 0.0);
    }

    #[test]
    fn test_cell_agent_execution_recording() {
        let mut agent = CellAgent::new(0, 0);
        agent.record_execution(3.5, true);
        agent.record_execution(4.0, true);
        agent.record_execution(10.0, false);

        assert_eq!(agent.execution_count, 3);
        assert_eq!(agent.success_count, 2);
        assert_eq!(agent.error_count, 1);
        assert!((agent.avg_execution_time_us() - 5.833).abs() < 0.01);
        assert!((agent.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_grid_creation() {
        let grid = CellAgentGrid::new(4, 4);
        assert_eq!(grid.total_cells(), 16);
        assert!(grid.get_agent(0, 0).is_some());
        assert!(grid.get_agent(3, 3).is_some());
        assert!(grid.get_agent(4, 0).is_none());
    }

    #[test]
    fn test_set_cell_value() {
        let mut grid = CellAgentGrid::new(2, 2);
        grid.set_cell_value(0, 0, 42.0);
        let agent = grid.get_agent(0, 0).unwrap();
        assert_eq!(agent.value, 42.0);
        assert_eq!(agent.state, CellAgentState::Completed);
    }

    #[test]
    fn test_fiber_assignment() {
        let mut grid = CellAgentGrid::new(2, 2);
        grid.assign_fiber(1, 1, "crdt_merge".into());
        let agent = grid.get_agent(1, 1).unwrap();
        assert_eq!(agent.fiber_affinity, "crdt_merge");
    }

    #[test]
    fn test_soa_conversion() {
        let mut grid = CellAgentGrid::new(2, 2);
        grid.set_cell_value(0, 0, 10.0);
        grid.set_cell_value(1, 1, 20.0);

        let soa = grid.to_soa();
        assert_eq!(soa.count, 4);
        assert_eq!(soa.values[0], 10.0);
        assert_eq!(soa.values[3], 20.0);
    }

    #[test]
    fn test_execution_log() {
        let mut grid = CellAgentGrid::new(2, 2);
        let record = AgentExecutionRecord {
            agent_row: 0,
            agent_col: 0,
            fiber_type: "cell_update".into(),
            execution_time_us: 2.5,
            registers_used: 32,
            shared_memory_used: 256,
            success: true,
            timestamp_epoch: 1000,
        };
        grid.record_execution(record);

        assert_eq!(grid.execution_log().len(), 1);
        let drained = grid.drain_execution_log();
        assert_eq!(drained.len(), 1);
        assert!(grid.execution_log().is_empty());
    }

    #[test]
    fn test_grid_stats() {
        let mut grid = CellAgentGrid::new(3, 3);
        grid.set_cell_value(0, 0, 1.0);
        let record = AgentExecutionRecord {
            agent_row: 0,
            agent_col: 0,
            fiber_type: "default".into(),
            execution_time_us: 1.0,
            registers_used: 16,
            shared_memory_used: 128,
            success: true,
            timestamp_epoch: 1,
        };
        grid.record_execution(record);

        let stats = grid.stats();
        assert_eq!(stats.total_cells, 9);
        assert_eq!(stats.active_cells, 1);
        assert_eq!(stats.execution_records, 1);
    }
}
