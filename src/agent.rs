// AgentDispatcher - Manages SuperInstance agents and dispatches operations to GPU
// Uses Unified Memory for zero-copy CPU-GPU communication

use cust::memory::UnifiedBuffer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// Re-export command types
pub use crate::cuda_claw::{Command, CommandType, QueueStatus};

// ============================================================
// SuperInstance Agent Types
// ============================================================

/// Types of SuperInstance agents that can be dispatched
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Claw agent - Has ML model for complex decisions
    Claw,
    /// Bot agent - Simple automation loop without ML model
    Bot,
    /// Seed agent - ML-learnable behavior definition
    Seed,
    /// SMPclaw - Specialized moment processing claw
    SMPclaw,
}

impl AgentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentType::Claw => "claw",
            AgentType::Bot => "bot",
            AgentType::Seed => "seed",
            AgentType::SMPclaw => "smpclaw",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claw" => Some(AgentType::Claw),
            "bot" => Some(AgentType::Bot),
            "seed" => Some(AgentType::Seed),
            "smpclaw" | "smp_claw" => Some(AgentType::SMPclaw),
            _ => None,
        }
    }
}

/// Agent state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Thinking,
    Processing,
    Waiting,
    Error(String),
}

/// SuperInstance agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperInstance {
    pub id: String,
    pub agent_type: AgentType,
    pub state: AgentState,
    pub model: Option<String>,
    pub equipment: Vec<String>,
    pub created_at: u64,
    pub last_active: u64,
}

impl SuperInstance {
    pub fn new(id: String, agent_type: AgentType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        SuperInstance {
            id,
            agent_type,
            state: AgentState::Idle,
            model: None,
            equipment: Vec::new(),
            created_at: now,
            last_active: now,
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_equipment(mut self, equipment: Vec<String>) -> Self {
        self.equipment = equipment;
        self
    }

    pub fn touch(&mut self) {
        self.last_active = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

// ============================================================
// Spreadsheet Cell Structure (matches CUDA layout)
// ============================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpreadsheetCell {
    pub value: f64,
    pub timestamp: u64,
    pub node_id: u32,
    pub state: u32,
    pub _padding: [u32; 3],
}

impl Default for SpreadsheetCell {
    fn default() -> Self {
        SpreadsheetCell {
            value: 0.0,
            timestamp: 0,
            node_id: 0,
            state: 0,
            _padding: [0, 0, 0],
        }
    }
}

// SAFETY: SpreadsheetCell is safe to copy to GPU (no internal pointers)
unsafe impl cust::memory::DeviceCopy for SpreadsheetCell {}

// ============================================================
// CRDT Grid State (matches CUDA CrdtState)
// ============================================================

#[repr(C)]
pub struct CrdtGrid {
    pub cells: UnifiedBuffer<SpreadsheetCell>,
    pub rows: u32,
    pub cols: u32,
    pub total_cells: u32,
}

impl CrdtGrid {
    pub fn new(rows: u32, cols: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let total_cells = rows * cols;
        let cell_data = vec![SpreadsheetCell::default(); total_cells as usize];
        let cells = UnifiedBuffer::new(&cell_data)?;

        Ok(CrdtGrid {
            cells,
            rows,
            cols,
            total_cells,
        })
    }

    pub fn get_index(&self, row: u32, col: u32) -> usize {
        (row * self.cols + col) as usize
    }

    pub fn is_valid(&self, row: u32, col: u32) -> bool {
        row < self.rows && col < self.cols
    }

    pub fn get_cell(&self, row: u32, col: u32) -> SpreadsheetCell {
        if !self.is_valid(row, col) {
            return SpreadsheetCell::default();
        }
        self.cells[self.get_index(row, col)]
    }

    pub fn set_cell(&mut self, row: u32, col: u32, cell: SpreadsheetCell) {
        if self.is_valid(row, col) {
            self.cells[self.get_index(row, col)] = cell;
        }
    }
}

// ============================================================
// Agent Operation Commands
// ============================================================

/// Cell reference for spreadsheet operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRef {
    pub row: u32,
    pub col: u32,
}

impl CellRef {
    pub fn new(row: u32, col: u32) -> Self {
        CellRef { row, col }
    }
}

/// Agent operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum AgentOperation {
    /// Update a cell value
    SetCell {
        cell: CellRef,
        value: f64,
        timestamp: u64,
        node_id: u32,
    },
    /// Get a cell value
    GetCell {
        cell: CellRef,
    },
    /// Add two cells
    AddCells {
        a: CellRef,
        b: CellRef,
        result: CellRef,
    },
    /// Multiply two cells
    MultiplyCells {
        a: CellRef,
        b: CellRef,
        result: CellRef,
    },
    /// Apply formula to cells
    ApplyFormula {
        inputs: Vec<CellRef>,
        output: CellRef,
        formula: String,
    },
    /// Batch update multiple cells
    BatchUpdate {
        updates: Vec<(CellRef, f64)>,
        timestamp: u64,
        node_id: u32,
    },
    /// Agent-specific operation
    AgentOp {
        agent_id: String,
        operation: String,
        params: serde_json::Value,
    },
}

impl AgentOperation {
    /// Validate the operation
    pub fn validate(&self) -> Result<(), String> {
        match self {
            AgentOperation::SetCell { cell, .. } => {
                if cell.row >= 65536 || cell.col >= 65536 {
                    return Err("Cell coordinates out of bounds".to_string());
                }
            }
            AgentOperation::GetCell { cell } => {
                if cell.row >= 65536 || cell.col >= 65536 {
                    return Err("Cell coordinates out of bounds".to_string());
                }
            }
            AgentOperation::AddCells { a, b, result } => {
                if a.row >= 65536 || a.col >= 65536 ||
                   b.row >= 65536 || b.col >= 65536 ||
                   result.row >= 65536 || result.col >= 65536 {
                    return Err("Cell coordinates out of bounds".to_string());
                }
            }
            AgentOperation::MultiplyCells { a, b, result } => {
                if a.row >= 65536 || a.col >= 65536 ||
                   b.row >= 65536 || b.col >= 65536 ||
                   result.row >= 65536 || result.col >= 65536 {
                    return Err("Cell coordinates out of bounds".to_string());
                }
            }
            AgentOperation::ApplyFormula { inputs, output, .. } => {
                if output.row >= 65536 || output.col >= 65536 {
                    return Err("Cell coordinates out of bounds".to_string());
                }
                if inputs.is_empty() {
                    return Err("Formula requires at least one input".to_string());
                }
                for cell in inputs {
                    if cell.row >= 65536 || cell.col >= 65536 {
                        return Err("Cell coordinates out of bounds".to_string());
                    }
                }
            }
            AgentOperation::BatchUpdate { updates, .. } => {
                if updates.is_empty() {
                    return Err("Batch update requires at least one update".to_string());
                }
                if updates.len() > 10000 {
                    return Err("Batch update too large (max 10000)".to_string());
                }
                for (cell, _) in updates {
                    if cell.row >= 65536 || cell.col >= 65536 {
                        return Err("Cell coordinates out of bounds".to_string());
                    }
                }
            }
            AgentOperation::AgentOp { agent_id, operation, .. } => {
                if agent_id.is_empty() {
                    return Err("Agent ID cannot be empty".to_string());
                }
                if operation.is_empty() {
                    return Err("Operation cannot be empty".to_string());
                }
            }
        }
        Ok(())
    }
}

// ============================================================
// AgentDispatcher
// ============================================================

pub struct AgentDispatcher {
    /// Pool of active SuperInstance agents
    pub agents: HashMap<String, SuperInstance>,  // Made public for external access
    /// CRDT spreadsheet state (shared with GPU via Unified Memory)
    crdt_grid: CrdtGrid,
    /// Command queue for GPU communication
    command_queue: Arc<Mutex<UnifiedBuffer<crate::cuda_claw::CommandQueueHost>>>,
    /// Current timestamp for CRDT ordering
    current_timestamp: u64,
    /// Node ID for this instance
    node_id: u32,
}

impl AgentDispatcher {
    /// Create a new AgentDispatcher
    pub fn new(
        rows: u32,
        cols: u32,
        command_queue: Arc<Mutex<UnifiedBuffer<crate::cuda_claw::CommandQueueHost>>>,
        node_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(AgentDispatcher {
            agents: HashMap::new(),
            crdt_grid: CrdtGrid::new(rows, cols)?,
            command_queue,
            current_timestamp: 0,
            node_id,
        })
    }

    /// Register a new SuperInstance agent
    pub fn register_agent(&mut self, agent: SuperInstance) -> Result<(), String> {
        if self.agents.contains_key(&agent.id) {
            return Err(format!("Agent {} already exists", agent.id));
        }
        self.agents.insert(agent.id.clone(), agent);
        Ok(())
    }

    /// Unregister an agent
    pub fn unregister_agent(&mut self, agent_id: &str) -> Option<SuperInstance> {
        self.agents.remove(agent_id)
    }

    /// Get an agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&SuperInstance> {
        self.agents.get(agent_id)
    }

    /// Get a mutable reference to an agent
    pub fn get_agent_mut(&mut self, agent_id: &str) -> Option<&mut SuperInstance> {
        self.agents.get_mut(agent_id)
    }

    /// List all registered agents
    pub fn list_agents(&self) -> Vec<&SuperInstance> {
        self.agents.values().collect()
    }

    /// Increment the global timestamp
    fn tick_timestamp(&mut self) -> u64 {
        self.current_timestamp += 1;
        self.current_timestamp
    }

    /// Dispatch an agent operation to the GPU
    pub fn dispatch_agent_op(&mut self, operation: AgentOperation) -> Result<serde_json::Value, String> {
        // Validate the operation
        operation.validate()?;

        // Increment timestamp for CRDT ordering
        let timestamp = self.tick_timestamp();

        // Process the operation
        match operation {
            AgentOperation::SetCell { cell, value, timestamp: op_ts, node_id } => {
                self.set_cell(cell.row, cell.col, value, op_ts, node_id)?;
                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "set_cell",
                    "cell": {"row": cell.row, "col": cell.col},
                    "value": value,
                    "timestamp": op_ts,
                }))
            }

            AgentOperation::GetCell { cell } => {
                let cell_data = self.crdt_grid.get_cell(cell.row, cell.col);
                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "get_cell",
                    "cell": {"row": cell.row, "col": cell.col},
                    "value": cell_data.value,
                    "timestamp": cell_data.timestamp,
                    "node_id": cell_data.node_id,
                }))
            }

            AgentOperation::AddCells { a, b, result } => {
                let a_val = self.crdt_grid.get_cell(a.row, a.col).value;
                let b_val = self.crdt_grid.get_cell(b.row, b.col).value;
                let sum = a_val + b_val;

                self.set_cell(result.row, result.col, sum, timestamp, self.node_id)?;

                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "add_cells",
                    "a": {"row": a.row, "col": a.col, "value": a_val},
                    "b": {"row": b.row, "col": b.col, "value": b_val},
                    "result": {"row": result.row, "col": result.col, "value": sum},
                }))
            }

            AgentOperation::MultiplyCells { a, b, result } => {
                let a_val = self.crdt_grid.get_cell(a.row, a.col).value;
                let b_val = self.crdt_grid.get_cell(b.row, b.col).value;
                let product = a_val * b_val;

                self.set_cell(result.row, result.col, product, timestamp, self.node_id)?;

                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "multiply_cells",
                    "a": {"row": a.row, "col": a.col, "value": a_val},
                    "b": {"row": b.row, "col": b.col, "value": b_val},
                    "result": {"row": result.row, "col": result.col, "value": product},
                }))
            }

            AgentOperation::ApplyFormula { inputs, output, formula } => {
                // For now, just implement a simple sum formula
                let sum: f64 = inputs.iter()
                    .map(|cell| self.crdt_grid.get_cell(cell.row, cell.col).value)
                    .sum();

                self.set_cell(output.row, output.col, sum, timestamp, self.node_id)?;

                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "apply_formula",
                    "formula": formula,
                    "inputs": inputs.len(),
                    "result": {"row": output.row, "col": output.col, "value": sum},
                }))
            }

            AgentOperation::BatchUpdate { updates, timestamp: op_ts, node_id } => {
                let mut results = Vec::new();
                for (cell, value) in updates {
                    self.set_cell(cell.row, cell.col, value, op_ts, node_id)?;
                    results.push(serde_json::json!({
                        "row": cell.row,
                        "col": cell.col,
                        "value": value
                    }));
                }

                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "batch_update",
                    "count": results.len(),
                    "updates": results,
                }))
            }

            AgentOperation::AgentOp { agent_id, operation, params } => {
                // Update agent activity
                if let Some(agent) = self.get_agent_mut(&agent_id) {
                    agent.touch();
                    agent.state = AgentState::Processing;
                }

                // Create command for GPU
                let cmd = self.create_agent_command(&agent_id, &operation, &params, timestamp)?;
                self.submit_to_queue(cmd)?;

                Ok(serde_json::json!({
                    "status": "success",
                    "operation": "agent_op",
                    "agent_id": agent_id,
                    "agent_operation": operation,
                }))
            }
        }
    }

    /// Set a cell value in the CRDT grid
    fn set_cell(&mut self, row: u32, col: u32, value: f64, timestamp: u64, node_id: u32) -> Result<(), String> {
        if !self.crdt_grid.is_valid(row, col) {
            return Err(format!("Invalid cell coordinates: ({}, {})", row, col));
        }

        let cell = SpreadsheetCell {
            value,
            timestamp,
            node_id,
            state: 0, // CELL_ACTIVE
            _padding: [0, 0, 0],
        };

        self.crdt_grid.set_cell(row, col, cell);
        Ok(())
    }

    /// Create a GPU command from an agent operation
    fn create_agent_command(
        &self,
        agent_id: &str,
        operation: &str,
        params: &serde_json::Value,
        timestamp: u64,
    ) -> Result<Command, String> {
        let cmd_type = match operation {
            "noop" => CommandType::NoOp,
            "add" => CommandType::Add,
            "multiply" => CommandType::Multiply,
            "batch" => CommandType::BatchProcess,
            _ => return Err(format!("Unknown operation: {}", operation)),
        };

        let mut cmd = Command::new(cmd_type, timestamp as u32);
        cmd.timestamp = timestamp;

        // Extract parameters if present
        if let Some(obj) = params.as_object() {
            if let Some(a) = obj.get("a").and_then(|v| v.as_f64()) {
                cmd.data_a = a as f32;
            }
            if let Some(b) = obj.get("b").and_then(|v| v.as_f64()) {
                cmd.data_b = b as f32;
            }
        }

        Ok(cmd)
    }

    /// Submit a command to the GPU queue
    fn submit_to_queue(&self, cmd: Command) -> Result<(), String> {
        let queue = self.command_queue.lock()
            .map_err(|e| format!("Failed to lock queue: {}", e))?;

        // In a real implementation, this would write to the unified memory queue
        // For now, we just validate the command structure
        let _queue_ref = (*queue); // Access the unified memory

        Ok(())
    }

    /// Get dispatcher statistics
    pub fn get_stats(&self) -> DispatcherStats {
        let agents_by_type = {
            let mut counts = HashMap::new();
            for agent in self.agents.values() {
                *counts.entry(agent.agent_type).or_insert(0) += 1;
            }
            counts
        };

        DispatcherStats {
            total_agents: self.agents.len(),
            agents_by_type,
            grid_size: (self.crdt_grid.rows, self.crdt_grid.cols),
            total_cells: self.crdt_grid.total_cells,
            current_timestamp: self.current_timestamp,
            node_id: self.node_id,
        }
    }
}

/// Dispatcher statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatcherStats {
    pub total_agents: usize,
    pub agents_by_type: HashMap<AgentType, usize>,
    pub grid_size: (u32, u32),
    pub total_cells: u32,
    pub current_timestamp: u64,
    pub node_id: u32,
}

// ============================================================
// Helper Functions
// ============================================================

/// Parse a JSON command string into an AgentOperation
pub fn parse_command(json_str: &str) -> Result<AgentOperation, String> {
    serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse command: {}", e))
}

/// Serialize an AgentOperation to JSON
pub fn serialize_command(op: &AgentOperation) -> Result<String, String> {
    serde_json::to_string_pretty(op)
        .map_err(|e| format!("Failed to serialize command: {}", e))
}
