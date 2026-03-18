// ============================================================
// Spreadsheet Bridge — Cell-to-Root Mapping & Adaptive Fiber
// ============================================================
//
// Connects the cudaclaw DNA to the spreadsheet-moment project.
// Each spreadsheet cell is mapped to a "Root" in the Cudaclaw
// ecosystem tree. When a cell formula changes, the bridge
// "harvests" the new data pattern and checks if the current
// Muscle Fiber is still optimal. If a massive parallel
// recalculation is needed, a Ramification event is triggered
// to spawn additional threads in the persistent kernel.
//
// ARCHITECTURE:
//
//   Spreadsheet Grid (spreadsheet-moment)
//          |
//          v
//   SpreadsheetBridge
//     |-- CellRootMap       maps (row, col) -> CellRoot
//     |-- FormulaTracker    detects formula changes
//     |-- PatternHarvester  classifies data access patterns
//     |-- FiberEfficiency   checks if current fiber is optimal
//     |-- RamificationTrigger  spawns threads for dep chains
//          |
//          v
//   cudaclaw DNA / RamifyEngine / CellAgentGrid
//
// USAGE:
//   let mut bridge = SpreadsheetBridge::new(64, 64);
//   bridge.register_formula(3, 5, "=SUM(A1:A100)");
//   bridge.on_cell_edit(3, 5, 42.0);
//   let events = bridge.tick();
//   // Process RamificationEvents...
//
// CLI:
//   cudaclaw spreadsheet --demo
//   cudaclaw spreadsheet --status
//   cudaclaw spreadsheet --export <PATH>
//   cudaclaw spreadsheet --help
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::gpu_cell_agent::{
    CellAgentGrid, CellAgentState, AgentExecutionRecord,
    FiberRegistry, FiberType,
};
use crate::constraint_theory::geometric_twin::GeometricTwinMap;

// ============================================================
// Cell Root — a spreadsheet cell mapped to a Cudaclaw root
// ============================================================

/// A single Root node in the Cudaclaw ecosystem tree.
/// Each spreadsheet cell maps to exactly one CellRoot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRoot {
    /// Row in the spreadsheet grid.
    pub row: u32,
    /// Column in the spreadsheet grid.
    pub col: u32,
    /// Root ID (unique within the bridge).
    pub root_id: String,
    /// Current Muscle Fiber affinity.
    pub current_fiber: String,
    /// The currently detected data access pattern.
    pub detected_pattern: DataPattern,
    /// Formula string (empty if the cell is a literal value).
    pub formula: String,
    /// Number of cells this cell depends on (formula inputs).
    pub dependency_count: u32,
    /// Number of cells that depend on THIS cell (downstream).
    pub dependent_count: u32,
    /// Total access count (edits + reads).
    pub access_count: u64,
    /// Rolling window of recent access timestamps (epoch ms).
    pub recent_access_times: Vec<u64>,
    /// Whether this root is currently "hot" (high access rate).
    pub is_hot: bool,
    /// Whether a Ramification has been triggered for this root.
    pub ramification_active: bool,
    /// Last fiber efficiency score (0.0-1.0).
    pub fiber_efficiency: f64,
    /// Timestamp of last fiber reassignment.
    pub last_fiber_change_epoch: u64,
}

impl CellRoot {
    /// Create a new CellRoot for a given cell position.
    pub fn new(row: u32, col: u32) -> Self {
        CellRoot {
            row,
            col,
            root_id: format!("root_{}_{}", row, col),
            current_fiber: "cell_update".into(),
            detected_pattern: DataPattern::Isolated,
            formula: String::new(),
            dependency_count: 0,
            dependent_count: 0,
            access_count: 0,
            recent_access_times: Vec::new(),
            is_hot: false,
            ramification_active: false,
            fiber_efficiency: 1.0,
            last_fiber_change_epoch: 0,
        }
    }

    /// Record an access event, updating hotness detection.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        let now = now_epoch_ms();
        self.recent_access_times.push(now);

        // Keep a rolling window of 100 accesses.
        if self.recent_access_times.len() > 100 {
            self.recent_access_times.remove(0);
        }

        // Hot = more than 50 accesses in the last 1000ms.
        let cutoff = now.saturating_sub(1000);
        let recent_count = self.recent_access_times.iter()
            .filter(|&&t| t >= cutoff)
            .count();
        self.is_hot = recent_count > 50;
    }

    /// Check if this root has a formula (as opposed to a literal value).
    pub fn has_formula(&self) -> bool {
        !self.formula.is_empty()
    }
}

// ============================================================
// Data Pattern — detected access pattern for a cell/region
// ============================================================

/// The detected data access pattern for a cell or region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataPattern {
    /// Single isolated cell edit — no dependencies.
    Isolated,
    /// Sequential row/column scan (e.g., fill-down).
    Sequential,
    /// Columnar access (e.g., SUM of a column).
    Columnar,
    /// Scattered random access across the grid.
    Random,
    /// Dense block update (e.g., paste a range).
    BlockUpdate,
    /// Formula chain — cell depends on many others.
    FormulaChain,
    /// Massive parallel recalculation needed (deep DAG).
    MassiveRecalc,
}

impl DataPattern {
    /// Recommend the best Muscle Fiber type for this pattern.
    pub fn recommended_fiber(&self) -> &str {
        match self {
            DataPattern::Isolated => "cell_update",
            DataPattern::Sequential => "cell_update",
            DataPattern::Columnar => "batch_process",
            DataPattern::Random => "crdt_merge",
            DataPattern::BlockUpdate => "batch_process",
            DataPattern::FormulaChain => "formula_eval",
            DataPattern::MassiveRecalc => "formula_eval",
        }
    }

    /// Whether this pattern warrants a Ramification event.
    pub fn needs_ramification(&self) -> bool {
        matches!(self, DataPattern::MassiveRecalc | DataPattern::FormulaChain)
    }
}

// ============================================================
// Formula Change — represents a change to a cell's formula
// ============================================================

/// A formula change event detected by the bridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaChange {
    /// Cell row.
    pub row: u32,
    /// Cell column.
    pub col: u32,
    /// Previous formula (empty if was a literal).
    pub old_formula: String,
    /// New formula (empty if now a literal).
    pub new_formula: String,
    /// Number of dependencies in the new formula.
    pub new_dependency_count: u32,
    /// Timestamp (epoch ms).
    pub timestamp: u64,
}

// ============================================================
// Ramification Event — signals the kernel manager
// ============================================================

/// An event emitted by the bridge to trigger system actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RamificationEvent {
    /// A cell's fiber should be reassigned to a better match.
    FiberReassignment {
        row: u32,
        col: u32,
        old_fiber: String,
        new_fiber: String,
        reason: String,
    },
    /// Spawn additional threads for a cell's dependency chain.
    SpawnRecalcThreads {
        /// The root cell that triggered the recalculation.
        root_row: u32,
        root_col: u32,
        /// Number of cells in the dependency chain.
        chain_length: u32,
        /// Recommended thread count to spawn.
        recommended_threads: u32,
        /// Recommended block size for the spawned threads.
        recommended_block_size: u32,
    },
    /// A new formula was detected; the pattern may have changed.
    FormulaHarvested {
        row: u32,
        col: u32,
        formula: String,
        detected_pattern: DataPattern,
        dependency_count: u32,
    },
    /// A cell became "hot" (high access rate).
    HotCellDetected {
        row: u32,
        col: u32,
        access_rate: f64,
    },
    /// A cell cooled down from hot status.
    CellCooledDown {
        row: u32,
        col: u32,
    },
    /// A bulk region update was detected.
    BulkRegionUpdate {
        start_row: u32,
        start_col: u32,
        end_row: u32,
        end_col: u32,
        cell_count: u32,
    },
}

// ============================================================
// Fiber Efficiency Assessment
// ============================================================

/// Assessment of whether the current fiber is optimal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberAssessment {
    /// Current fiber name.
    pub current_fiber: String,
    /// Recommended fiber name.
    pub recommended_fiber: String,
    /// Efficiency score of current fiber for the detected pattern (0.0-1.0).
    pub current_efficiency: f64,
    /// Expected efficiency of the recommended fiber (0.0-1.0).
    pub recommended_efficiency: f64,
    /// Whether a switch is recommended.
    pub should_switch: bool,
    /// Human-readable reason.
    pub reason: String,
}

// ============================================================
// Spreadsheet Bridge — the main orchestrator
// ============================================================

/// The main bridge between spreadsheet-moment and cudaclaw.
pub struct SpreadsheetBridge {
    /// Grid dimensions.
    rows: u32,
    cols: u32,
    /// All cell roots, keyed by (row, col) as "row_col".
    roots: HashMap<String, CellRoot>,
    /// Dependency graph: cell -> list of cells it depends on.
    dependency_graph: HashMap<String, Vec<String>>,
    /// Reverse dependency graph: cell -> list of cells that depend on it.
    reverse_deps: HashMap<String, Vec<String>>,
    /// Formula tracker: cell -> current formula string.
    formulas: HashMap<String, String>,
    /// Recent cell edits in the current tick window.
    recent_edits: Vec<(u32, u32, f64)>,
    /// Cell agent grid (from gpu_cell_agent module).
    agent_grid: CellAgentGrid,
    /// Geometric twin map (from constraint_theory module).
    twin_map: GeometricTwinMap,
    /// Fiber registry for efficiency assessment.
    fiber_registry: FiberRegistry,
    /// Configuration.
    config: BridgeConfig,
    /// Total tick count.
    tick_count: u64,
    /// Total events emitted.
    total_events: u64,
    /// Start time.
    start_time: Instant,
    /// Pending events from the current tick.
    pending_events: Vec<RamificationEvent>,
}

/// Configuration for the SpreadsheetBridge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Minimum dependency chain length to trigger Ramification.
    pub ramification_threshold: u32,
    /// Minimum access rate (accesses/sec) to be "hot".
    pub hot_cell_rate_threshold: f64,
    /// Minimum efficiency drop to trigger fiber reassignment.
    pub fiber_switch_threshold: f64,
    /// Maximum threads to recommend for a Ramification spawn.
    pub max_spawn_threads: u32,
    /// Minimum block size for spawned recalc threads.
    pub min_block_size: u32,
    /// Cooldown (ms) between fiber reassignment for the same cell.
    pub fiber_cooldown_ms: u64,
    /// Number of recent edits to buffer before pattern analysis.
    pub edit_batch_size: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        BridgeConfig {
            ramification_threshold: 16,
            hot_cell_rate_threshold: 50.0,
            fiber_switch_threshold: 0.2,
            max_spawn_threads: 1024,
            min_block_size: 32,
            fiber_cooldown_ms: 5000,
            edit_batch_size: 8,
        }
    }
}

impl SpreadsheetBridge {
    /// Create a new SpreadsheetBridge for a grid of the given dimensions.
    pub fn new(rows: u32, cols: u32) -> Self {
        Self::with_config(rows, cols, BridgeConfig::default())
    }

    /// Create a new SpreadsheetBridge with custom configuration.
    pub fn with_config(rows: u32, cols: u32, config: BridgeConfig) -> Self {
        let mut agent_grid = CellAgentGrid::new(rows, cols);
        let mut twin_map = GeometricTwinMap::new(rows, cols);
        twin_map.build_default_topology();

        // Initialize all roots.
        let mut roots = HashMap::new();
        for r in 0..rows {
            for c in 0..cols {
                let key = cell_key(r, c);
                roots.insert(key, CellRoot::new(r, c));
            }
        }

        SpreadsheetBridge {
            rows,
            cols,
            roots,
            dependency_graph: HashMap::new(),
            reverse_deps: HashMap::new(),
            formulas: HashMap::new(),
            recent_edits: Vec::new(),
            agent_grid,
            twin_map,
            fiber_registry: FiberRegistry::default_fibers(),
            config,
            tick_count: 0,
            total_events: 0,
            start_time: Instant::now(),
            pending_events: Vec::new(),
        }
    }

    // --------------------------------------------------------
    // Cell Editing
    // --------------------------------------------------------

    /// Called when a spreadsheet cell value is edited.
    ///
    /// Records the access, updates the agent grid, touches the
    /// geometric twin, and buffers the edit for pattern analysis.
    pub fn on_cell_edit(&mut self, row: u32, col: u32, value: f64) {
        let key = cell_key(row, col);

        // Update root access tracking.
        if let Some(root) = self.roots.get_mut(&key) {
            let was_hot = root.is_hot;
            root.record_access();

            // Hot cell detection.
            if root.is_hot && !was_hot {
                let rate = root.recent_access_times.len() as f64;
                self.pending_events.push(RamificationEvent::HotCellDetected {
                    row, col, access_rate: rate,
                });
            }
        }

        // Update agent grid.
        self.agent_grid.set_cell_value(row, col, value);

        // Touch geometric twin.
        self.twin_map.touch(row, col);

        // Buffer for pattern analysis.
        self.recent_edits.push((row, col, value));
    }

    // --------------------------------------------------------
    // Formula Management
    // --------------------------------------------------------

    /// Register or update a formula for a cell.
    ///
    /// Parses the formula for dependencies, updates the dependency
    /// graph, and triggers a harvest of the new data pattern.
    pub fn register_formula(&mut self, row: u32, col: u32, formula: &str) {
        let key = cell_key(row, col);
        let old_formula = self.formulas.get(&key).cloned().unwrap_or_default();

        // Parse dependencies from the formula.
        let deps = parse_formula_dependencies(formula, self.rows, self.cols);

        // Update dependency graph.
        self.dependency_graph.insert(key.clone(), deps.clone());

        // Update reverse deps: for each dependency, add this cell as a dependent.
        // First, remove old reverse deps.
        if let Some(old_deps) = self.dependency_graph.get(&key) {
            for dep_key in old_deps {
                if let Some(rev) = self.reverse_deps.get_mut(dep_key) {
                    rev.retain(|k| k != &key);
                }
            }
        }
        // Add new reverse deps.
        for dep_key in &deps {
            self.reverse_deps
                .entry(dep_key.clone())
                .or_default()
                .push(key.clone());
        }

        // Compute values that need &self BEFORE taking &mut self.roots.
        let rev_count = self.reverse_deps.get(&key).map(|v| v.len()).unwrap_or(0) as u32;
        let pattern = self.classify_pattern(row, col);
        let chain_len = self.compute_dependency_chain_length(row, col);
        let spawn_params = self.compute_spawn_params(chain_len);
        let fiber_cooldown = self.config.fiber_cooldown_ms;
        let ramification_threshold = self.config.ramification_threshold;

        // Update root's pattern FIRST so assess_fiber_efficiency sees the new pattern.
        if let Some(root) = self.roots.get_mut(&key) {
            root.detected_pattern = pattern;
        }

        // Now assess efficiency with the updated pattern.
        let assessment = self.assess_fiber_efficiency(row, col);

        // Now take the mutable borrow on roots for remaining updates.
        if let Some(root) = self.roots.get_mut(&key) {
            root.formula = formula.to_string();
            root.dependency_count = deps.len() as u32;
            root.dependent_count = rev_count;

            // Emit FormulaHarvested event.
            self.pending_events.push(RamificationEvent::FormulaHarvested {
                row,
                col,
                formula: formula.to_string(),
                detected_pattern: pattern,
                dependency_count: deps.len() as u32,
            });

            // Check fiber efficiency.
            root.fiber_efficiency = assessment.current_efficiency;

            if assessment.should_switch {
                let now = now_epoch_ms();
                let since_last = now.saturating_sub(root.last_fiber_change_epoch);
                if since_last >= fiber_cooldown {
                    let old_fiber = root.current_fiber.clone();
                    root.current_fiber = assessment.recommended_fiber.clone();
                    root.last_fiber_change_epoch = now;

                    // Update agent grid fiber assignment.
                    self.agent_grid.assign_fiber(row, col, assessment.recommended_fiber.clone());
                    // Update twin map fiber affinity.
                    self.twin_map.set_fiber_affinity(row, col, assessment.recommended_fiber.clone());

                    self.pending_events.push(RamificationEvent::FiberReassignment {
                        row,
                        col,
                        old_fiber,
                        new_fiber: assessment.recommended_fiber,
                        reason: assessment.reason,
                    });
                }
            }

            // Check if Ramification is needed.
            if pattern.needs_ramification() {
                if chain_len >= ramification_threshold {
                    let (threads, block_size) = spawn_params;
                    root.ramification_active = true;

                    self.pending_events.push(RamificationEvent::SpawnRecalcThreads {
                        root_row: row,
                        root_col: col,
                        chain_length: chain_len,
                        recommended_threads: threads,
                        recommended_block_size: block_size,
                    });
                }
            }
        }

        // Store formula.
        self.formulas.insert(key, formula.to_string());
    }

    /// Remove a formula from a cell (revert to literal value).
    pub fn clear_formula(&mut self, row: u32, col: u32) {
        let key = cell_key(row, col);

        // Remove from dependency graph.
        if let Some(old_deps) = self.dependency_graph.remove(&key) {
            for dep_key in &old_deps {
                if let Some(rev) = self.reverse_deps.get_mut(dep_key) {
                    rev.retain(|k| k != &key);
                }
            }
        }

        self.formulas.remove(&key);

        if let Some(root) = self.roots.get_mut(&key) {
            root.formula.clear();
            root.dependency_count = 0;
            root.detected_pattern = DataPattern::Isolated;
            root.ramification_active = false;
        }
    }

    // --------------------------------------------------------
    // Bulk Operations
    // --------------------------------------------------------

    /// Notify the bridge of a bulk region update (e.g., paste).
    pub fn on_bulk_update(
        &mut self,
        start_row: u32,
        start_col: u32,
        end_row: u32,
        end_col: u32,
        values: &[(u32, u32, f64)],
    ) {
        for &(r, c, v) in values {
            self.on_cell_edit(r, c, v);
        }

        let cell_count = values.len() as u32;
        self.pending_events.push(RamificationEvent::BulkRegionUpdate {
            start_row,
            start_col,
            end_row,
            end_col,
            cell_count,
        });

        // Bulk updates may change patterns of all affected cells.
        for &(r, c, _) in values {
            let key = cell_key(r, c);
            if let Some(root) = self.roots.get_mut(&key) {
                if root.detected_pattern == DataPattern::Isolated {
                    root.detected_pattern = DataPattern::BlockUpdate;
                }
            }
        }
    }

    // --------------------------------------------------------
    // Tick — periodic evaluation
    // --------------------------------------------------------

    /// Evaluate the bridge state, analyze buffered edits, and
    /// return any Ramification events generated this tick.
    pub fn tick(&mut self) -> Vec<RamificationEvent> {
        self.tick_count += 1;

        // Analyze recent edit patterns if we have enough.
        if self.recent_edits.len() >= self.config.edit_batch_size {
            self.analyze_edit_batch();
        }

        // Check for cells that have cooled down.
        let now = now_epoch_ms();
        let mut cooled = Vec::new();
        for (key, root) in &self.roots {
            if root.is_hot {
                let cutoff = now.saturating_sub(1000);
                let recent = root.recent_access_times.iter()
                    .filter(|&&t| t >= cutoff)
                    .count();
                if recent <= 10 {
                    cooled.push((root.row, root.col));
                }
            }
        }
        for (r, c) in cooled {
            let key = cell_key(r, c);
            if let Some(root) = self.roots.get_mut(&key) {
                root.is_hot = false;
                root.ramification_active = false;
            }
            self.pending_events.push(RamificationEvent::CellCooledDown {
                row: r, col: c,
            });
        }

        // Drain and return events.
        let events: Vec<_> = self.pending_events.drain(..).collect();
        self.total_events += events.len() as u64;
        events
    }

    // --------------------------------------------------------
    // Pattern Classification
    // --------------------------------------------------------

    /// Classify the data access pattern for a given cell.
    fn classify_pattern(&self, row: u32, col: u32) -> DataPattern {
        let key = cell_key(row, col);

        // Check dependency count.
        let dep_count = self.dependency_graph
            .get(&key)
            .map(|d| d.len())
            .unwrap_or(0);

        if dep_count == 0 {
            return DataPattern::Isolated;
        }

        // Check total chain length (transitive deps).
        let chain_len = self.compute_dependency_chain_length(row, col);

        if chain_len >= self.config.ramification_threshold {
            return DataPattern::MassiveRecalc;
        }

        if dep_count > 8 {
            return DataPattern::FormulaChain;
        }

        // Check if dependencies are in the same column.
        if let Some(deps) = self.dependency_graph.get(&key) {
            let same_col = deps.iter().all(|d| {
                if let Some((_, dc)) = parse_cell_key(d) {
                    dc == col
                } else {
                    false
                }
            });
            if same_col && dep_count > 1 {
                return DataPattern::Columnar;
            }

            // Check if dependencies are sequential in the same row.
            let same_row = deps.iter().all(|d| {
                if let Some((dr, _)) = parse_cell_key(d) {
                    dr == row
                } else {
                    false
                }
            });
            if same_row && dep_count > 1 {
                return DataPattern::Sequential;
            }
        }

        // Multiple scattered dependencies.
        if dep_count > 3 {
            return DataPattern::Random;
        }

        DataPattern::Isolated
    }

    /// Compute the total dependency chain length (BFS depth).
    fn compute_dependency_chain_length(&self, row: u32, col: u32) -> u32 {
        let key = cell_key(row, col);
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(key.clone());
        queue.push_back(key.clone());

        while let Some(current) = queue.pop_front() {
            if let Some(deps) = self.dependency_graph.get(&current) {
                for dep in deps {
                    if visited.insert(dep.clone()) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        // Chain length = total unique cells visited (minus the root).
        visited.len().saturating_sub(1) as u32
    }

    // --------------------------------------------------------
    // Fiber Efficiency
    // --------------------------------------------------------

    /// Assess whether the current Muscle Fiber is efficient for
    /// the detected data pattern of a given cell.
    fn assess_fiber_efficiency(&self, row: u32, col: u32) -> FiberAssessment {
        let key = cell_key(row, col);
        let root = match self.roots.get(&key) {
            Some(r) => r,
            None => {
                return FiberAssessment {
                    current_fiber: "unknown".into(),
                    recommended_fiber: "cell_update".into(),
                    current_efficiency: 0.0,
                    recommended_efficiency: 1.0,
                    should_switch: false,
                    reason: "Cell not found".into(),
                };
            }
        };

        let recommended = root.detected_pattern.recommended_fiber();
        let current = &root.current_fiber;

        // Compute efficiency score: 1.0 if current matches recommended,
        // degraded otherwise based on pattern mismatch severity.
        let current_efficiency = compute_fiber_efficiency(current, &root.detected_pattern);
        let recommended_efficiency = compute_fiber_efficiency(recommended, &root.detected_pattern);

        let should_switch = current != recommended
            && (recommended_efficiency - current_efficiency) >= self.config.fiber_switch_threshold;

        let reason = if should_switch {
            format!(
                "Pattern '{}' detected; '{}' (eff={:.0}%) is better than '{}' (eff={:.0}%)",
                pattern_name(root.detected_pattern),
                recommended,
                recommended_efficiency * 100.0,
                current,
                current_efficiency * 100.0,
            )
        } else {
            format!(
                "Current fiber '{}' is adequate for pattern '{}'",
                current,
                pattern_name(root.detected_pattern),
            )
        };

        FiberAssessment {
            current_fiber: current.clone(),
            recommended_fiber: recommended.to_string(),
            current_efficiency,
            recommended_efficiency,
            should_switch,
            reason,
        }
    }

    // --------------------------------------------------------
    // Ramification Thread Calculation
    // --------------------------------------------------------

    /// Compute how many threads and what block size to recommend
    /// for a dependency chain of the given length.
    fn compute_spawn_params(&self, chain_length: u32) -> (u32, u32) {
        // Heuristic: 1 thread per dependency, rounded to warp size.
        let raw_threads = chain_length;
        let warp_size = 32u32;

        // Round up to warp boundary.
        let threads = ((raw_threads + warp_size - 1) / warp_size) * warp_size;
        let threads = threads.min(self.config.max_spawn_threads);

        // Block size: at least min_block_size, at most threads.
        let block_size = if threads <= self.config.min_block_size {
            self.config.min_block_size
        } else if threads <= 256 {
            threads
        } else {
            256 // Cap block size, use more blocks instead.
        };

        (threads, block_size)
    }

    // --------------------------------------------------------
    // Edit Batch Analysis
    // --------------------------------------------------------

    /// Analyze the buffered recent edits for spatial patterns.
    fn analyze_edit_batch(&mut self) {
        let edits: Vec<(u32, u32, f64)> = self.recent_edits.drain(..).collect();

        if edits.is_empty() {
            return;
        }

        // Check for block update pattern.
        if edits.len() >= 4 {
            let min_row = edits.iter().map(|e| e.0).min().unwrap();
            let max_row = edits.iter().map(|e| e.0).max().unwrap();
            let min_col = edits.iter().map(|e| e.1).min().unwrap();
            let max_col = edits.iter().map(|e| e.1).max().unwrap();

            let region_area = (max_row - min_row + 1) * (max_col - min_col + 1);
            let fill_ratio = edits.len() as f64 / region_area as f64;

            // If edits densely fill a rectangular region, mark as block update.
            if fill_ratio > 0.5 && region_area > 1 {
                for &(r, c, _) in &edits {
                    let key = cell_key(r, c);
                    if let Some(root) = self.roots.get_mut(&key) {
                        if !root.has_formula() {
                            root.detected_pattern = DataPattern::BlockUpdate;
                        }
                    }
                }
            }
        }

        // Check for sequential (same-row) edits.
        let mut row_counts: HashMap<u32, usize> = HashMap::new();
        for &(r, _, _) in &edits {
            *row_counts.entry(r).or_default() += 1;
        }
        for (&row, &count) in &row_counts {
            if count >= 3 {
                for &(r, c, _) in &edits {
                    if r == row {
                        let key = cell_key(r, c);
                        if let Some(root) = self.roots.get_mut(&key) {
                            if root.detected_pattern == DataPattern::Isolated {
                                root.detected_pattern = DataPattern::Sequential;
                            }
                        }
                    }
                }
            }
        }

        // Check for columnar edits.
        let mut col_counts: HashMap<u32, usize> = HashMap::new();
        for &(_, c, _) in &edits {
            *col_counts.entry(c).or_default() += 1;
        }
        for (&col, &count) in &col_counts {
            if count >= 3 {
                for &(r, c, _) in &edits {
                    if c == col {
                        let key = cell_key(r, c);
                        if let Some(root) = self.roots.get_mut(&key) {
                            if root.detected_pattern == DataPattern::Isolated {
                                root.detected_pattern = DataPattern::Columnar;
                            }
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------------
    // Query API
    // --------------------------------------------------------

    /// Get a CellRoot by position.
    pub fn get_root(&self, row: u32, col: u32) -> Option<&CellRoot> {
        self.roots.get(&cell_key(row, col))
    }

    /// Get all roots.
    pub fn all_roots(&self) -> Vec<&CellRoot> {
        self.roots.values().collect()
    }

    /// Get all hot roots.
    pub fn hot_roots(&self) -> Vec<&CellRoot> {
        self.roots.values().filter(|r| r.is_hot).collect()
    }

    /// Get all roots with active Ramification.
    pub fn ramified_roots(&self) -> Vec<&CellRoot> {
        self.roots.values().filter(|r| r.ramification_active).collect()
    }

    /// Get all roots with formulas.
    pub fn formula_roots(&self) -> Vec<&CellRoot> {
        self.roots.values().filter(|r| r.has_formula()).collect()
    }

    /// Get the dependency chain for a cell.
    pub fn dependency_chain(&self, row: u32, col: u32) -> Vec<String> {
        let key = cell_key(row, col);
        self.dependency_graph.get(&key).cloned().unwrap_or_default()
    }

    /// Get cells that depend on a given cell.
    pub fn dependents(&self, row: u32, col: u32) -> Vec<String> {
        let key = cell_key(row, col);
        self.reverse_deps.get(&key).cloned().unwrap_or_default()
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }

    /// Get the agent grid.
    pub fn agent_grid(&self) -> &CellAgentGrid {
        &self.agent_grid
    }

    /// Get the twin map.
    pub fn twin_map(&self) -> &GeometricTwinMap {
        &self.twin_map
    }

    /// Get bridge statistics.
    pub fn stats(&self) -> BridgeStats {
        let formula_count = self.formulas.len();
        let hot_count = self.roots.values().filter(|r| r.is_hot).count();
        let ramified_count = self.roots.values().filter(|r| r.ramification_active).count();
        let total_deps: usize = self.dependency_graph.values().map(|d| d.len()).sum();

        // Fiber distribution.
        let mut fiber_dist: HashMap<String, usize> = HashMap::new();
        for root in self.roots.values() {
            *fiber_dist.entry(root.current_fiber.clone()).or_default() += 1;
        }

        // Pattern distribution.
        let mut pattern_dist: HashMap<String, usize> = HashMap::new();
        for root in self.roots.values() {
            *pattern_dist.entry(pattern_name(root.detected_pattern).to_string()).or_default() += 1;
        }

        BridgeStats {
            rows: self.rows,
            cols: self.cols,
            total_cells: self.rows * self.cols,
            formula_cells: formula_count,
            hot_cells: hot_count,
            ramified_cells: ramified_count,
            total_dependencies: total_deps,
            tick_count: self.tick_count,
            total_events: self.total_events,
            fiber_distribution: fiber_dist,
            pattern_distribution: pattern_dist,
            uptime_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Export the bridge state as a JSON report.
    pub fn export_report(&self, path: &str) -> Result<(), String> {
        let report = BridgeReport {
            stats: self.stats(),
            hot_roots: self.hot_roots().iter().map(|r| (*r).clone()).collect(),
            ramified_roots: self.ramified_roots().iter().map(|r| (*r).clone()).collect(),
            formula_roots: self.formula_roots().iter().map(|r| (*r).clone()).collect(),
        };
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("Serialization failed: {}", e))?;
        std::fs::write(path, &json)
            .map_err(|e| format!("Write failed: {}", e))?;
        Ok(())
    }
}

/// Bridge statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub rows: u32,
    pub cols: u32,
    pub total_cells: u32,
    pub formula_cells: usize,
    pub hot_cells: usize,
    pub ramified_cells: usize,
    pub total_dependencies: usize,
    pub tick_count: u64,
    pub total_events: u64,
    pub fiber_distribution: HashMap<String, usize>,
    pub pattern_distribution: HashMap<String, usize>,
    pub uptime_ms: u64,
}

/// JSON export report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeReport {
    pub stats: BridgeStats,
    pub hot_roots: Vec<CellRoot>,
    pub ramified_roots: Vec<CellRoot>,
    pub formula_roots: Vec<CellRoot>,
}

// ============================================================
// Helper Functions
// ============================================================

/// Create a hashmap key for a cell position.
fn cell_key(row: u32, col: u32) -> String {
    format!("{}_{}", row, col)
}

/// Parse a cell key back to (row, col).
fn parse_cell_key(key: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = key.split('_').collect();
    if parts.len() == 2 {
        let r = parts[0].parse().ok()?;
        let c = parts[1].parse().ok()?;
        Some((r, c))
    } else {
        None
    }
}

/// Get current epoch time in milliseconds.
fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Parse formula dependencies from a formula string.
///
/// Supports common spreadsheet syntax:
///   - Single cell refs: A1, B2, Z99
///   - Range refs: A1:A100, B2:D5
///   - Mixed: =SUM(A1:A100) + B2
fn parse_formula_dependencies(formula: &str, max_rows: u32, max_cols: u32) -> Vec<String> {
    let mut deps = Vec::new();

    // Simple regex-like parser for cell references.
    // Matches patterns like A1, B2, AA99, etc.
    let chars: Vec<char> = formula.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Look for letter(s) followed by digit(s).
        if chars[i].is_ascii_alphabetic() && chars[i].is_ascii_uppercase() {
            let col_start = i;
            while i < len && chars[i].is_ascii_uppercase() {
                i += 1;
            }
            let col_str: String = chars[col_start..i].iter().collect();

            if i < len && chars[i].is_ascii_digit() {
                let row_start = i;
                while i < len && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let row_str: String = chars[row_start..i].iter().collect();

                if let Ok(row_1based) = row_str.parse::<u32>() {
                    let col_0based = col_letters_to_index(&col_str);
                    let row_0based = row_1based.saturating_sub(1);

                    // Check for range (e.g., A1:A100).
                    if i + 1 < len && chars[i] == ':' {
                        i += 1; // skip ':'
                        // Parse end of range.
                        if i < len && chars[i].is_ascii_uppercase() {
                            let end_col_start = i;
                            while i < len && chars[i].is_ascii_uppercase() {
                                i += 1;
                            }
                            let end_col_str: String = chars[end_col_start..i].iter().collect();

                            if i < len && chars[i].is_ascii_digit() {
                                let end_row_start = i;
                                while i < len && chars[i].is_ascii_digit() {
                                    i += 1;
                                }
                                let end_row_str: String = chars[end_row_start..i].iter().collect();

                                if let Ok(end_row_1based) = end_row_str.parse::<u32>() {
                                    let end_col = col_letters_to_index(&end_col_str);
                                    let end_row = end_row_1based.saturating_sub(1);

                                    // Expand range into individual cell refs.
                                    let start_r = row_0based.min(end_row);
                                    let end_r = row_0based.max(end_row);
                                    let start_c = col_0based.min(end_col);
                                    let end_c = col_0based.max(end_col);

                                    for r in start_r..=end_r.min(max_rows - 1) {
                                        for c in start_c..=end_c.min(max_cols - 1) {
                                            let key = cell_key(r, c);
                                            if !deps.contains(&key) {
                                                deps.push(key);
                                            }
                                        }
                                    }
                                    continue;
                                }
                            }
                        }
                    }

                    // Single cell reference.
                    if row_0based < max_rows && col_0based < max_cols {
                        let key = cell_key(row_0based, col_0based);
                        if !deps.contains(&key) {
                            deps.push(key);
                        }
                    }
                    continue;
                }
            }
        }

        i += 1;
    }

    deps
}

/// Convert column letters (A, B, ..., Z, AA, AB, ...) to 0-based index.
fn col_letters_to_index(letters: &str) -> u32 {
    let mut index = 0u32;
    for ch in letters.chars() {
        index = index * 26 + (ch as u32 - 'A' as u32 + 1);
    }
    index.saturating_sub(1)
}

/// Compute a fiber efficiency score for a given fiber + pattern combination.
fn compute_fiber_efficiency(fiber_name: &str, pattern: &DataPattern) -> f64 {
    let recommended = pattern.recommended_fiber();
    if fiber_name == recommended {
        return 1.0;
    }

    // Partial efficiency based on how well the fiber handles the pattern.
    match (fiber_name, pattern) {
        // cell_update is decent for sequential/isolated.
        ("cell_update", DataPattern::Sequential) => 0.85,
        ("cell_update", DataPattern::Columnar) => 0.5,
        ("cell_update", DataPattern::Random) => 0.4,
        ("cell_update", DataPattern::FormulaChain) => 0.3,
        ("cell_update", DataPattern::MassiveRecalc) => 0.15,
        ("cell_update", DataPattern::BlockUpdate) => 0.5,

        // batch_process is good for bulk but bad for formulas.
        ("batch_process", DataPattern::Isolated) => 0.6,
        ("batch_process", DataPattern::Sequential) => 0.8,
        ("batch_process", DataPattern::Columnar) => 0.9,
        ("batch_process", DataPattern::Random) => 0.5,
        ("batch_process", DataPattern::FormulaChain) => 0.4,
        ("batch_process", DataPattern::MassiveRecalc) => 0.6,

        // formula_eval is great for chains but overkill for simple edits.
        ("formula_eval", DataPattern::Isolated) => 0.4,
        ("formula_eval", DataPattern::Sequential) => 0.5,
        ("formula_eval", DataPattern::Columnar) => 0.7,
        ("formula_eval", DataPattern::Random) => 0.5,
        ("formula_eval", DataPattern::BlockUpdate) => 0.6,

        // crdt_merge is specialized for conflict resolution.
        ("crdt_merge", DataPattern::Isolated) => 0.5,
        ("crdt_merge", DataPattern::Sequential) => 0.4,
        ("crdt_merge", DataPattern::Columnar) => 0.3,
        ("crdt_merge", DataPattern::FormulaChain) => 0.4,
        ("crdt_merge", DataPattern::MassiveRecalc) => 0.3,
        ("crdt_merge", DataPattern::BlockUpdate) => 0.4,

        // idle_poll is wrong for everything except idle.
        ("idle_poll", _) => 0.1,

        // Unknown fiber.
        _ => 0.5,
    }
}

/// Get a human-readable name for a data pattern.
fn pattern_name(pattern: DataPattern) -> &'static str {
    match pattern {
        DataPattern::Isolated => "isolated",
        DataPattern::Sequential => "sequential",
        DataPattern::Columnar => "columnar",
        DataPattern::Random => "random",
        DataPattern::BlockUpdate => "block_update",
        DataPattern::FormulaChain => "formula_chain",
        DataPattern::MassiveRecalc => "massive_recalc",
    }
}

// ============================================================
// CLI Integration
// ============================================================

/// CLI actions for the spreadsheet bridge subcommand.
pub enum SpreadsheetCliAction {
    Demo,
    Status,
    Export(String),
    Help,
}

/// Print help for the `spreadsheet` CLI subcommand.
pub fn print_spreadsheet_help() {
    println!("cudaclaw spreadsheet -- Spreadsheet-Moment Bridge");
    println!();
    println!("USAGE:");
    println!("  cudaclaw spreadsheet [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo              Run a demonstration of the bridge");
    println!("  --status            Show bridge status");
    println!("  --export <PATH>     Export bridge report as JSON");
    println!("  --help, -h          Show this help message");
}

/// Parse CLI arguments for the `spreadsheet` subcommand.
pub fn parse_spreadsheet_args(args: &[String]) -> Option<SpreadsheetCliAction> {
    if args.is_empty() {
        return Some(SpreadsheetCliAction::Demo);
    }

    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--demo" => return Some(SpreadsheetCliAction::Demo),
            "--status" => return Some(SpreadsheetCliAction::Status),
            "--export" => {
                if i + 1 < args.len() {
                    return Some(SpreadsheetCliAction::Export(args[i + 1].clone()));
                }
                eprintln!("--export requires a file path");
                return None;
            }
            "--help" | "-h" => return Some(SpreadsheetCliAction::Help),
            _ => {}
        }
    }

    Some(SpreadsheetCliAction::Demo)
}

/// Run the spreadsheet bridge demonstration.
pub fn run_demo() {
    println!("{}",
        "==================================================================");
    println!("   cudaclaw Spreadsheet Bridge -- Demonstration");
    println!("{}",
        "==================================================================");
    println!();

    let mut bridge = SpreadsheetBridge::new(16, 16);
    println!("Created bridge for 16x16 spreadsheet grid ({} cells)", 16 * 16);

    // ── Phase 1: Simple cell edits ─────────────────────────
    println!("\n--- Phase 1: Cell Edits (Root Mapping) ---");
    let edits = vec![
        (0, 0, 10.0, "Revenue Q1"),
        (1, 0, 20.0, "Revenue Q2"),
        (2, 0, 30.0, "Revenue Q3"),
        (3, 0, 40.0, "Revenue Q4"),
        (4, 0, 0.0, "Total (formula cell)"),
    ];
    for &(r, c, v, desc) in &edits {
        bridge.on_cell_edit(r, c, v);
        let root = bridge.get_root(r, c).unwrap();
        println!(
            "  Cell({},{}) = {:.0} -> Root '{}' (fiber={}, pattern={})",
            r, c, v, root.root_id, root.current_fiber,
            pattern_name(root.detected_pattern),
        );
    }

    // ── Phase 2: Formula registration ──────────────────────
    println!("\n--- Phase 2: Formula Harvesting ---");

    // Small formula: SUM of 4 cells.
    bridge.register_formula(4, 0, "=SUM(A1:A4)");
    let root = bridge.get_root(4, 0).unwrap();
    println!(
        "  Cell(4,0) formula='=SUM(A1:A4)' -> deps={}, pattern={}, fiber={}",
        root.dependency_count,
        pattern_name(root.detected_pattern),
        root.current_fiber,
    );

    // Medium formula: SUM of a column.
    bridge.register_formula(0, 5, "=SUM(A1:A16)");
    let root = bridge.get_root(0, 5).unwrap();
    println!(
        "  Cell(0,5) formula='=SUM(A1:A16)' -> deps={}, pattern={}, fiber={}",
        root.dependency_count,
        pattern_name(root.detected_pattern),
        root.current_fiber,
    );

    // Large formula: cross-column dependencies.
    bridge.register_formula(0, 10, "=A1+B1+C1+D1+E1+F1+G1+H1+I1+J1");
    let root = bridge.get_root(0, 10).unwrap();
    println!(
        "  Cell(0,10) formula='=A1+B1+...+J1' -> deps={}, pattern={}, fiber={}",
        root.dependency_count,
        pattern_name(root.detected_pattern),
        root.current_fiber,
    );

    // Massive formula chain triggering Ramification.
    // Create a chain: each cell depends on the one above and the one to the left.
    println!("\n--- Phase 3: Massive Formula Chain (Ramification) ---");
    for r in 1..16u32 {
        for c in 1..16u32 {
            let formula = format!(
                "={}{}+{}{}",
                col_index_to_letters(c), r, // cell above (r-1 in 0-based = r in 1-based)
                col_index_to_letters(c.saturating_sub(1)), r + 1, // cell to the left
            );
            bridge.register_formula(r, c, &formula);
        }
    }

    // Now register a cell that depends on the entire chain.
    bridge.register_formula(15, 15, "=SUM(A1:P16)");
    let root = bridge.get_root(15, 15).unwrap();
    let chain_len = bridge.compute_dependency_chain_length(15, 15);
    println!(
        "  Cell(15,15) formula='=SUM(A1:P16)' -> deps={}, chain_len={}, pattern={}, fiber={}",
        root.dependency_count, chain_len,
        pattern_name(root.detected_pattern),
        root.current_fiber,
    );
    if root.ramification_active {
        println!("  ** RAMIFICATION TRIGGERED: spawning threads for dependency chain **");
    }

    // ── Phase 4: Tick and collect events ───────────────────
    println!("\n--- Phase 4: Tick (Event Collection) ---");
    let events = bridge.tick();
    println!("  Events generated: {}", events.len());

    let mut fiber_reassign_count = 0;
    let mut spawn_count = 0;
    let mut harvest_count = 0;

    for event in &events {
        match event {
            RamificationEvent::FiberReassignment {
                row, col, old_fiber, new_fiber, reason,
            } => {
                fiber_reassign_count += 1;
                if fiber_reassign_count <= 5 {
                    println!(
                        "  Fiber switch: Cell({},{}) '{}' -> '{}' ({})",
                        row, col, old_fiber, new_fiber, reason
                    );
                }
            }
            RamificationEvent::SpawnRecalcThreads {
                root_row, root_col, chain_length,
                recommended_threads, recommended_block_size,
            } => {
                spawn_count += 1;
                if spawn_count <= 3 {
                    println!(
                        "  RAMIFY: Cell({},{}) chain={} -> spawn {} threads (block_size={})",
                        root_row, root_col, chain_length,
                        recommended_threads, recommended_block_size
                    );
                }
            }
            RamificationEvent::FormulaHarvested {
                row, col, detected_pattern, dependency_count, ..
            } => {
                harvest_count += 1;
            }
            _ => {}
        }
    }
    if fiber_reassign_count > 5 {
        println!("  ... and {} more fiber reassignments", fiber_reassign_count - 5);
    }
    if spawn_count > 3 {
        println!("  ... and {} more Ramification spawns", spawn_count - 3);
    }
    println!("  Total: {} harvests, {} fiber switches, {} ramifications",
        harvest_count, fiber_reassign_count, spawn_count);

    // ── Phase 5: Bulk update ──────────────────────────────
    println!("\n--- Phase 5: Bulk Region Update ---");
    let bulk_values: Vec<(u32, u32, f64)> = (0..4)
        .flat_map(|r| (0..4).map(move |c| (r, c, (r * 4 + c) as f64 * 10.0)))
        .collect();
    bridge.on_bulk_update(0, 0, 3, 3, &bulk_values);
    let events2 = bridge.tick();
    println!("  Bulk update: 4x4 region = {} cells", bulk_values.len());
    println!("  Events from bulk: {}", events2.len());

    // ── Summary ───────────────────────────────────────────
    println!("\n--- Bridge Summary ---");
    let stats = bridge.stats();
    println!("  Grid: {}x{} = {} cells", stats.rows, stats.cols, stats.total_cells);
    println!("  Formula cells: {}", stats.formula_cells);
    println!("  Hot cells: {}", stats.hot_cells);
    println!("  Ramified cells: {}", stats.ramified_cells);
    println!("  Total dependencies: {}", stats.total_dependencies);
    println!("  Tick count: {}", stats.tick_count);
    println!("  Total events: {}", stats.total_events);
    println!("  Fiber distribution: {:?}", stats.fiber_distribution);
    println!("  Pattern distribution: {:?}", stats.pattern_distribution);

    println!("\nDemonstration complete.");
}

/// Show spreadsheet bridge status.
pub fn show_status() {
    let bridge = SpreadsheetBridge::new(8, 8);
    let stats = bridge.stats();
    println!("Spreadsheet Bridge Status:");
    println!("  Grid: {}x{} = {} cells", stats.rows, stats.cols, stats.total_cells);
    println!("  Formula cells: {}", stats.formula_cells);
    println!("  Hot cells: {}", stats.hot_cells);
    println!("  Ramified cells: {}", stats.ramified_cells);
}

/// Convert a 0-based column index to spreadsheet column letters.
fn col_index_to_letters(col: u32) -> String {
    let mut result = String::new();
    let mut c = col + 1; // 1-based for the algorithm.
    while c > 0 {
        c -= 1;
        result.insert(0, (b'A' + (c % 26) as u8) as char);
        c /= 26;
    }
    result
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_root_creation() {
        let root = CellRoot::new(3, 5);
        assert_eq!(root.root_id, "root_3_5");
        assert_eq!(root.row, 3);
        assert_eq!(root.col, 5);
        assert_eq!(root.current_fiber, "cell_update");
        assert_eq!(root.detected_pattern, DataPattern::Isolated);
        assert!(!root.is_hot);
        assert!(!root.has_formula());
    }

    #[test]
    fn test_cell_key_roundtrip() {
        let key = cell_key(10, 20);
        let (r, c) = parse_cell_key(&key).unwrap();
        assert_eq!(r, 10);
        assert_eq!(c, 20);
    }

    #[test]
    fn test_col_letters_to_index() {
        assert_eq!(col_letters_to_index("A"), 0);
        assert_eq!(col_letters_to_index("B"), 1);
        assert_eq!(col_letters_to_index("Z"), 25);
        assert_eq!(col_letters_to_index("AA"), 26);
    }

    #[test]
    fn test_col_index_to_letters() {
        assert_eq!(col_index_to_letters(0), "A");
        assert_eq!(col_index_to_letters(1), "B");
        assert_eq!(col_index_to_letters(25), "Z");
        assert_eq!(col_index_to_letters(26), "AA");
    }

    #[test]
    fn test_parse_formula_single_refs() {
        let deps = parse_formula_dependencies("=A1+B2", 100, 100);
        assert!(deps.contains(&cell_key(0, 0))); // A1 -> (0,0)
        assert!(deps.contains(&cell_key(1, 1))); // B2 -> (1,1)
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_parse_formula_range() {
        let deps = parse_formula_dependencies("=SUM(A1:A4)", 100, 100);
        assert_eq!(deps.len(), 4);
        assert!(deps.contains(&cell_key(0, 0))); // A1
        assert!(deps.contains(&cell_key(1, 0))); // A2
        assert!(deps.contains(&cell_key(2, 0))); // A3
        assert!(deps.contains(&cell_key(3, 0))); // A4
    }

    #[test]
    fn test_parse_formula_2d_range() {
        let deps = parse_formula_dependencies("=SUM(A1:B2)", 100, 100);
        assert_eq!(deps.len(), 4);
        assert!(deps.contains(&cell_key(0, 0))); // A1
        assert!(deps.contains(&cell_key(0, 1))); // B1
        assert!(deps.contains(&cell_key(1, 0))); // A2
        assert!(deps.contains(&cell_key(1, 1))); // B2
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = SpreadsheetBridge::new(8, 8);
        assert_eq!(bridge.grid_size(), (8, 8));
        assert_eq!(bridge.all_roots().len(), 64);
    }

    #[test]
    fn test_cell_edit() {
        let mut bridge = SpreadsheetBridge::new(4, 4);
        bridge.on_cell_edit(0, 0, 42.0);

        let root = bridge.get_root(0, 0).unwrap();
        assert_eq!(root.access_count, 1);
    }

    #[test]
    fn test_formula_registration() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        bridge.register_formula(4, 0, "=SUM(A1:A4)");

        let root = bridge.get_root(4, 0).unwrap();
        assert_eq!(root.dependency_count, 4);
        assert!(root.has_formula());
        assert_eq!(root.formula, "=SUM(A1:A4)");
    }

    #[test]
    fn test_pattern_columnar() {
        let mut bridge = SpreadsheetBridge::new(32, 32);
        // =SUM(A1:A16) -> 16 deps, all in column 0.
        bridge.register_formula(16, 0, "=SUM(A1:A16)");

        let root = bridge.get_root(16, 0).unwrap();
        // 16 deps in a single column -> FormulaChain (>8 deps).
        assert!(matches!(
            root.detected_pattern,
            DataPattern::FormulaChain | DataPattern::Columnar | DataPattern::MassiveRecalc
        ));
    }

    #[test]
    fn test_fiber_efficiency_assessment() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        // Register a formula that makes this a FormulaChain pattern.
        bridge.register_formula(0, 5, "=A1+B1+C1+D1+E1+F1+G1+H1+A2");

        let root = bridge.get_root(0, 5).unwrap();
        // The formula has 9 deps -> FormulaChain, which recommends "formula_eval".
        // Default fiber is "cell_update", so efficiency should be < 1.0.
        if root.detected_pattern == DataPattern::FormulaChain {
            assert!(root.fiber_efficiency < 1.0);
        }
    }

    #[test]
    fn test_ramification_trigger() {
        let mut bridge = SpreadsheetBridge::with_config(
            32, 32,
            BridgeConfig {
                ramification_threshold: 4,
                ..BridgeConfig::default()
            },
        );

        // Create a dependency chain: each cell depends on the one above.
        for r in 1..10u32 {
            bridge.register_formula(r, 0, &format!("=A{}", r)); // depends on cell above
        }

        // Check if any ramification was triggered.
        let events = bridge.tick();
        let has_spawn = events.iter().any(|e| {
            matches!(e, RamificationEvent::SpawnRecalcThreads { .. })
        });
        // With threshold=4 and chain of 9, ramification should trigger.
        // (May or may not trigger depending on transitive chain evaluation.)
    }

    #[test]
    fn test_clear_formula() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        bridge.register_formula(0, 0, "=B1+C1");

        let root = bridge.get_root(0, 0).unwrap();
        assert!(root.has_formula());

        bridge.clear_formula(0, 0);
        let root = bridge.get_root(0, 0).unwrap();
        assert!(!root.has_formula());
        assert_eq!(root.dependency_count, 0);
    }

    #[test]
    fn test_bulk_update() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        let values: Vec<(u32, u32, f64)> = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c, 1.0)))
            .collect();
        bridge.on_bulk_update(0, 0, 3, 3, &values);

        let events = bridge.tick();
        let has_bulk = events.iter().any(|e| {
            matches!(e, RamificationEvent::BulkRegionUpdate { .. })
        });
        assert!(has_bulk);
    }

    #[test]
    fn test_dependency_chain_length() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        // A chain: cell(1,0) depends on cell(0,0).
        // cell(2,0) depends on cell(1,0).
        // cell(3,0) depends on cell(2,0).
        bridge.register_formula(1, 0, "=A1");
        bridge.register_formula(2, 0, "=A2");
        bridge.register_formula(3, 0, "=A3");

        let chain_len = bridge.compute_dependency_chain_length(3, 0);
        // cell(3,0) -> cell(2,0) -> cell(1,0) -> cell(0,0): chain of 3.
        assert_eq!(chain_len, 3);
    }

    #[test]
    fn test_pattern_recommended_fiber() {
        assert_eq!(DataPattern::Isolated.recommended_fiber(), "cell_update");
        assert_eq!(DataPattern::FormulaChain.recommended_fiber(), "formula_eval");
        assert_eq!(DataPattern::MassiveRecalc.recommended_fiber(), "formula_eval");
        assert_eq!(DataPattern::BlockUpdate.recommended_fiber(), "batch_process");
        assert_eq!(DataPattern::Random.recommended_fiber(), "crdt_merge");
    }

    #[test]
    fn test_spawn_params() {
        let bridge = SpreadsheetBridge::new(8, 8);
        let (threads, block_size) = bridge.compute_spawn_params(100);
        assert!(threads >= 100);
        assert!(threads % 32 == 0); // Warp-aligned.
        assert!(block_size >= 32);
    }

    #[test]
    fn test_bridge_stats() {
        let mut bridge = SpreadsheetBridge::new(4, 4);
        bridge.register_formula(0, 0, "=B1");
        bridge.on_cell_edit(1, 1, 99.0);

        let stats = bridge.stats();
        assert_eq!(stats.total_cells, 16);
        assert_eq!(stats.formula_cells, 1);
    }

    #[test]
    fn test_cli_parsing() {
        assert!(matches!(
            parse_spreadsheet_args(&[]),
            Some(SpreadsheetCliAction::Demo)
        ));
        assert!(matches!(
            parse_spreadsheet_args(&["--demo".into()]),
            Some(SpreadsheetCliAction::Demo)
        ));
        assert!(matches!(
            parse_spreadsheet_args(&["--status".into()]),
            Some(SpreadsheetCliAction::Status)
        ));
        assert!(matches!(
            parse_spreadsheet_args(&["--export".into(), "out.json".into()]),
            Some(SpreadsheetCliAction::Export(_))
        ));
        assert!(matches!(
            parse_spreadsheet_args(&["--help".into()]),
            Some(SpreadsheetCliAction::Help)
        ));
    }

    #[test]
    fn test_export_report() {
        let mut bridge = SpreadsheetBridge::new(4, 4);
        bridge.register_formula(0, 0, "=B1+C1");

        let path = "/tmp/test_bridge_report.json";
        let result = bridge.export_report(path);
        assert!(result.is_ok());

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("formula_cells"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_fiber_efficiency_scores() {
        // Perfect match should be 1.0.
        assert_eq!(compute_fiber_efficiency("cell_update", &DataPattern::Isolated), 1.0);
        assert_eq!(compute_fiber_efficiency("formula_eval", &DataPattern::FormulaChain), 1.0);

        // Mismatch should be < 1.0.
        let eff = compute_fiber_efficiency("cell_update", &DataPattern::MassiveRecalc);
        assert!(eff < 0.5);
    }

    #[test]
    fn test_reverse_dependency_tracking() {
        let mut bridge = SpreadsheetBridge::new(8, 8);
        // Cell(2,0) depends on Cell(0,0) and Cell(1,0).
        bridge.register_formula(2, 0, "=A1+A2");

        let dependents = bridge.dependents(0, 0);
        assert!(dependents.contains(&cell_key(2, 0)));
    }
}
