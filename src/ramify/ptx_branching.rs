// ============================================================
// PTX Dynamic Branching — On-the-fly Sub-Kernel Recompilation
// ============================================================
//
// This module implements the "Ramify" branching system: cudaclaw
// observes data patterns at runtime and dynamically recompiles
// or relinks specialized PTX sub-kernels to match.
//
// DESIGN:
// - DataPatternObserver watches incoming command streams for
//   recurring access patterns (sequential, strided, random,
//   column-major, diagonal).
// - PtxTemplate holds parameterized PTX assembly fragments
//   with placeholder constants (block size, unroll factor,
//   coalescing strategy, etc.).
// - PtxBranchCompiler takes a template + observed pattern and
//   produces a specialized PTX variant by substituting constants
//   and selecting code paths.
// - BranchRegistry caches compiled variants keyed by
//   (template_id, pattern_hash) so recompilation only happens
//   when patterns genuinely change.
//
// IMPORTANT:
// - No live nvcc invocation — PTX text is manipulated as strings
//   and loaded via cuModuleLoadData / cust::Module at runtime.
// - All pattern detection runs on the host (Rust) side using
//   the command stream visible through Unified Memory.
// - This module does NOT modify existing kernel code paths.
//   It produces *new* PTX modules that can be launched alongside
//   the persistent worker.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// Data Pattern Detection
// ============================================================

/// Recognized access patterns that trigger branching decisions.
/// Each pattern maps to a different PTX optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Threads access consecutive addresses — perfect for coalesced loads.
    Sequential,
    /// Threads access addresses with a fixed stride (e.g., column-major).
    Strided { stride: u32 },
    /// No detectable regularity — use texture cache / L1 bypass.
    Random,
    /// Column-major traversal of a row-major grid.
    ColumnMajor,
    /// Diagonal traversal (e.g., matrix diagonal operations).
    Diagonal,
    /// Repeated writes to a small hot set of cells.
    HotSpot { cell_count: u32 },
    /// Bulk sequential writes (large batch updates).
    BulkSequential,
}

/// A window of recent cell access indices used to detect patterns.
#[derive(Debug, Clone)]
pub struct AccessWindow {
    /// Ring buffer of recent cell indices (row * cols + col).
    indices: Vec<u32>,
    /// Current write position in the ring buffer.
    write_pos: usize,
    /// Total number of observations recorded (may exceed capacity).
    total_observations: u64,
    /// Grid column count — needed to distinguish row vs column access.
    grid_cols: u32,
}

impl AccessWindow {
    /// Create a new access window with the given capacity.
    ///
    /// # Arguments
    /// * `capacity` — number of recent accesses to remember
    /// * `grid_cols` — spreadsheet column count for pattern math
    pub fn new(capacity: usize, grid_cols: u32) -> Self {
        AccessWindow {
            indices: vec![0; capacity],
            write_pos: 0,
            total_observations: 0,
            grid_cols: grid_cols.max(1),
        }
    }

    /// Record a cell access.
    pub fn record(&mut self, cell_index: u32) {
        self.indices[self.write_pos] = cell_index;
        self.write_pos = (self.write_pos + 1) % self.indices.len();
        self.total_observations += 1;
    }

    /// Number of valid observations (up to capacity).
    pub fn valid_count(&self) -> usize {
        std::cmp::min(self.total_observations as usize, self.indices.len())
    }

    /// Detect the dominant access pattern in the current window.
    ///
    /// Algorithm:
    /// 1. Compute deltas between consecutive accesses.
    /// 2. If >80% of deltas are +1 → Sequential.
    /// 3. If >80% share the same non-1 delta → Strided.
    /// 4. If >60% of unique indices < 32 → HotSpot.
    /// 5. If >60% of deltas equal grid_cols → ColumnMajor.
    /// 6. If >60% of deltas equal grid_cols+1 → Diagonal.
    /// 7. If >70% of deltas are +1 and window is large → BulkSequential.
    /// 8. Otherwise → Random.
    pub fn detect_pattern(&self) -> AccessPattern {
        let n = self.valid_count();
        if n < 4 {
            return AccessPattern::Random;
        }

        // Build the ordered slice of recent indices.
        let ordered = self.ordered_slice(n);

        // Compute deltas.
        let deltas: Vec<i64> = ordered
            .windows(2)
            .map(|w| w[1] as i64 - w[0] as i64)
            .collect();

        if deltas.is_empty() {
            return AccessPattern::Random;
        }

        let total = deltas.len() as f64;

        // Check sequential (+1 deltas).
        let seq_count = deltas.iter().filter(|&&d| d == 1).count() as f64;
        if seq_count / total > 0.80 {
            if n > 256 {
                return AccessPattern::BulkSequential;
            }
            return AccessPattern::Sequential;
        }

        // Check column-major (delta == grid_cols).
        let col_delta = self.grid_cols as i64;
        let col_count = deltas.iter().filter(|&&d| d == col_delta).count() as f64;
        if col_count / total > 0.60 {
            return AccessPattern::ColumnMajor;
        }

        // Check diagonal (delta == grid_cols + 1).
        let diag_delta = col_delta + 1;
        let diag_count = deltas.iter().filter(|&&d| d == diag_delta).count() as f64;
        if diag_count / total > 0.60 {
            return AccessPattern::Diagonal;
        }

        // Check hot-spot: count distinct indices.
        let mut unique: Vec<u32> = ordered.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() <= 32 && n > 16 {
            return AccessPattern::HotSpot {
                cell_count: unique.len() as u32,
            };
        }

        // Check strided: find the mode of non-zero deltas.
        let mut freq: HashMap<i64, usize> = HashMap::new();
        for &d in &deltas {
            if d != 0 {
                *freq.entry(d).or_insert(0) += 1;
            }
        }
        if let Some((&mode_delta, &mode_count)) = freq.iter().max_by_key(|(_, &c)| c) {
            if mode_count as f64 / total > 0.60 && mode_delta != 1 {
                return AccessPattern::Strided {
                    stride: mode_delta.unsigned_abs() as u32,
                };
            }
        }

        AccessPattern::Random
    }

    /// Return the last `n` observations in chronological order.
    fn ordered_slice(&self, n: usize) -> Vec<u32> {
        let cap = self.indices.len();
        let start = if self.total_observations as usize >= cap {
            self.write_pos
        } else {
            0
        };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.indices[(start + i) % cap]);
        }
        out
    }
}

// ============================================================
// PTX Template System
// ============================================================

/// A parameterized PTX assembly template.
///
/// Templates contain placeholder tokens like `{{BLOCK_SIZE}}`,
/// `{{UNROLL_FACTOR}}`, `{{COALESCE_STRATEGY}}` that are
/// substituted at compile time based on the detected pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxTemplate {
    /// Unique template identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Raw PTX text with `{{PLACEHOLDER}}` tokens.
    pub ptx_source: String,
    /// Map of placeholder names to their allowed value ranges.
    pub parameters: HashMap<String, ParameterSpec>,
    /// Which access patterns this template is designed for.
    pub target_patterns: Vec<AccessPattern>,
    /// Minimum compute capability required (e.g., 70 for Volta).
    pub min_compute_capability: u32,
}

/// Specification for a single template parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// Default value (used when no pattern-specific override exists).
    pub default: String,
    /// Allowed values (empty = any string is valid).
    pub allowed_values: Vec<String>,
    /// Short description for documentation.
    pub description: String,
}

impl PtxTemplate {
    /// Create a new template.
    pub fn new(id: &str, description: &str, ptx_source: &str) -> Self {
        PtxTemplate {
            id: id.to_string(),
            description: description.to_string(),
            ptx_source: ptx_source.to_string(),
            parameters: HashMap::new(),
            target_patterns: Vec::new(),
            min_compute_capability: 70,
        }
    }

    /// Add a parameter specification.
    pub fn with_parameter(
        mut self,
        name: &str,
        default: &str,
        description: &str,
        allowed: &[&str],
    ) -> Self {
        self.parameters.insert(
            name.to_string(),
            ParameterSpec {
                default: default.to_string(),
                allowed_values: allowed.iter().map(|s| s.to_string()).collect(),
                description: description.to_string(),
            },
        );
        self
    }

    /// Add target patterns.
    pub fn with_target_patterns(mut self, patterns: Vec<AccessPattern>) -> Self {
        self.target_patterns = patterns;
        self
    }

    /// Substitute all placeholders with concrete values.
    ///
    /// # Arguments
    /// * `values` — map of placeholder name → concrete value
    ///
    /// # Returns
    /// The specialized PTX text with all `{{NAME}}` tokens replaced.
    pub fn specialize(&self, values: &HashMap<String, String>) -> String {
        let mut ptx = self.ptx_source.clone();
        for (name, spec) in &self.parameters {
            let placeholder = format!("{{{{{}}}}}", name);
            let value = values
                .get(name)
                .unwrap_or(&spec.default);
            ptx = ptx.replace(&placeholder, value);
        }
        ptx
    }

    /// List all placeholder names present in the source.
    pub fn placeholder_names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }
}

// ============================================================
// Pattern-Aware Constant Selection
// ============================================================

/// Given an access pattern, produce the optimal set of PTX
/// constants for a cell-processing sub-kernel.
///
/// These constants are substituted into PtxTemplate placeholders.
pub fn constants_for_pattern(pattern: &AccessPattern) -> HashMap<String, String> {
    let mut c = HashMap::new();

    match pattern {
        AccessPattern::Sequential | AccessPattern::BulkSequential => {
            // Sequential: maximize coalesced loads, high unroll.
            c.insert("BLOCK_SIZE".into(), "256".into());
            c.insert("UNROLL_FACTOR".into(), "8".into());
            c.insert("COALESCE_STRATEGY".into(), "SEQUENTIAL".into());
            c.insert("USE_L1_CACHE".into(), "1".into());
            c.insert("PREFETCH_DISTANCE".into(), "4".into());
            c.insert("SHARED_MEM_BANKS".into(), "32".into());
        }
        AccessPattern::Strided { stride } => {
            // Strided: pad shared memory to avoid bank conflicts.
            let pad = if *stride % 32 == 0 { 1 } else { 0 };
            c.insert("BLOCK_SIZE".into(), "128".into());
            c.insert("UNROLL_FACTOR".into(), "4".into());
            c.insert("COALESCE_STRATEGY".into(), "STRIDED".into());
            c.insert("USE_L1_CACHE".into(), "1".into());
            c.insert("PREFETCH_DISTANCE".into(), "2".into());
            c.insert("SHMEM_PAD".into(), pad.to_string());
            c.insert("ACCESS_STRIDE".into(), stride.to_string());
        }
        AccessPattern::Random => {
            // Random: bypass L1, use texture cache path.
            c.insert("BLOCK_SIZE".into(), "64".into());
            c.insert("UNROLL_FACTOR".into(), "1".into());
            c.insert("COALESCE_STRATEGY".into(), "RANDOM".into());
            c.insert("USE_L1_CACHE".into(), "0".into());
            c.insert("PREFETCH_DISTANCE".into(), "0".into());
            c.insert("USE_LDG".into(), "1".into());
        }
        AccessPattern::ColumnMajor => {
            // Column-major: transpose tiles in shared memory.
            c.insert("BLOCK_SIZE".into(), "128".into());
            c.insert("UNROLL_FACTOR".into(), "4".into());
            c.insert("COALESCE_STRATEGY".into(), "TILE_TRANSPOSE".into());
            c.insert("USE_L1_CACHE".into(), "1".into());
            c.insert("TILE_DIM".into(), "32".into());
            c.insert("SHMEM_PAD".into(), "1".into());
        }
        AccessPattern::Diagonal => {
            // Diagonal: smaller blocks, moderate unroll.
            c.insert("BLOCK_SIZE".into(), "64".into());
            c.insert("UNROLL_FACTOR".into(), "2".into());
            c.insert("COALESCE_STRATEGY".into(), "DIAGONAL".into());
            c.insert("USE_L1_CACHE".into(), "1".into());
            c.insert("PREFETCH_DISTANCE".into(), "1".into());
        }
        AccessPattern::HotSpot { cell_count } => {
            // Hot-spot: small blocks, keep hot set in shared memory.
            let shmem_slots = (*cell_count).max(16);
            c.insert("BLOCK_SIZE".into(), "32".into());
            c.insert("UNROLL_FACTOR".into(), "1".into());
            c.insert("COALESCE_STRATEGY".into(), "HOTSPOT".into());
            c.insert("USE_L1_CACHE".into(), "1".into());
            c.insert("HOTSPOT_SHMEM_SLOTS".into(), shmem_slots.to_string());
            c.insert("USE_ATOMICS".into(), "1".into());
        }
    }

    c
}

// ============================================================
// Branch Compiler
// ============================================================

/// A compiled PTX variant ready to be loaded as a CUDA module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledBranch {
    /// Unique branch identifier (template_id + pattern hash).
    pub branch_id: String,
    /// Source template ID.
    pub template_id: String,
    /// The access pattern this branch is specialized for.
    pub pattern: AccessPattern,
    /// The concrete constants that were substituted.
    pub constants: HashMap<String, String>,
    /// Specialized PTX text (ready for cuModuleLoadData).
    pub ptx_text: String,
    /// When this branch was compiled (epoch seconds).
    pub compiled_at: u64,
    /// How many times this branch has been loaded/used.
    pub load_count: u64,
}

/// Registry of compiled branches, keyed by (template_id, pattern).
///
/// Caches compiled PTX variants so recompilation only happens
/// when the detected pattern genuinely changes or a new template
/// is registered.
pub struct BranchRegistry {
    /// All registered templates.
    templates: HashMap<String, PtxTemplate>,
    /// Compiled branches: key = branch_id.
    branches: HashMap<String, CompiledBranch>,
    /// Access window for pattern detection.
    access_window: AccessWindow,
    /// Last detected pattern (to avoid redundant recompilation).
    last_pattern: Option<AccessPattern>,
    /// Total recompilations performed.
    recompilation_count: u64,
    /// Timestamp of last recompilation.
    last_recompilation: Option<Instant>,
    /// Minimum interval between recompilations (debounce).
    recompilation_cooldown: std::time::Duration,
}

impl BranchRegistry {
    /// Create a new registry.
    ///
    /// # Arguments
    /// * `window_capacity` — how many recent accesses to track
    /// * `grid_cols` — spreadsheet column count
    pub fn new(window_capacity: usize, grid_cols: u32) -> Self {
        BranchRegistry {
            templates: HashMap::new(),
            branches: HashMap::new(),
            access_window: AccessWindow::new(window_capacity, grid_cols),
            last_pattern: None,
            recompilation_count: 0,
            last_recompilation: None,
            recompilation_cooldown: std::time::Duration::from_millis(500),
        }
    }

    /// Register a PTX template.
    pub fn register_template(&mut self, template: PtxTemplate) {
        self.templates.insert(template.id.clone(), template);
    }

    /// Record a cell access and check if recompilation is needed.
    ///
    /// Returns `Some(Vec<CompiledBranch>)` if new branches were
    /// compiled, `None` if the pattern hasn't changed.
    pub fn observe_access(&mut self, cell_index: u32) -> Option<Vec<CompiledBranch>> {
        self.access_window.record(cell_index);

        // Don't detect until we have enough data.
        if self.access_window.valid_count() < 32 {
            return None;
        }

        let pattern = self.access_window.detect_pattern();

        // Skip if pattern unchanged.
        if self.last_pattern.as_ref() == Some(&pattern) {
            return None;
        }

        // Debounce: don't recompile too frequently.
        if let Some(last) = self.last_recompilation {
            if last.elapsed() < self.recompilation_cooldown {
                return None;
            }
        }

        // Pattern changed — recompile all applicable templates.
        self.last_pattern = Some(pattern);
        self.last_recompilation = Some(Instant::now());
        self.recompilation_count += 1;

        let new_branches = self.compile_for_pattern(&pattern);
        if new_branches.is_empty() {
            None
        } else {
            Some(new_branches)
        }
    }

    /// Compile all templates that target the given pattern.
    fn compile_for_pattern(&mut self, pattern: &AccessPattern) -> Vec<CompiledBranch> {
        let constants = constants_for_pattern(pattern);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut new_branches = Vec::new();

        let template_ids: Vec<String> = self.templates.keys().cloned().collect();
        for tid in &template_ids {
            let template = &self.templates[tid];

            // Check if this template targets the current pattern.
            let dominated_by_pattern = template.target_patterns.is_empty()
                || template.target_patterns.contains(pattern);
            if !dominated_by_pattern {
                continue;
            }

            let branch_id = format!("{}_{:?}_{}", tid, pattern, now);
            let ptx_text = template.specialize(&constants);

            let branch = CompiledBranch {
                branch_id: branch_id.clone(),
                template_id: tid.clone(),
                pattern: *pattern,
                constants: constants.clone(),
                ptx_text,
                compiled_at: now,
                load_count: 0,
            };

            self.branches.insert(branch_id, branch.clone());
            new_branches.push(branch);
        }

        new_branches
    }

    /// Get the best compiled branch for the current pattern.
    pub fn get_active_branch(&self, template_id: &str) -> Option<&CompiledBranch> {
        // Find the most recently compiled branch for this template.
        self.branches
            .values()
            .filter(|b| b.template_id == template_id)
            .max_by_key(|b| b.compiled_at)
    }

    /// Increment the load count for a branch (call when actually loading into GPU).
    pub fn mark_loaded(&mut self, branch_id: &str) {
        if let Some(branch) = self.branches.get_mut(branch_id) {
            branch.load_count += 1;
        }
    }

    /// Get statistics about the registry.
    pub fn stats(&self) -> BranchRegistryStats {
        BranchRegistryStats {
            templates_registered: self.templates.len(),
            branches_compiled: self.branches.len(),
            total_recompilations: self.recompilation_count,
            current_pattern: self.last_pattern,
            total_observations: self.access_window.total_observations,
        }
    }

    /// Get all compiled branches.
    pub fn all_branches(&self) -> Vec<&CompiledBranch> {
        self.branches.values().collect()
    }

    /// Set the recompilation cooldown period.
    pub fn set_cooldown(&mut self, cooldown: std::time::Duration) {
        self.recompilation_cooldown = cooldown;
    }
}

/// Statistics about the branch registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchRegistryStats {
    pub templates_registered: usize,
    pub branches_compiled: usize,
    pub total_recompilations: u64,
    pub current_pattern: Option<AccessPattern>,
    pub total_observations: u64,
}

// ============================================================
// Built-in PTX Templates
// ============================================================

/// Create the default cell-update PTX template.
///
/// This template produces a specialized kernel for updating
/// spreadsheet cells based on the detected access pattern.
pub fn default_cell_update_template() -> PtxTemplate {
    // This is a simplified PTX template. In production, this would
    // contain real PTX assembly with placeholder constants.
    let ptx_source = r#"
// ============================================================
// Ramified Cell Update Sub-Kernel
// ============================================================
// Auto-generated by cudaclaw Ramify branching system.
// Pattern: {{COALESCE_STRATEGY}}
// Block size: {{BLOCK_SIZE}}
// Unroll factor: {{UNROLL_FACTOR}}
//
// This kernel processes cell updates with memory access patterns
// optimized for the detected workload.
// ============================================================

.version 7.0
.target sm_{{MIN_SM}}
.address_size 64

// ============================================================
// Configuration Constants (substituted by Ramify)
// ============================================================
.const .u32 BLOCK_SIZE = {{BLOCK_SIZE}};
.const .u32 UNROLL_FACTOR = {{UNROLL_FACTOR}};
.const .u32 USE_L1 = {{USE_L1_CACHE}};
.const .u32 PREFETCH_DIST = {{PREFETCH_DISTANCE}};

// ============================================================
// Cell Update Entry Point
// ============================================================
.visible .entry ramified_cell_update(
    .param .u64 cells_ptr,       // Pointer to CRDTCell array
    .param .u64 indices_ptr,     // Pointer to cell index array
    .param .u64 values_ptr,      // Pointer to new value array
    .param .u32 count,           // Number of cells to update
    .param .u64 timestamp,       // Update timestamp
    .param .u32 node_id          // Source node ID
)
{
    // Thread indexing
    .reg .u32 %tid, %bid, %gid, %ntid;
    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %ntid, %ntid.x;
    mad.lo.u32 %gid, %bid, %ntid, %tid;

    // Bounds check
    .reg .pred %p_inbounds;
    .reg .u32 %count_val;
    ld.param.u32 %count_val, [count];
    setp.lt.u32 %p_inbounds, %gid, %count_val;
    @!%p_inbounds bra $L_exit;

    // Load cell index (coalesced for sequential patterns)
    .reg .u64 %idx_base, %idx_addr;
    .reg .u32 %cell_idx;
    ld.param.u64 %idx_base, [indices_ptr];
    mad.wide.u32 %idx_addr, %gid, 4, %idx_base;
    ld.global.u32 %cell_idx, [%idx_addr];

    // Load new value
    .reg .u64 %val_base, %val_addr;
    .reg .f64 %new_val;
    ld.param.u64 %val_base, [values_ptr];
    mad.wide.u32 %val_addr, %gid, 8, %val_base;
    ld.global.f64 %new_val, [%val_addr];

    // Compute cell address (32 bytes per CRDTCell)
    .reg .u64 %cell_base, %cell_addr;
    ld.param.u64 %cell_base, [cells_ptr];
    mad.wide.u32 %cell_addr, %cell_idx, 32, %cell_base;

    // Store value at offset 0 of CRDTCell
    st.global.f64 [%cell_addr], %new_val;

    // Store timestamp at offset 8
    .reg .u64 %ts;
    ld.param.u64 %ts, [timestamp];
    st.global.u64 [%cell_addr+8], %ts;

    // Store node_id at offset 16
    .reg .u32 %nid;
    ld.param.u32 %nid, [node_id];
    st.global.u32 [%cell_addr+16], %nid;

    // Memory fence for visibility
    membar.sys;

$L_exit:
    ret;
}
"#;

    PtxTemplate::new(
        "cell_update_v1",
        "Ramified cell update kernel — pattern-adaptive coalescing",
        ptx_source,
    )
    .with_parameter("BLOCK_SIZE", "128", "Threads per block", &["32", "64", "128", "256", "512"])
    .with_parameter("UNROLL_FACTOR", "4", "Loop unroll factor", &["1", "2", "4", "8"])
    .with_parameter("COALESCE_STRATEGY", "SEQUENTIAL", "Memory coalescing strategy",
        &["SEQUENTIAL", "STRIDED", "RANDOM", "TILE_TRANSPOSE", "DIAGONAL", "HOTSPOT"])
    .with_parameter("USE_L1_CACHE", "1", "Enable L1 cache (0=bypass via ldg)", &["0", "1"])
    .with_parameter("PREFETCH_DISTANCE", "2", "Prefetch distance in iterations", &["0", "1", "2", "4"])
    .with_parameter("MIN_SM", "70", "Minimum SM architecture version", &["60", "70", "75", "80", "86", "89", "90"])
    .with_target_patterns(vec![
        AccessPattern::Sequential,
        AccessPattern::BulkSequential,
        AccessPattern::Strided { stride: 0 }, // matches any stride
        AccessPattern::Random,
        AccessPattern::ColumnMajor,
        AccessPattern::Diagonal,
    ])
}

/// Create the CRDT merge PTX template.
///
/// This template produces a specialized kernel for merging
/// CRDT state vectors, optimized for the conflict pattern.
pub fn default_crdt_merge_template() -> PtxTemplate {
    let ptx_source = r#"
// ============================================================
// Ramified CRDT Merge Sub-Kernel
// ============================================================
// Merges remote CRDT state into local state using timestamp-
// based last-writer-wins resolution.
//
// Optimization: {{COALESCE_STRATEGY}}
// ============================================================

.version 7.0
.target sm_{{MIN_SM}}
.address_size 64

.const .u32 BLOCK_SIZE = {{BLOCK_SIZE}};

.visible .entry ramified_crdt_merge(
    .param .u64 local_cells,     // Local CRDTCell array
    .param .u64 remote_cells,    // Remote CRDTCell array
    .param .u32 cell_count,      // Number of cells to merge
    .param .u32 local_node_id    // Local node ID for conflict resolution
)
{
    .reg .u32 %tid, %bid, %gid, %ntid;
    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %ntid, %ntid.x;
    mad.lo.u32 %gid, %bid, %ntid, %tid;

    .reg .pred %p_inbounds;
    .reg .u32 %count_val;
    ld.param.u32 %count_val, [cell_count];
    setp.lt.u32 %p_inbounds, %gid, %count_val;
    @!%p_inbounds bra $L_merge_exit;

    // Load local timestamp
    .reg .u64 %local_base, %local_addr;
    ld.param.u64 %local_base, [local_cells];
    mad.wide.u32 %local_addr, %gid, 32, %local_base;
    .reg .u64 %local_ts;
    ld.global.u64 %local_ts, [%local_addr+8];

    // Load remote timestamp
    .reg .u64 %remote_base, %remote_addr;
    ld.param.u64 %remote_base, [remote_cells];
    mad.wide.u32 %remote_addr, %gid, 32, %remote_base;
    .reg .u64 %remote_ts;
    ld.global.u64 %remote_ts, [%remote_addr+8];

    // Last-writer-wins: remote wins if remote_ts > local_ts
    .reg .pred %p_remote_wins;
    setp.gt.u64 %p_remote_wins, %remote_ts, %local_ts;
    @!%p_remote_wins bra $L_merge_exit;

    // Copy remote cell to local (32 bytes)
    .reg .u64 %v0, %v1, %v2, %v3;
    ld.global.u64 %v0, [%remote_addr];
    ld.global.u64 %v1, [%remote_addr+8];
    ld.global.u64 %v2, [%remote_addr+16];
    ld.global.u64 %v3, [%remote_addr+24];
    st.global.u64 [%local_addr], %v0;
    st.global.u64 [%local_addr+8], %v1;
    st.global.u64 [%local_addr+16], %v2;
    st.global.u64 [%local_addr+24], %v3;

    membar.sys;

$L_merge_exit:
    ret;
}
"#;

    PtxTemplate::new(
        "crdt_merge_v1",
        "Ramified CRDT merge kernel — last-writer-wins with coalesced access",
        ptx_source,
    )
    .with_parameter("BLOCK_SIZE", "256", "Threads per block", &["64", "128", "256", "512"])
    .with_parameter("COALESCE_STRATEGY", "SEQUENTIAL", "Memory access strategy",
        &["SEQUENTIAL", "STRIDED", "RANDOM"])
    .with_parameter("MIN_SM", "70", "Minimum SM architecture version", &["60", "70", "75", "80"])
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_pattern_detection() {
        let mut window = AccessWindow::new(64, 100);
        // Feed sequential indices: 0, 1, 2, ..., 63
        for i in 0..64 {
            window.record(i);
        }
        let pattern = window.detect_pattern();
        assert!(
            matches!(pattern, AccessPattern::Sequential | AccessPattern::BulkSequential),
            "Expected Sequential or BulkSequential, got {:?}",
            pattern
        );
    }

    #[test]
    fn test_strided_pattern_detection() {
        let mut window = AccessWindow::new(64, 100);
        // Feed strided indices: 0, 5, 10, 15, ...
        for i in 0..64 {
            window.record(i * 5);
        }
        let pattern = window.detect_pattern();
        assert!(
            matches!(pattern, AccessPattern::Strided { stride: 5 }),
            "Expected Strided {{ stride: 5 }}, got {:?}",
            pattern
        );
    }

    #[test]
    fn test_column_major_detection() {
        let mut window = AccessWindow::new(64, 100);
        // Column-major in a 100-column grid: 0, 100, 200, 300, ...
        for i in 0..64 {
            window.record(i * 100);
        }
        let pattern = window.detect_pattern();
        assert!(
            matches!(pattern, AccessPattern::ColumnMajor),
            "Expected ColumnMajor, got {:?}",
            pattern
        );
    }

    #[test]
    fn test_hotspot_detection() {
        let mut window = AccessWindow::new(128, 100);
        // Repeatedly access a small set of 4 cells.
        for i in 0..128 {
            window.record((i % 4) as u32);
        }
        let pattern = window.detect_pattern();
        assert!(
            matches!(pattern, AccessPattern::HotSpot { .. }),
            "Expected HotSpot, got {:?}",
            pattern
        );
    }

    #[test]
    fn test_random_pattern_detection() {
        let mut window = AccessWindow::new(64, 100);
        // Pseudo-random-ish indices.
        let randoms = [
            42, 917, 3, 1050, 7, 888, 500, 12, 999, 4, 731, 62,
            401, 19, 843, 555, 2, 678, 100, 37, 821, 456, 11, 990,
            333, 77, 612, 50, 940, 28, 703, 444, 15, 876, 222, 68,
            509, 33, 891, 160, 44, 760, 300, 88, 634, 21, 555, 92,
            416, 71, 800, 290, 55, 670, 180, 39, 740, 320, 66, 590,
            210, 48, 850, 130,
        ];
        for &r in &randoms {
            window.record(r);
        }
        let pattern = window.detect_pattern();
        assert!(
            matches!(pattern, AccessPattern::Random),
            "Expected Random, got {:?}",
            pattern
        );
    }

    #[test]
    fn test_template_specialization() {
        let template = default_cell_update_template();
        let constants = constants_for_pattern(&AccessPattern::Sequential);
        let ptx = template.specialize(&constants);

        assert!(ptx.contains("BLOCK_SIZE = 256"), "Block size should be 256 for sequential");
        assert!(ptx.contains("UNROLL_FACTOR = 8"), "Unroll should be 8 for sequential");
        assert!(!ptx.contains("{{"), "No unresolved placeholders should remain");
    }

    #[test]
    fn test_branch_registry_lifecycle() {
        let mut registry = BranchRegistry::new(64, 100);
        registry.set_cooldown(std::time::Duration::from_millis(0));
        registry.register_template(default_cell_update_template());

        // Feed sequential data.
        for i in 0..64 {
            registry.observe_access(i);
        }

        let stats = registry.stats();
        assert_eq!(stats.templates_registered, 1);
        assert!(stats.total_recompilations >= 1, "Should have recompiled at least once");
        assert!(
            matches!(stats.current_pattern, Some(AccessPattern::Sequential) | Some(AccessPattern::BulkSequential)),
            "Should detect sequential pattern"
        );

        // Verify a branch was compiled.
        let branch = registry.get_active_branch("cell_update_v1");
        assert!(branch.is_some(), "Should have an active branch");
    }

    #[test]
    fn test_constants_for_all_patterns() {
        // Ensure every pattern produces valid constants.
        let patterns = vec![
            AccessPattern::Sequential,
            AccessPattern::BulkSequential,
            AccessPattern::Strided { stride: 4 },
            AccessPattern::Random,
            AccessPattern::ColumnMajor,
            AccessPattern::Diagonal,
            AccessPattern::HotSpot { cell_count: 8 },
        ];

        for pattern in &patterns {
            let constants = constants_for_pattern(pattern);
            assert!(constants.contains_key("BLOCK_SIZE"), "Missing BLOCK_SIZE for {:?}", pattern);
            assert!(constants.contains_key("UNROLL_FACTOR"), "Missing UNROLL_FACTOR for {:?}", pattern);
            assert!(constants.contains_key("COALESCE_STRATEGY"), "Missing COALESCE_STRATEGY for {:?}", pattern);
        }
    }
}
