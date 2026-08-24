// ============================================================
// Geometric Twin — Cell-to-Constraint Graph Mapping
// ============================================================
//
// Every spreadsheet cell is a "geometric twin" node in a
// constraint graph. When a cell is operated on, the constraints
// inherited from its twin node are evaluated. This creates a
// living topology where the spreadsheet grid IS the constraint
// graph.
//
// TOPOLOGY:
//   Each cell (row, col) maps to a TwinNode that carries:
//     - A unique node ID
//     - Inherited constraint IDs (from the DNA)
//     - Adjacency edges to neighbouring twin nodes
//     - A "fiber affinity" that tells the kernel launcher
//       which Muscle Fiber (optimized kernel) best serves
//       this node's workload pattern
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Twin Node
// ============================================================

/// A node in the geometric twin graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinNode {
    /// Unique node ID (derived from cell coordinates).
    pub id: String,
    /// Row in the spreadsheet grid.
    pub row: u32,
    /// Column in the spreadsheet grid.
    pub col: u32,
    /// Constraint IDs inherited by this node.
    pub inherited_constraint_ids: Vec<String>,
    /// Adjacency list: IDs of neighbouring twin nodes.
    pub neighbors: Vec<String>,
    /// Muscle Fiber affinity tag (set by ML feedback).
    pub fiber_affinity: String,
    /// Weight of this node (proportional to access frequency).
    pub weight: f64,
    /// Whether this node is a "hot" node (frequently accessed).
    pub is_hot: bool,
}

impl TwinNode {
    /// Create a new twin node for a cell.
    pub fn new(row: u32, col: u32) -> Self {
        TwinNode {
            id: format!("twin_{}_{}", row, col),
            row,
            col,
            inherited_constraint_ids: Vec::new(),
            neighbors: Vec::new(),
            fiber_affinity: "default".into(),
            weight: 1.0,
            is_hot: false,
        }
    }

    /// Add a constraint to this node.
    pub fn inherit_constraint(&mut self, constraint_id: String) {
        if !self.inherited_constraint_ids.contains(&constraint_id) {
            self.inherited_constraint_ids.push(constraint_id);
        }
    }

    /// Add a neighbor edge.
    pub fn add_neighbor(&mut self, neighbor_id: String) {
        if !self.neighbors.contains(&neighbor_id) {
            self.neighbors.push(neighbor_id);
        }
    }

    /// Increase weight (called on each access).
    pub fn touch(&mut self) {
        self.weight += 1.0;
        self.is_hot = self.weight > 100.0;
    }
}

// ============================================================
// Twin Binding
// ============================================================

/// A binding between a spreadsheet cell and its twin node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinBinding {
    /// Cell row.
    pub cell_row: u32,
    /// Cell column.
    pub cell_col: u32,
    /// Twin node ID.
    pub twin_node_id: String,
    /// Constraint IDs inherited through this binding.
    pub inherited_constraint_ids: Vec<String>,
    /// SM index where this cell's agent is located.
    pub sm_affinity: Option<u32>,
}

// ============================================================
// Twin Topology
// ============================================================

/// The edge structure connecting twin nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinTopology {
    /// Total node count.
    pub node_count: usize,
    /// Total edge count.
    pub edge_count: usize,
    /// Maximum degree (neighbors) of any node.
    pub max_degree: usize,
    /// Average degree.
    pub avg_degree: f64,
    /// Number of hot nodes.
    pub hot_node_count: usize,
}

// ============================================================
// Geometric Twin Map
// ============================================================

/// The complete geometric twin graph for a spreadsheet grid.
pub struct GeometricTwinMap {
    /// Grid dimensions.
    rows: u32,
    cols: u32,
    /// All twin nodes, keyed by node ID.
    nodes: HashMap<String, TwinNode>,
    /// All bindings (one per cell).
    bindings: Vec<TwinBinding>,
    /// Default constraint IDs to inherit for every node.
    default_constraints: Vec<String>,
}

impl GeometricTwinMap {
    /// Create a new empty twin map for a grid.
    pub fn new(rows: u32, cols: u32) -> Self {
        GeometricTwinMap {
            rows,
            cols,
            nodes: HashMap::new(),
            bindings: Vec::new(),
            default_constraints: vec![
                "resource.register_budget".into(),
                "resource.shared_memory_ceiling".into(),
                "latency.p99_rtt_ceiling".into(),
                "correctness.crdt_monotonicity".into(),
                "efficiency.min_coalescing_ratio".into(),
            ],
        }
    }

    /// Build the default 4-connected grid topology.
    /// Each cell gets a twin node connected to its N/S/E/W neighbors.
    pub fn build_default_topology(&mut self) {
        // Create all nodes.
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut node = TwinNode::new(r, c);
                for cid in &self.default_constraints {
                    node.inherit_constraint(cid.clone());
                }
                self.nodes.insert(node.id.clone(), node);
            }
        }

        // Wire adjacency (4-connected grid).
        let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let node_id = format!("twin_{}_{}", r, c);
                let mut neighbor_ids = Vec::new();
                for (dr, dc) in &directions {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < self.rows as i32 && nc >= 0 && nc < self.cols as i32 {
                        neighbor_ids.push(format!("twin_{}_{}", nr, nc));
                    }
                }
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    for nid in neighbor_ids {
                        node.add_neighbor(nid);
                    }
                }
            }
        }

        // Build bindings.
        self.bindings.clear();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let node_id = format!("twin_{}_{}", r, c);
                let inherited = self.default_constraints.clone();
                self.bindings.push(TwinBinding {
                    cell_row: r,
                    cell_col: c,
                    twin_node_id: node_id,
                    inherited_constraint_ids: inherited,
                    sm_affinity: None,
                });
            }
        }
    }

    /// Get a twin node by cell coordinates.
    pub fn get_node(&self, row: u32, col: u32) -> Option<&TwinNode> {
        let id = format!("twin_{}_{}", row, col);
        self.nodes.get(&id)
    }

    /// Get a mutable twin node by cell coordinates.
    pub fn get_node_mut(&mut self, row: u32, col: u32) -> Option<&mut TwinNode> {
        let id = format!("twin_{}_{}", row, col);
        self.nodes.get_mut(&id)
    }

    /// Touch a node (record an access).
    pub fn touch(&mut self, row: u32, col: u32) {
        if let Some(node) = self.get_node_mut(row, col) {
            node.touch();
        }
    }

    /// Get the constraint IDs for a cell.
    pub fn constraints_for_cell(&self, row: u32, col: u32) -> Vec<String> {
        match self.get_node(row, col) {
            Some(node) => node.inherited_constraint_ids.clone(),
            None => Vec::new(),
        }
    }

    /// Set SM affinity for a cell's binding.
    pub fn set_sm_affinity(&mut self, row: u32, col: u32, sm: u32) {
        for binding in &mut self.bindings {
            if binding.cell_row == row && binding.cell_col == col {
                binding.sm_affinity = Some(sm);
            }
        }
    }

    /// Set the fiber affinity for a node (called by ML feedback).
    pub fn set_fiber_affinity(&mut self, row: u32, col: u32, fiber: String) {
        if let Some(node) = self.get_node_mut(row, col) {
            node.fiber_affinity = fiber;
        }
    }

    /// Add a custom constraint to a specific cell's twin node.
    pub fn add_constraint_to_cell(&mut self, row: u32, col: u32, constraint_id: String) {
        if let Some(node) = self.get_node_mut(row, col) {
            node.inherit_constraint(constraint_id.clone());
        }
        for binding in &mut self.bindings {
            if binding.cell_row == row && binding.cell_col == col {
                if !binding.inherited_constraint_ids.contains(&constraint_id) {
                    binding.inherited_constraint_ids.push(constraint_id.clone());
                }
            }
        }
    }

    /// Get the topology summary.
    pub fn topology(&self) -> TwinTopology {
        let max_degree = self.nodes.values().map(|n| n.neighbors.len()).max().unwrap_or(0);
        let total_edges: usize = self.nodes.values().map(|n| n.neighbors.len()).sum();
        let avg_degree = if self.nodes.is_empty() {
            0.0
        } else {
            total_edges as f64 / self.nodes.len() as f64
        };
        let hot_count = self.nodes.values().filter(|n| n.is_hot).count();

        TwinTopology {
            node_count: self.nodes.len(),
            edge_count: total_edges / 2, // Each edge is counted twice.
            max_degree,
            avg_degree,
            hot_node_count: hot_count,
        }
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Get all bindings.
    pub fn bindings(&self) -> &[TwinBinding] {
        &self.bindings
    }

    /// Get all hot nodes.
    pub fn hot_nodes(&self) -> Vec<&TwinNode> {
        self.nodes.values().filter(|n| n.is_hot).collect()
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }

    /// Get the fiber affinity distribution.
    pub fn fiber_distribution(&self) -> HashMap<String, usize> {
        let mut dist = HashMap::new();
        for node in self.nodes.values() {
            *dist.entry(node.fiber_affinity.clone()).or_insert(0) += 1;
        }
        dist
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twin_node_creation() {
        let node = TwinNode::new(3, 5);
        assert_eq!(node.id, "twin_3_5");
        assert_eq!(node.row, 3);
        assert_eq!(node.col, 5);
        assert!(!node.is_hot);
    }

    #[test]
    fn test_twin_node_touch() {
        let mut node = TwinNode::new(0, 0);
        for _ in 0..101 {
            node.touch();
        }
        assert!(node.is_hot);
        assert!(node.weight > 100.0);
    }

    #[test]
    fn test_twin_map_default_topology() {
        let mut map = GeometricTwinMap::new(4, 4);
        map.build_default_topology();

        assert_eq!(map.node_count(), 16);
        assert_eq!(map.binding_count(), 16);

        // Corner node has 2 neighbors.
        let corner = map.get_node(0, 0).unwrap();
        assert_eq!(corner.neighbors.len(), 2);

        // Edge node has 3 neighbors.
        let edge = map.get_node(0, 1).unwrap();
        assert_eq!(edge.neighbors.len(), 3);

        // Interior node has 4 neighbors.
        let interior = map.get_node(1, 1).unwrap();
        assert_eq!(interior.neighbors.len(), 4);
    }

    #[test]
    fn test_constraints_for_cell() {
        let mut map = GeometricTwinMap::new(2, 2);
        map.build_default_topology();

        let constraints = map.constraints_for_cell(0, 0);
        assert!(constraints.contains(&"resource.register_budget".to_string()));
        assert!(constraints.contains(&"latency.p99_rtt_ceiling".to_string()));
    }

    #[test]
    fn test_sm_affinity() {
        let mut map = GeometricTwinMap::new(2, 2);
        map.build_default_topology();

        map.set_sm_affinity(0, 0, 5);
        let binding = map.bindings().iter().find(|b| b.cell_row == 0 && b.cell_col == 0).unwrap();
        assert_eq!(binding.sm_affinity, Some(5));
    }

    #[test]
    fn test_fiber_affinity() {
        let mut map = GeometricTwinMap::new(2, 2);
        map.build_default_topology();

        map.set_fiber_affinity(1, 1, "crdt_merge".into());
        let node = map.get_node(1, 1).unwrap();
        assert_eq!(node.fiber_affinity, "crdt_merge");
    }

    #[test]
    fn test_topology_summary() {
        let mut map = GeometricTwinMap::new(3, 3);
        map.build_default_topology();

        let topo = map.topology();
        assert_eq!(topo.node_count, 9);
        assert_eq!(topo.max_degree, 4);
        assert_eq!(topo.hot_node_count, 0);
    }

    #[test]
    fn test_add_constraint_to_cell() {
        let mut map = GeometricTwinMap::new(2, 2);
        map.build_default_topology();

        map.add_constraint_to_cell(0, 0, "custom.my_constraint".into());
        let node = map.get_node(0, 0).unwrap();
        assert!(node.inherited_constraint_ids.contains(&"custom.my_constraint".to_string()));
    }

    #[test]
    fn test_fiber_distribution() {
        let mut map = GeometricTwinMap::new(2, 2);
        map.build_default_topology();

        map.set_fiber_affinity(0, 0, "crdt_merge".into());
        map.set_fiber_affinity(0, 1, "crdt_merge".into());

        let dist = map.fiber_distribution();
        assert_eq!(*dist.get("crdt_merge").unwrap_or(&0), 2);
        assert_eq!(*dist.get("default").unwrap_or(&0), 2);
    }
}
