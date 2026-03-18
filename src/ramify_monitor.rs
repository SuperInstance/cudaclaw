// ============================================================
// Ramify Monitor — Tree Visualization Dashboard
// ============================================================
//
// Visualizes the cudaclaw ecosystem as a living tree:
//
//   CANOPY  — High-level harvesting of data and spreadsheet
//             results. Aggregate throughput, cell update rates,
//             formula recalculation counts, CRDT merge success.
//
//   ROOTS   — Low-level PTX branches and hardware-level
//             throughput. SM utilization, register pressure,
//             shared memory budgets, active PTX variant info.
//
//   MICRO-  — Independent, disconnected agents exhausting
//   ORGANISMS  resources. Each micro-organism is an agent with
//             its own latency/heat "exhaust" that impacts
//             downstream components.
//
// The monitor can render to:
//   1. Terminal (ANSI color tree with live stats)
//   2. HTML dashboard (static file with JS auto-refresh)
//   3. JSON snapshot (machine-readable for CI/tooling)
//
// CLI:
//   cudaclaw monitor --demo       Run a demo with simulated data
//   cudaclaw monitor --html FILE  Generate HTML dashboard
//   cudaclaw monitor --json FILE  Export JSON snapshot
//   cudaclaw monitor --help       Show help
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// Tree Data Model
// ============================================================

/// A single node in the Ramify tree visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Unique identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Which layer this node belongs to.
    pub layer: TreeLayer,
    /// Metric value (meaning depends on layer/type).
    pub value: f64,
    /// Unit label for the value (e.g., "ops/s", "%", "KB").
    pub unit: String,
    /// Health status (0.0 = critical, 1.0 = healthy).
    pub health: f64,
    /// Optional detail string (shown on hover/expand).
    pub detail: String,
    /// Child node IDs.
    pub children: Vec<String>,
}

/// The three visualization layers of the tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeLayer {
    /// High-level data harvesting and spreadsheet results.
    Canopy,
    /// Low-level PTX branches and hardware throughput.
    Roots,
    /// Independent agents exhausting resources.
    MicroOrganism,
}

/// A connection between two nodes representing data flow or impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEdge {
    /// Source node ID.
    pub from: String,
    /// Destination node ID.
    pub to: String,
    /// Edge weight (bandwidth, frequency, etc.).
    pub weight: f64,
    /// Edge label.
    pub label: String,
}

/// Complete tree snapshot at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSnapshot {
    /// Timestamp (epoch seconds).
    pub timestamp: u64,
    /// All nodes in the tree.
    pub nodes: Vec<TreeNode>,
    /// Edges connecting nodes.
    pub edges: Vec<TreeEdge>,
    /// Global statistics.
    pub global_stats: GlobalStats,
}

/// Aggregate statistics across the entire system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStats {
    /// Total commands processed.
    pub total_commands: u64,
    /// Overall throughput (commands/second).
    pub throughput_ops: f64,
    /// System-wide P99 latency in microseconds.
    pub p99_latency_us: f64,
    /// Number of active PTX branches.
    pub active_ptx_branches: u32,
    /// Number of active shared memory bridges.
    pub active_bridges: u32,
    /// Number of registered agents.
    pub total_agents: u32,
    /// Number of critical SMs (nutrient score < threshold).
    pub critical_sms: u32,
    /// Average SM nutrient score (0.0-1.0).
    pub avg_nutrient_score: f64,
    /// GPU temperature (Celsius), if available.
    pub gpu_temp_celsius: Option<u32>,
    /// GPU utilization percentage, if available.
    pub gpu_utilization_pct: Option<u32>,
    /// Whether thermal throttling is active.
    pub thermal_throttle: bool,
    /// Uptime in seconds.
    pub uptime_secs: f64,
}

// ============================================================
// Tree Builder — constructs a TreeSnapshot from engine state
// ============================================================

/// Builds a TreeSnapshot from Ramify engine statistics and
/// simulated or real system data.
pub struct TreeBuilder {
    nodes: Vec<TreeNode>,
    edges: Vec<TreeEdge>,
    node_map: HashMap<String, usize>,
}

impl TreeBuilder {
    pub fn new() -> Self {
        TreeBuilder {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add a node to the tree.
    pub fn add_node(&mut self, node: TreeNode) {
        let idx = self.nodes.len();
        self.node_map.insert(node.id.clone(), idx);
        self.nodes.push(node);
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, edge: TreeEdge) {
        self.edges.push(edge);
    }

    /// Build the final snapshot.
    pub fn build(self, stats: GlobalStats) -> TreeSnapshot {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        TreeSnapshot {
            timestamp: now,
            nodes: self.nodes,
            edges: self.edges,
            global_stats: stats,
        }
    }

    /// Build a demonstration tree with simulated data.
    pub fn build_demo() -> TreeSnapshot {
        let mut builder = TreeBuilder::new();

        // ── CANOPY: High-level data harvesting ──────────────
        builder.add_node(TreeNode {
            id: "canopy_root".into(),
            label: "Data Harvesting".into(),
            layer: TreeLayer::Canopy,
            value: 2_450_000.0,
            unit: "cells/sec".into(),
            health: 0.92,
            detail: "Aggregate cell update throughput across all agents".into(),
            children: vec![
                "canopy_edits".into(),
                "canopy_formulas".into(),
                "canopy_crdt".into(),
                "canopy_results".into(),
            ],
        });

        builder.add_node(TreeNode {
            id: "canopy_edits".into(),
            label: "Cell Edits".into(),
            layer: TreeLayer::Canopy,
            value: 1_800_000.0,
            unit: "edits/sec".into(),
            health: 0.95,
            detail: "Direct cell value updates via lock-free dispatcher".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "canopy_formulas".into(),
            label: "Formula Recalc".into(),
            layer: TreeLayer::Canopy,
            value: 450_000.0,
            unit: "recalcs/sec".into(),
            health: 0.88,
            detail: "Parallel prefix-sum formula recalculation pipeline".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "canopy_crdt".into(),
            label: "CRDT Merges".into(),
            layer: TreeLayer::Canopy,
            value: 180_000.0,
            unit: "merges/sec".into(),
            health: 0.91,
            detail: "Warp-aggregated CAS merges with PTX atomics".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "canopy_results".into(),
            label: "Result Writeback".into(),
            layer: TreeLayer::Canopy,
            value: 98.7,
            unit: "% success".into(),
            health: 0.99,
            detail: "Completion flag writeback success rate".into(),
            children: vec![],
        });

        // ── ROOTS: PTX branches and hardware throughput ─────
        builder.add_node(TreeNode {
            id: "roots_root".into(),
            label: "Hardware Layer".into(),
            layer: TreeLayer::Roots,
            value: 128.0,
            unit: "SMs".into(),
            health: 0.85,
            detail: "Streaming Multiprocessor utilization and PTX variant info".into(),
            children: vec![
                "roots_ptx".into(),
                "roots_shmem".into(),
                "roots_regs".into(),
                "roots_pcie".into(),
            ],
        });

        builder.add_node(TreeNode {
            id: "roots_ptx".into(),
            label: "PTX Branches".into(),
            layer: TreeLayer::Roots,
            value: 4.0,
            unit: "variants".into(),
            health: 0.90,
            detail: "Active PTX variants: Sequential(256), Strided(128), Random(64), HotSpot(32)".into(),
            children: vec![
                "roots_ptx_seq".into(),
                "roots_ptx_strided".into(),
                "roots_ptx_random".into(),
            ],
        });

        builder.add_node(TreeNode {
            id: "roots_ptx_seq".into(),
            label: "Sequential".into(),
            layer: TreeLayer::Roots,
            value: 256.0,
            unit: "block_size".into(),
            health: 0.95,
            detail: "Coalesced loads, unroll=8, L1 enabled, prefetch=4".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "roots_ptx_strided".into(),
            label: "Strided".into(),
            layer: TreeLayer::Roots,
            value: 128.0,
            unit: "block_size".into(),
            health: 0.82,
            detail: "Padded shmem banks, unroll=4, stride-aware prefetch".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "roots_ptx_random".into(),
            label: "Random".into(),
            layer: TreeLayer::Roots,
            value: 64.0,
            unit: "block_size".into(),
            health: 0.65,
            detail: "L1 bypass, texture cache (__ldg), no prefetch".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "roots_shmem".into(),
            label: "Shared Memory".into(),
            layer: TreeLayer::Roots,
            value: 72.3,
            unit: "% used".into(),
            health: 0.73,
            detail: "58 KB working set + 2 active bridges = 66 KB / 96 KB per SM".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "roots_regs".into(),
            label: "Register File".into(),
            layer: TreeLayer::Roots,
            value: 61.5,
            unit: "% used".into(),
            health: 0.80,
            detail: "40,192 / 65,536 registers per SM across all agents".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "roots_pcie".into(),
            label: "PCIe Bus".into(),
            layer: TreeLayer::Roots,
            value: 3.2,
            unit: "us RTT".into(),
            health: 0.88,
            detail: "Host-GPU round-trip time via Unified Memory (P99)".into(),
            children: vec![],
        });

        // ── MICRO-ORGANISMS: Agents exhausting resources ────
        builder.add_node(TreeNode {
            id: "micro_root".into(),
            label: "Agent Ecosystem".into(),
            layer: TreeLayer::MicroOrganism,
            value: 5.0,
            unit: "agents".into(),
            health: 0.78,
            detail: "Independent GPU agents with resource exhaust metrics".into(),
            children: vec![
                "micro_greedy".into(),
                "micro_crdt".into(),
                "micro_formatter".into(),
                "micro_formula".into(),
                "micro_bridge".into(),
            ],
        });

        builder.add_node(TreeNode {
            id: "micro_greedy".into(),
            label: "greedy_formula_engine".into(),
            layer: TreeLayer::MicroOrganism,
            value: 91.5,
            unit: "% SM resources".into(),
            health: 0.25,
            detail: "SM 0 | 60K regs, 90KB shmem, 44 warps | PRUNED 2x, priority 15".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "micro_crdt".into(),
            label: "crdt_merger".into(),
            layer: TreeLayer::MicroOrganism,
            value: 18.3,
            unit: "% SM resources".into(),
            health: 0.82,
            detail: "SM 0 | 12K regs, 16KB shmem, 8 warps | priority 50".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "micro_formatter".into(),
            label: "cell_formatter".into(),
            layer: TreeLayer::MicroOrganism,
            value: 4.1,
            unit: "% SM resources".into(),
            health: 0.96,
            detail: "SM 1 | 4K regs, 2KB shmem, 2 warps | priority 80".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "micro_formula".into(),
            label: "parallel_recalc".into(),
            layer: TreeLayer::MicroOrganism,
            value: 35.2,
            unit: "% SM resources".into(),
            health: 0.70,
            detail: "SM 2 | 24K regs, 32KB shmem, 16 warps | Kogge-Stone scan active".into(),
            children: vec![],
        });

        builder.add_node(TreeNode {
            id: "micro_bridge".into(),
            label: "shmem_bridge_A1".into(),
            layer: TreeLayer::MicroOrganism,
            value: 265.0,
            unit: "ns saved/xfer".into(),
            health: 0.93,
            detail: "claw:A1 <-> smpclaw:twin_A1 | 8KB mailbox, SM 0 co-located".into(),
            children: vec![],
        });

        // ── Edges: data flow and impact ─────────────────────
        builder.add_edge(TreeEdge {
            from: "canopy_edits".into(),
            to: "roots_ptx_seq".into(),
            weight: 1_800_000.0,
            label: "coalesced edits".into(),
        });

        builder.add_edge(TreeEdge {
            from: "canopy_formulas".into(),
            to: "roots_ptx_strided".into(),
            weight: 450_000.0,
            label: "formula chains".into(),
        });

        builder.add_edge(TreeEdge {
            from: "canopy_crdt".into(),
            to: "roots_ptx_random".into(),
            weight: 180_000.0,
            label: "CAS merges".into(),
        });

        builder.add_edge(TreeEdge {
            from: "micro_greedy".into(),
            to: "roots_regs".into(),
            weight: 60000.0,
            label: "register exhaust".into(),
        });

        builder.add_edge(TreeEdge {
            from: "micro_greedy".into(),
            to: "roots_shmem".into(),
            weight: 90000.0,
            label: "shmem exhaust".into(),
        });

        builder.add_edge(TreeEdge {
            from: "micro_bridge".into(),
            to: "roots_shmem".into(),
            weight: 8192.0,
            label: "bridge allocation".into(),
        });

        builder.add_edge(TreeEdge {
            from: "micro_formula".into(),
            to: "canopy_formulas".into(),
            weight: 450_000.0,
            label: "recalc results".into(),
        });

        let stats = GlobalStats {
            total_commands: 12_500_000,
            throughput_ops: 2_450_000.0,
            p99_latency_us: 3.2,
            active_ptx_branches: 4,
            active_bridges: 1,
            total_agents: 5,
            critical_sms: 1,
            avg_nutrient_score: 0.72,
            gpu_temp_celsius: Some(68),
            gpu_utilization_pct: Some(87),
            thermal_throttle: false,
            uptime_secs: 42.7,
        };

        builder.build(stats)
    }
}

// ============================================================
// Terminal Renderer — ANSI color tree in the terminal
// ============================================================

mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYAN: &str = "\x1b[36m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const GREEN: &str = "\x1b[32m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const RED: &str = "\x1b[31m";
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BLUE: &str = "\x1b[34m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const WHITE: &str = "\x1b[37m";
    pub const BG_GREEN: &str = "\x1b[42m";
    pub const BG_YELLOW: &str = "\x1b[43m";
    pub const BG_RED: &str = "\x1b[41m";
}

/// Render a TreeSnapshot to the terminal as an ANSI-colored tree.
pub fn render_terminal(snapshot: &TreeSnapshot) {
    let bar = "═".repeat(66);
    println!("\n{}╔{}╗{}", colors::BRIGHT_CYAN, bar, colors::RESET);
    println!("{}║  {}cudaclaw Ramify Monitor — Ecosystem Tree Visualization{}       ║{}",
        colors::BRIGHT_CYAN, colors::BOLD, colors::RESET, colors::RESET);
    println!("{}╚{}╝{}\n", colors::BRIGHT_CYAN, bar, colors::RESET);

    // ── Global Stats Bar ────────────────────────────────────
    render_global_stats(&snapshot.global_stats);

    // ── Canopy Layer ────────────────────────────────────────
    println!("\n{}{}  CANOPY — Data Harvesting & Spreadsheet Results  {}",
        colors::BG_GREEN, colors::BOLD, colors::RESET);

    let canopy_nodes: Vec<&TreeNode> = snapshot.nodes.iter()
        .filter(|n| n.layer == TreeLayer::Canopy)
        .collect();

    if let Some(root) = canopy_nodes.iter().find(|n| n.id == "canopy_root") {
        render_node(root, 0, &snapshot.nodes);
        for child_id in &root.children {
            if let Some(child) = snapshot.nodes.iter().find(|n| n.id == *child_id) {
                render_node(child, 1, &snapshot.nodes);
            }
        }
    }

    // ── Roots Layer ─────────────────────────────────────────
    println!("\n{}{}  ROOTS — PTX Branches & Hardware Throughput  {}",
        colors::BG_YELLOW, colors::BOLD, colors::RESET);

    let roots_nodes: Vec<&TreeNode> = snapshot.nodes.iter()
        .filter(|n| n.layer == TreeLayer::Roots)
        .collect();

    if let Some(root) = roots_nodes.iter().find(|n| n.id == "roots_root") {
        render_node(root, 0, &snapshot.nodes);
        for child_id in &root.children {
            if let Some(child) = snapshot.nodes.iter().find(|n| n.id == *child_id) {
                render_node(child, 1, &snapshot.nodes);
                for grandchild_id in &child.children {
                    if let Some(gc) = snapshot.nodes.iter().find(|n| n.id == *grandchild_id) {
                        render_node(gc, 2, &snapshot.nodes);
                    }
                }
            }
        }
    }

    // ── Micro-Organisms Layer ───────────────────────────────
    println!("\n{}{}  MICRO-ORGANISMS — Agents & Resource Exhaust  {}",
        colors::BG_RED, colors::BOLD, colors::RESET);

    if let Some(root) = snapshot.nodes.iter().find(|n| n.id == "micro_root") {
        render_node(root, 0, &snapshot.nodes);
        for child_id in &root.children {
            if let Some(child) = snapshot.nodes.iter().find(|n| n.id == *child_id) {
                render_agent_node(child);
            }
        }
    }

    // ── Data Flow Edges ─────────────────────────────────────
    if !snapshot.edges.is_empty() {
        println!("\n{}  DATA FLOW & IMPACT CONNECTIONS{}", colors::DIM, colors::RESET);
        for edge in &snapshot.edges {
            let color = if edge.weight > 100_000.0 {
                colors::BRIGHT_GREEN
            } else if edge.weight > 1000.0 {
                colors::YELLOW
            } else {
                colors::DIM
            };
            println!("  {}  {} ──({:.0} {})──> {}{}",
                color, edge.from, edge.weight, edge.label, edge.to, colors::RESET);
        }
    }

    println!("\n{}─────────────────────────────────────────────────────────────{}", colors::DIM, colors::RESET);
}

fn render_global_stats(stats: &GlobalStats) {
    let throttle_indicator = if stats.thermal_throttle {
        format!("{}THROTTLED{}", colors::BRIGHT_RED, colors::RESET)
    } else {
        format!("{}OK{}", colors::BRIGHT_GREEN, colors::RESET)
    };

    let temp_str = stats.gpu_temp_celsius
        .map(|t| format!("{}C", t))
        .unwrap_or_else(|| "N/A".into());

    let util_str = stats.gpu_utilization_pct
        .map(|u| format!("{}%", u))
        .unwrap_or_else(|| "N/A".into());

    println!("{}Global Statistics:{}", colors::BOLD, colors::RESET);
    println!("  Throughput: {}{:.2}M ops/s{}  P99: {}{:.2} us{}  Commands: {}{}{}",
        colors::BRIGHT_GREEN, stats.throughput_ops / 1_000_000.0, colors::RESET,
        colors::BRIGHT_CYAN, stats.p99_latency_us, colors::RESET,
        colors::WHITE, format_count(stats.total_commands), colors::RESET);
    println!("  PTX Branches: {}{}{}  Bridges: {}{}{}  Agents: {}{}{}  Critical SMs: {}{}{}",
        colors::YELLOW, stats.active_ptx_branches, colors::RESET,
        colors::MAGENTA, stats.active_bridges, colors::RESET,
        colors::BLUE, stats.total_agents, colors::RESET,
        if stats.critical_sms > 0 { colors::BRIGHT_RED } else { colors::GREEN },
        stats.critical_sms, colors::RESET);
    println!("  GPU: {} | Temp: {} | Util: {} | Nutrient: {}{:.0}%{} | Uptime: {:.1}s",
        throttle_indicator, temp_str, util_str,
        nutrient_color(stats.avg_nutrient_score),
        stats.avg_nutrient_score * 100.0,
        colors::RESET,
        stats.uptime_secs);
}

fn render_node(node: &TreeNode, depth: usize, _all_nodes: &[TreeNode]) {
    let indent = "  ".repeat(depth + 1);
    let branch = if depth == 0 { "╠══" } else { "├──" };
    let health_bar = health_bar_str(node.health);
    let color = layer_color(node.layer);

    println!("{}{}{} {}{}{} ({}{:.1} {}{}){} {}",
        indent, color, branch,
        colors::BOLD, node.label, colors::RESET,
        colors::WHITE, node.value, node.unit, colors::RESET,
        colors::RESET,
        health_bar);

    if !node.detail.is_empty() {
        println!("{}    {}{}{}",
            indent, colors::DIM, node.detail, colors::RESET);
    }
}

fn render_agent_node(node: &TreeNode) {
    let health_bar = health_bar_str(node.health);
    let status_icon = if node.health < 0.3 {
        format!("{} !", colors::BRIGHT_RED)
    } else if node.health < 0.6 {
        format!("{} ~", colors::YELLOW)
    } else {
        format!("{} +", colors::GREEN)
    };

    println!("  {}├──{} {}{}{} [{:.1} {}] {} {}{}",
        colors::MAGENTA, colors::RESET,
        colors::BOLD, node.label, colors::RESET,
        node.value, node.unit,
        health_bar,
        status_icon, colors::RESET);

    if !node.detail.is_empty() {
        println!("  {}    {}{}{}",
            colors::DIM, colors::DIM, node.detail, colors::RESET);
    }
}

fn health_bar_str(health: f64) -> String {
    let filled = (health * 10.0).round() as usize;
    let empty = 10 - filled.min(10);
    let color = if health >= 0.8 {
        colors::BRIGHT_GREEN
    } else if health >= 0.5 {
        colors::YELLOW
    } else {
        colors::BRIGHT_RED
    };
    format!("{}[{}{}]{}",
        color,
        "#".repeat(filled),
        ".".repeat(empty),
        colors::RESET)
}

fn nutrient_color(score: f64) -> &'static str {
    if score >= 0.7 { colors::BRIGHT_GREEN }
    else if score >= 0.4 { colors::YELLOW }
    else { colors::BRIGHT_RED }
}

fn layer_color(layer: TreeLayer) -> &'static str {
    match layer {
        TreeLayer::Canopy => colors::BRIGHT_GREEN,
        TreeLayer::Roots => colors::YELLOW,
        TreeLayer::MicroOrganism => colors::BRIGHT_MAGENTA,
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

// ============================================================
// HTML Dashboard Generator
// ============================================================

/// Generate a self-contained HTML dashboard from a TreeSnapshot.
pub fn generate_html_dashboard(snapshot: &TreeSnapshot) -> String {
    let json_data = serde_json::to_string_pretty(snapshot)
        .unwrap_or_else(|_| "{}".to_string());

    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>cudaclaw Ramify Monitor</title>
<style>
  :root {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --cyan: #39d2c0;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    padding: 24px;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.4em;
    color: var(--cyan);
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 20px;
  }}
  h1 span {{ color: var(--text-dim); font-weight: normal; font-size: 0.7em; }}
  .stats-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
  }}
  .stat-card .label {{ color: var(--text-dim); font-size: 0.75em; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: bold; margin: 4px 0; }}
  .stat-card .unit {{ color: var(--text-dim); font-size: 0.8em; }}
  .stat-green .value {{ color: var(--green); }}
  .stat-cyan .value {{ color: var(--cyan); }}
  .stat-yellow .value {{ color: var(--yellow); }}
  .stat-red .value {{ color: var(--red); }}
  .stat-blue .value {{ color: var(--blue); }}
  .stat-purple .value {{ color: var(--purple); }}

  .layer {{
    margin-bottom: 24px;
  }}
  .layer-header {{
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
    font-weight: bold;
    font-size: 0.95em;
    letter-spacing: 0.05em;
  }}
  .layer-canopy .layer-header {{ background: #1a3a1a; color: var(--green); border: 1px solid #2a5a2a; }}
  .layer-roots .layer-header {{ background: #3a2a0a; color: var(--yellow); border: 1px solid #5a4a1a; }}
  .layer-micro .layer-header {{ background: #2a1a3a; color: var(--purple); border: 1px solid #4a2a5a; }}

  .layer-body {{
    background: var(--card);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 16px;
  }}
  .tree-node {{
    display: flex;
    align-items: center;
    padding: 6px 0;
    gap: 12px;
  }}
  .tree-node.depth-0 {{ font-size: 1.05em; }}
  .tree-node.depth-1 {{ padding-left: 24px; }}
  .tree-node.depth-2 {{ padding-left: 48px; color: var(--text-dim); font-size: 0.9em; }}
  .node-branch {{ color: var(--text-dim); min-width: 30px; }}
  .node-label {{ font-weight: bold; min-width: 180px; }}
  .node-value {{ color: var(--cyan); min-width: 120px; }}
  .node-health {{
    min-width: 120px;
    height: 12px;
    background: #21262d;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
  }}
  .node-health-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
  }}
  .health-good {{ background: var(--green); }}
  .health-warn {{ background: var(--yellow); }}
  .health-crit {{ background: var(--red); }}
  .node-detail {{
    color: var(--text-dim);
    font-size: 0.8em;
    padding-left: 66px;
    padding-bottom: 4px;
  }}

  .edges {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 24px;
  }}
  .edges h3 {{
    color: var(--text-dim);
    font-size: 0.85em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .edge {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    font-size: 0.85em;
    color: var(--text-dim);
  }}
  .edge-from {{ color: var(--blue); min-width: 180px; text-align: right; }}
  .edge-arrow {{ color: var(--text-dim); }}
  .edge-label {{ color: var(--yellow); min-width: 160px; }}
  .edge-to {{ color: var(--green); }}

  footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 0.75em;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<h1>cudaclaw Ramify Monitor <span>Ecosystem Tree Dashboard</span></h1>

<div class="stats-bar" id="stats-bar"></div>

<div id="layers"></div>

<div class="edges" id="edges"></div>

<footer>
  cudaclaw Ramify Monitor &mdash; Generated <span id="gen-time"></span>
  &mdash; <a href="#" onclick="location.reload()" style="color:var(--cyan)">Refresh</a>
</footer>

<script>
const DATA = {json_data};

function healthClass(h) {{
  if (h >= 0.8) return 'health-good';
  if (h >= 0.5) return 'health-warn';
  return 'health-crit';
}}

function fmtNum(n) {{
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
  return n.toFixed(1);
}}

function renderStats(stats) {{
  const bar = document.getElementById('stats-bar');
  const items = [
    {{ label: 'Throughput',  value: fmtNum(stats.throughput_ops), unit: 'ops/s',  cls: 'stat-green' }},
    {{ label: 'P99 Latency', value: stats.p99_latency_us.toFixed(2), unit: 'us', cls: 'stat-cyan' }},
    {{ label: 'Commands',    value: fmtNum(stats.total_commands), unit: 'total',  cls: 'stat-blue' }},
    {{ label: 'PTX Branches',value: stats.active_ptx_branches, unit: 'active',   cls: 'stat-yellow' }},
    {{ label: 'Bridges',     value: stats.active_bridges, unit: 'active',         cls: 'stat-purple' }},
    {{ label: 'Agents',      value: stats.total_agents, unit: 'registered',       cls: 'stat-blue' }},
    {{ label: 'Critical SMs',value: stats.critical_sms, unit: 'count',            cls: stats.critical_sms > 0 ? 'stat-red' : 'stat-green' }},
    {{ label: 'Nutrient',    value: (stats.avg_nutrient_score * 100).toFixed(0) + '%', unit: 'avg',
       cls: stats.avg_nutrient_score >= 0.7 ? 'stat-green' : stats.avg_nutrient_score >= 0.4 ? 'stat-yellow' : 'stat-red' }},
    {{ label: 'GPU Temp',    value: stats.gpu_temp_celsius != null ? stats.gpu_temp_celsius + 'C' : 'N/A', unit: '',
       cls: (stats.gpu_temp_celsius||0) >= 85 ? 'stat-red' : 'stat-green' }},
    {{ label: 'Throttle',    value: stats.thermal_throttle ? 'YES' : 'NO', unit: '',
       cls: stats.thermal_throttle ? 'stat-red' : 'stat-green' }},
  ];
  bar.innerHTML = items.map(i =>
    `<div class="stat-card ${{i.cls}}">
       <div class="label">${{i.label}}</div>
       <div class="value">${{i.value}}</div>
       <div class="unit">${{i.unit}}</div>
     </div>`
  ).join('');
}}

function nodesByLayer(layer) {{
  return DATA.nodes.filter(n => n.layer === layer);
}}

function renderLayer(containerId, layerName, layerClass, layerEnum) {{
  const container = document.getElementById(containerId);
  const nodes = nodesByLayer(layerEnum);
  if (nodes.length === 0) return;

  const roots = nodes.filter(n => n.id.endsWith('_root'));
  const root = roots[0] || nodes[0];

  let html = `<div class="layer ${{layerClass}}">
    <div class="layer-header">${{layerName}}</div>
    <div class="layer-body">`;

  html += renderTreeNode(root, 0);
  for (const cid of (root.children || [])) {{
    const child = DATA.nodes.find(n => n.id === cid);
    if (child) {{
      html += renderTreeNode(child, 1);
      for (const gcid of (child.children || [])) {{
        const gc = DATA.nodes.find(n => n.id === gcid);
        if (gc) html += renderTreeNode(gc, 2);
      }}
    }}
  }}

  html += '</div></div>';
  container.innerHTML += html;
}}

function renderTreeNode(node, depth) {{
  const branch = depth === 0 ? '===' : depth === 1 ? '|--' : '`--';
  const hpct = (node.health * 100).toFixed(0);
  const hcls = healthClass(node.health);
  let html = `<div class="tree-node depth-${{depth}}">
    <span class="node-branch">${{branch}}</span>
    <span class="node-label">${{node.label}}</span>
    <span class="node-value">${{fmtNum(node.value)}} ${{node.unit}}</span>
    <div class="node-health">
      <div class="node-health-fill ${{hcls}}" style="width:${{hpct}}%"></div>
    </div>
  </div>`;
  if (node.detail) {{
    html += `<div class="node-detail">${{node.detail}}</div>`;
  }}
  return html;
}}

function renderEdges() {{
  const container = document.getElementById('edges');
  if (!DATA.edges || DATA.edges.length === 0) {{ container.style.display = 'none'; return; }}
  let html = '<h3>Data Flow &amp; Impact Connections</h3>';
  for (const e of DATA.edges) {{
    html += `<div class="edge">
      <span class="edge-from">${{e.from}}</span>
      <span class="edge-arrow">--(${{fmtNum(e.weight)}})--&gt;</span>
      <span class="edge-label">${{e.label}}</span>
      <span class="edge-to">${{e.to}}</span>
    </div>`;
  }}
  container.innerHTML = html;
}}

// Render
renderStats(DATA.global_stats);
renderLayer('layers', 'CANOPY - Data Harvesting & Spreadsheet Results', 'layer-canopy', 'Canopy');
renderLayer('layers', 'ROOTS - PTX Branches & Hardware Throughput', 'layer-roots', 'Roots');
renderLayer('layers', 'MICRO-ORGANISMS - Agents & Resource Exhaust', 'layer-micro', 'MicroOrganism');
renderEdges();
document.getElementById('gen-time').textContent = new Date(DATA.timestamp * 1000).toLocaleString();
</script>
</body>
</html>"##)
}

// ============================================================
// JSON Export
// ============================================================

/// Export a TreeSnapshot as pretty-printed JSON.
pub fn export_json(snapshot: &TreeSnapshot) -> String {
    serde_json::to_string_pretty(snapshot)
        .unwrap_or_else(|_| "{}".to_string())
}

// ============================================================
// CLI Integration
// ============================================================

/// CLI actions for the `monitor` subcommand.
pub enum MonitorCliAction {
    /// Run a terminal demo with simulated data.
    Demo,
    /// Generate an HTML dashboard file.
    Html(String),
    /// Export a JSON snapshot file.
    Json(String),
}

/// Print help text for the `monitor` CLI subcommand.
pub fn print_monitor_help() {
    println!("cudaclaw monitor -- Ramify Monitor Tree Visualization Dashboard");
    println!();
    println!("USAGE:");
    println!("  cudaclaw monitor [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --demo               Run a terminal demo with simulated data");
    println!("  --html <FILE>        Generate an HTML dashboard to FILE");
    println!("  --json <FILE>        Export a JSON snapshot to FILE");
    println!("  --help, -h           Show this help message");
    println!();
    println!("DESCRIPTION:");
    println!("  Visualizes the cudaclaw ecosystem as a tree:");
    println!();
    println!("    CANOPY          High-level data harvesting and spreadsheet results.");
    println!("                    Aggregate throughput, cell edits, formula recalc,");
    println!("                    CRDT merges, result writeback success rates.");
    println!();
    println!("    ROOTS           Low-level PTX branches and hardware-level throughput.");
    println!("                    SM utilization, register pressure, shared memory");
    println!("                    budgets, active PTX variant info, PCIe RTT.");
    println!();
    println!("    MICRO-ORGANISMS Independent agents exhausting GPU resources.");
    println!("                    Each organism shows its resource consumption,");
    println!("                    'exhaust' (latency/heat), and impact on the system.");
    println!();
    println!("EXAMPLES:");
    println!("  cudaclaw monitor --demo");
    println!("  cudaclaw monitor --html ramify_dashboard.html");
    println!("  cudaclaw monitor --json snapshot.json");
}

/// Parse CLI arguments for the `monitor` subcommand.
pub fn parse_monitor_args(args: &[String]) -> Option<MonitorCliAction> {
    if args.is_empty() {
        return Some(MonitorCliAction::Demo);
    }

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--demo" => return Some(MonitorCliAction::Demo),
            "--html" => {
                i += 1;
                if i < args.len() {
                    return Some(MonitorCliAction::Html(args[i].clone()));
                } else {
                    println!("Error: --html requires a file path argument");
                    return None;
                }
            }
            "--json" => {
                i += 1;
                if i < args.len() {
                    return Some(MonitorCliAction::Json(args[i].clone()));
                } else {
                    println!("Error: --json requires a file path argument");
                    return None;
                }
            }
            "--help" | "-h" => return None,
            _ => {}
        }
        i += 1;
    }

    Some(MonitorCliAction::Demo)
}

/// Run the Ramify Monitor demo (terminal visualization).
pub fn run_demo() {
    let snapshot = TreeBuilder::build_demo();
    render_terminal(&snapshot);

    println!("\n{}Demo complete.{} Use {} to generate an HTML file.",
        colors::BOLD, colors::RESET,
        "--html <file>");
}

/// Run the HTML dashboard generation.
pub fn run_html_export(path: &str) {
    let snapshot = TreeBuilder::build_demo();
    let html = generate_html_dashboard(&snapshot);
    std::fs::write(path, &html).expect("Failed to write HTML dashboard");
    println!("Ramify Monitor HTML dashboard written to: {}", path);
    println!("Open in a browser to view the interactive tree visualization.");
}

/// Run the JSON snapshot export.
pub fn run_json_export(path: &str) {
    let snapshot = TreeBuilder::build_demo();
    let json = export_json(&snapshot);
    std::fs::write(path, &json).expect("Failed to write JSON snapshot");
    println!("Ramify Monitor JSON snapshot written to: {} ({} bytes)", path, json.len());
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_builder_demo() {
        let snapshot = TreeBuilder::build_demo();
        assert!(!snapshot.nodes.is_empty());
        assert!(!snapshot.edges.is_empty());
        assert!(snapshot.global_stats.throughput_ops > 0.0);
    }

    #[test]
    fn test_tree_layers_present() {
        let snapshot = TreeBuilder::build_demo();
        let has_canopy = snapshot.nodes.iter().any(|n| n.layer == TreeLayer::Canopy);
        let has_roots = snapshot.nodes.iter().any(|n| n.layer == TreeLayer::Roots);
        let has_micro = snapshot.nodes.iter().any(|n| n.layer == TreeLayer::MicroOrganism);
        assert!(has_canopy, "Missing Canopy layer");
        assert!(has_roots, "Missing Roots layer");
        assert!(has_micro, "Missing MicroOrganism layer");
    }

    #[test]
    fn test_canopy_nodes() {
        let snapshot = TreeBuilder::build_demo();
        let canopy: Vec<_> = snapshot.nodes.iter()
            .filter(|n| n.layer == TreeLayer::Canopy)
            .collect();
        assert!(canopy.len() >= 4, "Canopy should have root + 3 children");

        let root = canopy.iter().find(|n| n.id == "canopy_root").unwrap();
        assert_eq!(root.label, "Data Harvesting");
        assert!(root.value > 0.0);
    }

    #[test]
    fn test_roots_nodes() {
        let snapshot = TreeBuilder::build_demo();
        let roots: Vec<_> = snapshot.nodes.iter()
            .filter(|n| n.layer == TreeLayer::Roots)
            .collect();
        assert!(roots.len() >= 5, "Roots should have root + children + grandchildren");

        let ptx = roots.iter().find(|n| n.id == "roots_ptx").unwrap();
        assert_eq!(ptx.label, "PTX Branches");
    }

    #[test]
    fn test_micro_organism_nodes() {
        let snapshot = TreeBuilder::build_demo();
        let micro: Vec<_> = snapshot.nodes.iter()
            .filter(|n| n.layer == TreeLayer::MicroOrganism)
            .collect();
        assert!(micro.len() >= 4, "MicroOrganism should have root + agents");

        let greedy = micro.iter().find(|n| n.id == "micro_greedy").unwrap();
        assert!(greedy.health < 0.5, "Greedy agent should have poor health");
    }

    #[test]
    fn test_edges_connect_valid_nodes() {
        let snapshot = TreeBuilder::build_demo();
        let node_ids: Vec<&str> = snapshot.nodes.iter().map(|n| n.id.as_str()).collect();
        for edge in &snapshot.edges {
            assert!(node_ids.contains(&edge.from.as_str()),
                "Edge from '{}' references non-existent node", edge.from);
            assert!(node_ids.contains(&edge.to.as_str()),
                "Edge to '{}' references non-existent node", edge.to);
        }
    }

    #[test]
    fn test_global_stats() {
        let snapshot = TreeBuilder::build_demo();
        let s = &snapshot.global_stats;
        assert!(s.total_commands > 0);
        assert!(s.throughput_ops > 0.0);
        assert!(s.p99_latency_us > 0.0);
        assert!(s.total_agents > 0);
    }

    #[test]
    fn test_health_values_in_range() {
        let snapshot = TreeBuilder::build_demo();
        for node in &snapshot.nodes {
            assert!(node.health >= 0.0 && node.health <= 1.0,
                "Node '{}' has health {} outside [0,1]", node.id, node.health);
        }
    }

    #[test]
    fn test_html_generation() {
        let snapshot = TreeBuilder::build_demo();
        let html = generate_html_dashboard(&snapshot);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("cudaclaw Ramify Monitor"));
        assert!(html.contains("CANOPY"));
        assert!(html.contains("ROOTS"));
        assert!(html.contains("MICRO-ORGANISMS"));
        assert!(html.contains("Data Harvesting"));
        assert!(html.len() > 5000, "HTML should be substantial");
    }

    #[test]
    fn test_json_export() {
        let snapshot = TreeBuilder::build_demo();
        let json = export_json(&snapshot);
        assert!(json.contains("canopy_root"));
        assert!(json.contains("roots_root"));
        assert!(json.contains("micro_root"));
        // Verify it parses back
        let parsed: TreeSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.nodes.len(), snapshot.nodes.len());
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1500), "1.5K");
        assert_eq!(format_count(2_500_000), "2.50M");
        assert_eq!(format_count(1_200_000_000), "1.20B");
    }

    #[test]
    fn test_health_bar_str() {
        let bar = health_bar_str(0.95);
        assert!(bar.contains("##########"));

        let bar = health_bar_str(0.5);
        assert!(bar.contains("#####"));

        let bar = health_bar_str(0.1);
        assert!(bar.contains("#"));
    }

    #[test]
    fn test_cli_parsing() {
        let args: Vec<String> = vec!["--demo".into()];
        assert!(matches!(parse_monitor_args(&args), Some(MonitorCliAction::Demo)));

        let args: Vec<String> = vec!["--html".into(), "out.html".into()];
        match parse_monitor_args(&args) {
            Some(MonitorCliAction::Html(p)) => assert_eq!(p, "out.html"),
            _ => panic!("Expected Html action"),
        }

        let args: Vec<String> = vec!["--json".into(), "snap.json".into()];
        match parse_monitor_args(&args) {
            Some(MonitorCliAction::Json(p)) => assert_eq!(p, "snap.json"),
            _ => panic!("Expected Json action"),
        }

        let args: Vec<String> = vec!["--help".into()];
        assert!(parse_monitor_args(&args).is_none());
    }

    #[test]
    fn test_empty_args_defaults_to_demo() {
        let args: Vec<String> = vec![];
        assert!(matches!(parse_monitor_args(&args), Some(MonitorCliAction::Demo)));
    }
}
