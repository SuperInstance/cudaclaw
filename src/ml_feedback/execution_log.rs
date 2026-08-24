// ============================================================
// Execution Log — Ring Buffer for Agent Execution Records
// ============================================================
//
// Stores execution records from cell agents in a fixed-capacity
// ring buffer. Oldest entries are evicted when full. The log is
// consumed by the SuccessAnalyzer to detect patterns and
// recommend DNA mutations.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Execution Entry
// ============================================================

/// A single execution record from a cell agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEntry {
    /// Agent identifier (e.g., "cell_3_5").
    pub agent_id: String,
    /// Muscle Fiber type used (e.g., "cell_update").
    pub fiber_type: String,
    /// Execution time in microseconds.
    pub execution_time_us: f64,
    /// Registers consumed by this execution.
    pub registers_used: u32,
    /// Shared memory consumed (bytes).
    pub shared_memory_used: u32,
    /// Memory coalescing ratio achieved (0.0-1.0).
    pub coalescing_ratio: f64,
    /// Warp occupancy achieved (0.0-1.0).
    pub warp_occupancy: f64,
    /// Whether the execution succeeded.
    pub success: bool,
    /// Epoch timestamp of this execution.
    pub timestamp_epoch: u64,
}

// ============================================================
// Execution Summary
// ============================================================

/// Aggregate statistics over the execution log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Total entries in the log.
    pub total_entries: usize,
    /// Number of entries per fiber type.
    pub entries_by_fiber: HashMap<String, usize>,
    /// Overall success rate (0.0-1.0).
    pub overall_success_rate: f64,
    /// Overall average latency in microseconds.
    pub overall_avg_latency_us: f64,
    /// Overall P99 latency in microseconds.
    pub overall_p99_latency_us: f64,
    /// Average coalescing ratio.
    pub avg_coalescing_ratio: f64,
    /// Average warp occupancy.
    pub avg_warp_occupancy: f64,
}

// ============================================================
// Execution Log
// ============================================================

/// Ring-buffer log for agent execution records.
pub struct ExecutionLog {
    /// Maximum capacity.
    capacity: usize,
    /// Stored entries.
    entries: Vec<ExecutionEntry>,
    /// Write position (for ring buffer wrap).
    write_pos: usize,
    /// Whether the buffer has wrapped.
    wrapped: bool,
}

impl ExecutionLog {
    /// Create a new log with the given capacity.
    pub fn new(capacity: usize) -> Self {
        ExecutionLog {
            capacity,
            entries: Vec::with_capacity(capacity.min(1024)),
            write_pos: 0,
            wrapped: false,
        }
    }

    /// Record a new execution entry.
    pub fn record(&mut self, entry: ExecutionEntry) {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            self.entries[self.write_pos] = entry;
            self.wrapped = true;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    /// Get all entries (in insertion order, oldest first).
    pub fn entries(&self) -> &[ExecutionEntry] {
        &self.entries
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the buffer has wrapped (oldest entries evicted).
    pub fn has_wrapped(&self) -> bool {
        self.wrapped
    }

    /// Get entries for a specific fiber type.
    pub fn entries_for_fiber(&self, fiber_type: &str) -> Vec<&ExecutionEntry> {
        self.entries
            .iter()
            .filter(|e| e.fiber_type == fiber_type)
            .collect()
    }

    /// Get all unique fiber types in the log.
    pub fn fiber_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .entries
            .iter()
            .map(|e| e.fiber_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        types.sort();
        types
    }

    /// Compute aggregate summary statistics.
    pub fn summary(&self) -> ExecutionSummary {
        if self.entries.is_empty() {
            return ExecutionSummary {
                total_entries: 0,
                entries_by_fiber: HashMap::new(),
                overall_success_rate: 1.0,
                overall_avg_latency_us: 0.0,
                overall_p99_latency_us: 0.0,
                avg_coalescing_ratio: 1.0,
                avg_warp_occupancy: 1.0,
            };
        }

        let total = self.entries.len();

        // Count by fiber type.
        let mut by_fiber: HashMap<String, usize> = HashMap::new();
        for e in &self.entries {
            *by_fiber.entry(e.fiber_type.clone()).or_insert(0) += 1;
        }

        // Success rate.
        let successes = self.entries.iter().filter(|e| e.success).count();
        let success_rate = successes as f64 / total as f64;

        // Latency stats.
        let total_latency: f64 = self.entries.iter().map(|e| e.execution_time_us).sum();
        let avg_latency = total_latency / total as f64;

        let mut latencies: Vec<f64> = self.entries.iter().map(|e| e.execution_time_us).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = (total as f64 * 0.99).ceil() as usize - 1;
        let p99 = latencies[p99_idx.min(total - 1)];

        // Coalescing and occupancy averages.
        let avg_coal: f64 = self.entries.iter().map(|e| e.coalescing_ratio).sum::<f64>() / total as f64;
        let avg_occ: f64 = self.entries.iter().map(|e| e.warp_occupancy).sum::<f64>() / total as f64;

        ExecutionSummary {
            total_entries: total,
            entries_by_fiber: by_fiber,
            overall_success_rate: success_rate,
            overall_avg_latency_us: avg_latency,
            overall_p99_latency_us: p99,
            avg_coalescing_ratio: avg_coal,
            avg_warp_occupancy: avg_occ,
        }
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.write_pos = 0;
        self.wrapped = false;
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(fiber: &str, latency: f64, success: bool) -> ExecutionEntry {
        ExecutionEntry {
            agent_id: "test".into(),
            fiber_type: fiber.into(),
            execution_time_us: latency,
            registers_used: 24,
            shared_memory_used: 2048,
            coalescing_ratio: 0.9,
            warp_occupancy: 0.7,
            success,
            timestamp_epoch: 1000,
        }
    }

    #[test]
    fn test_log_creation() {
        let log = ExecutionLog::new(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut log = ExecutionLog::new(100);
        log.record(make_entry("cell_update", 2.5, true));
        log.record(make_entry("crdt_merge", 4.0, false));

        assert_eq!(log.len(), 2);
        assert_eq!(log.entries_for_fiber("cell_update").len(), 1);
        assert_eq!(log.entries_for_fiber("crdt_merge").len(), 1);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut log = ExecutionLog::new(3);
        for i in 0..5 {
            log.record(make_entry("test", i as f64, true));
        }
        assert_eq!(log.len(), 3);
        assert!(log.has_wrapped());
    }

    #[test]
    fn test_summary() {
        let mut log = ExecutionLog::new(100);
        for i in 0..100 {
            let success = i % 10 != 0; // 90% success
            log.record(make_entry("cell_update", 2.0 + (i as f64 * 0.1), success));
        }

        let summary = log.summary();
        assert_eq!(summary.total_entries, 100);
        assert!((summary.overall_success_rate - 0.9).abs() < 0.01);
        assert!(summary.overall_p99_latency_us > 10.0);
    }

    #[test]
    fn test_fiber_types() {
        let mut log = ExecutionLog::new(100);
        log.record(make_entry("cell_update", 1.0, true));
        log.record(make_entry("crdt_merge", 2.0, true));
        log.record(make_entry("cell_update", 3.0, true));

        let types = log.fiber_types();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&"cell_update".to_string()));
        assert!(types.contains(&"crdt_merge".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut log = ExecutionLog::new(100);
        log.record(make_entry("test", 1.0, true));
        assert_eq!(log.len(), 1);
        log.clear();
        assert!(log.is_empty());
    }
}
