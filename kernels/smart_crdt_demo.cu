// SmartCRDT Demo - Example usage of CRDT operations
// Demonstrates concurrent multi-warp updates and conflict resolution

#include "smart_crdt.cuh"

// ============================================================
// Demo Kernel: Concurrent Cell Updates
// ============================================================

// Simulate multiple agents (warps) updating cells concurrently
extern "C" __global__ void concurrent_update_demo(
    CrdtState* crdt,
    uint32_t updates_per_warp,
    uint64_t base_timestamp,
    uint32_t base_node_id
) {
    // Each warp acts as an independent agent
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id = threadIdx.x % 32;

    // Unique node ID for this warp
    const uint32_t node_id = base_node_id + warp_id;

    // Each warp performs multiple updates
    for (uint32_t i = 0; i < updates_per_warp; i++) {
        // Generate pseudo-random cell coordinates
        uint32_t row = (warp_id * updates_per_warp + i) % crdt->rows;
        uint32_t col = (lane_id * 3) % crdt->cols;

        // Generate value (simple counter)
        double value = warp_id * 100.0 + lane_id + i * 0.1;

        // Unique timestamp for this update
        uint64_t timestamp = base_timestamp + warp_id * updates_per_warp + i;

        // Perform merge operation (atomic, multi-warp safe)
        bool success = merge_op(crdt, row, col, value, timestamp, node_id);

        // Track results (optional)
        if (success && lane_id == 0) {
            atomicAdd(&crdt->global_version, 1);
        }
    }
}

// ============================================================
// Demo Kernel: Conflict Stress Test
// ============================================================

// Stress test: Multiple warps update the SAME cell to force conflicts
extern "C" __global__ void conflict_stress_test(
    CrdtState* crdt,
    uint32_t target_row,
    uint32_t target_col,
    uint32_t updates_per_thread
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t node_id = warp_id;

    // All warps target the same cell
    for (uint32_t i = 0; i < updates_per_thread; i++) {
        double value = warp_id * 10.0 + i;
        uint64_t timestamp = i;  // Same timestamp causes conflicts

        // Attempt merge (only highest node_id or timestamp wins)
        merge_op(crdt, target_row, target_col, value, timestamp, node_id);
    }
}

// ============================================================
// Demo Kernel: Coalesced Row Processing
// ============================================================

// Demonstrate coalesced access pattern for row-wise operations
extern "C" __global__ void row_aggregation_kernel(
    CrdtState* crdt,
    double* results  // Output: sum of each row
) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid_in_block = threadIdx.x;

    if (row >= crdt->rows) return;

    // Shared memory for warp-level reduction
    __shared__ double warp_sums[32];

    // Each thread processes one column (coalesced access)
    double thread_sum = 0.0;
    for (uint32_t col = tid_in_block; col < crdt->cols; col += blockDim.x) {
        uint32_t idx = row * crdt->cols + col;
        thread_sum += crdt->cells[idx].value;
    }

    // Warp-level reduction using shuffle
    const int lane_id = tid_in_block % 32;
    const int warp_id = tid_in_block / 32;

    // Shuffle down to sum across warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    // Lane 0 writes warp sum
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }

    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        double block_sum = (tid_in_block < (blockDim.x + 31) / 32)
            ? warp_sums[tid_in_block]
            : 0.0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }

        if (tid_in_block == 0) {
            results[row] = block_sum;
        }
    }
}

// ============================================================
// Demo Kernel: Cell Dependency Chain
// ============================================================

// Process dependent cells (e.g., formula dependencies)
extern "C" __global__ void dependency_chain_kernel(
    CrdtState* crdt,
    const uint32_t* dependency_rows,
    const uint32_t* dependency_cols,
    uint32_t chain_length
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= chain_length) return;

    // Start with first cell in chain
    uint32_t row = dependency_rows[idx];
    uint32_t col = dependency_cols[idx];

    if (!crdt->is_valid(row, col)) return;

    // Get initial value
    double value = crdt->get_cell(row, col).value;

    // Propagate through dependency chain
    for (uint32_t i = 1; i < chain_length && idx + i < chain_length; i++) {
        uint32_t next_row = dependency_rows[idx + i];
        uint32_t next_col = dependency_cols[idx + i];

        if (crdt->is_valid(next_row, next_col)) {
            // Simple formula: add previous value
            Cell& next_cell = crdt->get_cell(next_row, next_col);
            value += next_cell.value;
        }
    }

    // Update first cell with computed result
    merge_op(
        crdt,
        row,
        col,
        value,
        idx,  // timestamp
        0     // node_id (system update)
    );
}

// ============================================================
// Demo Kernel: SmartCRDT Diff Application
// ============================================================

// Apply a diff (set of changes) to the CRDT grid
extern "C" __global__ void apply_diff_kernel(
    CrdtState* crdt,
    const uint32_t* rows,
    const uint32_t* cols,
    const double* values,
    const uint64_t* timestamps,
    const uint32_t* node_ids,
    uint32_t diff_size,
    uint32_t* success_count  // Output counter
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= diff_size) return;

    bool success = merge_op(
        crdt,
        rows[idx],
        cols[idx],
        values[idx],
        timestamps[idx],
        node_ids[idx]
    );

    if (success) {
        atomicAdd(success_count, 1);
    }
}

// ============================================================
// Demo Kernel: Conflict Detection
// ============================================================

// Detect and mark cells with conflicting updates
extern "C" __global__ void detect_conflicts_kernel(
    CrdtState* crdt,
    uint32_t threshold_timestamp
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_cells = crdt->total_cells;

    if (idx >= total_cells) return;

    Cell& cell = crdt->cells[idx];

    // Mark cells with recent updates from multiple nodes as conflicts
    // (This is a simplified heuristic - real implementation would track node history)
    if (cell.timestamp > threshold_timestamp) {
        // In production, check if multiple nodes have updated this cell
        // For now, just mark as potential conflict
        if (cell.timestamp % 2 == 0) {  // Pseudo-random conflict marker
            atomicExch(reinterpret_cast<unsigned int*>(&cell.state), CELL_CONFLICT);
        }
    }
}

// ============================================================
// Demo Kernel: Batch Cell Copy
// ============================================================

// Copy a range of cells (optimized for coalesced access)
extern "C" __global__ void batch_copy_kernel(
    CrdtState* src,
    CrdtState* dst,
    uint32_t start_row,
    uint32_t end_row,
    uint32_t start_col,
    uint32_t end_col
) {
    const uint32_t row = start_row + blockIdx.x;
    const uint32_t col = start_col + threadIdx.x;

    if (row >= end_row || col >= end_col) return;
    if (!src->is_valid(row, col) || !dst->is_valid(row, col)) return;

    uint32_t idx = src->get_index(row, col);

    // Copy with CRDT semantics (preserve source metadata)
    Cell src_cell = src->cells[idx];
    merge_op(
        dst,
        row,
        col,
        src_cell.value,
        src_cell.timestamp,
        src_cell.node_id
    );
}

// ============================================================
// Demo Kernel: Spreadsheet Formula Evaluation
// ============================================================

// Evaluate a simple formula: A + B where A and B are cell references
extern "C" __global__ void formula_add_kernel(
    CrdtState* crdt,
    const uint32_t* a_rows,
    const uint32_t* a_cols,
    const uint32_t* b_rows,
    const uint32_t* b_cols,
    const uint32_t* result_rows,
    const uint32_t* result_cols,
    uint32_t formula_count
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= formula_count) return;

    uint32_t a_row = a_rows[idx];
    uint32_t a_col = a_cols[idx];
    uint32_t b_row = b_rows[idx];
    uint32_t b_col = b_cols[idx];
    uint32_t res_row = result_rows[idx];
    uint32_t res_col = result_cols[idx];

    if (!crdt->is_valid(a_row, a_col) ||
        !crdt->is_valid(b_row, b_col) ||
        !crdt->is_valid(res_row, res_col)) {
        return;
    }

    // Get operand values
    double a_value = crdt->get_cell(a_row, a_col).value;
    double b_value = crdt->get_cell(b_row, b_col).value;

    // Write result to destination cell
    merge_op(
        crdt,
        res_row,
        res_col,
        a_value + b_value,
        idx,  // timestamp based on formula index
        0     // system node_id
    );
}
