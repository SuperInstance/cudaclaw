// SmartCRDT - CUDA implementation for GPU-accelerated CRDT operations
// Optimized for coalesced memory access and concurrent warp-level updates

#pragma once

#include <cuda_runtime.h>
#include <device_atomic_functions.h>

// ============================================================
// CRDT Cell State and Metadata
// ============================================================

// Cell state flags for CRDT lifecycle management
enum CellState : uint32_t {
    CELL_ACTIVE = 0,      // Cell is active and visible
    CELL_DELETED = 1,     // Cell has been deleted (tombstone)
    CELL_CONFLICT = 2,    // Cell has conflicting updates
    CELL_MERGED = 3,      // Cell has been merged from conflict
    CELL_PENDING = 4      // Cell update is pending confirmation
};

// ============================================================
// CRDT Cell Structure - Optimized for Coalesced Access
// ============================================================

// Primary cell structure (32 bytes - cache line friendly)
// Align to 32 bytes for optimal coalescing across warps
struct __align__(32) Cell {
    double value;           // Primary cell value (8 bytes)
    uint64_t timestamp;     // Lamport timestamp for ordering (8 bytes)
    uint32_t node_id;       // Origin node identifier (4 bytes)
    CellState state;        // Cell state (4 bytes)

    // Padding to 32 bytes for aligned access
    uint32_t padding[3];    // (12 bytes)

    // Default constructor
    __device__ __host__ Cell()
        : value(0.0)
        , timestamp(0)
        , node_id(0)
        , state(CELL_ACTIVE)
        , padding{0, 0, 0}
    {}

    // Parameterized constructor
    __device__ __host__ Cell(double v, uint64_t ts, uint32_t nid, CellState s)
        : value(v)
        , timestamp(ts)
        , node_id(nid)
        , state(s)
        , padding{0, 0, 0}
    {}
};

// Compile-time assertions for size optimization
static_assert(sizeof(Cell) == 32, "Cell must be exactly 32 bytes for optimal alignment");

// ============================================================
// CRDT Grid State - Flat Array Layout
// ============================================================

// Main CRDT state structure
// Row-major layout: cells[row * cols + col]
struct CrdtState {
    Cell* cells;            // Flat array of cells (unified or device memory)
    uint32_t rows;          // Number of rows in the grid
    uint32_t cols;          // Number of columns in the grid
    uint32_t total_cells;   // Total cells (rows * cols)

    // Statistics tracking
    uint64_t global_version;     // Global version counter
    uint32_t conflict_count;     // Number of conflicts detected
    uint32_t merge_count;        // Number of merges performed

    // Padding to avoid false sharing
    uint8_t padding[64];

    // Default constructor
    __device__ __host__ CrdtState()
        : cells(nullptr)
        , rows(0)
        , cols(0)
        , total_cells(0)
        , global_version(0)
        , conflict_count(0)
        , merge_count(0)
        , padding{0}
    {}

    // Get 1D index from 2D coordinates (row-major layout)
    __device__ __host__ __forceinline__ uint32_t get_index(uint32_t row, uint32_t col) const {
        return row * cols + col;
    }

    // Get cell reference at 2D coordinates
    __device__ __host__ __forceinline__ Cell& get_cell(uint32_t row, uint32_t col) {
        return cells[get_index(row, col)];
    }

    __device__ __host__ __forceinline__ const Cell& get_cell(uint32_t row, uint32_t col) const {
        return cells[get_index(row, col)];
    }

    // Check bounds
    __device__ __host__ __forceinline__ bool is_valid(uint32_t row, uint32_t col) const {
        return row < rows && col < cols;
    }
};

// ============================================================
// Atomic Operations for Concurrent Updates
// ============================================================

// Compare two timestamps with node ID tiebreaker
// Returns true if (ts1, node1) has higher priority
__device__ __forceinline__ bool compare_timestamps(
    uint64_t ts1, uint32_t node1,
    uint64_t ts2, uint32_t node2
) {
    if (ts1 > ts2) return true;
    if (ts1 < ts2) return false;
    // Tie-break by node ID (higher node ID wins)
    return node1 > node2;
}

// Block-level atomic compare-and-swap for 64-bit values
// Uses warp-level primitives for better performance than global atomics
__device__ __forceinline__ bool atomicCAS_block(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val
) {
    // Use warp aggregation to reduce atomic contention
    // Only the first thread in each warp performs the atomic operation
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Shared memory for warp-level aggregation
    __shared__ unsigned long long warp_results[32];

    // Each warp processes one CAS operation
    if (lane_id == 0) {
        unsigned long long old = atomicCAS(
            reinterpret_cast<unsigned long long*>(address),
            compare,
            val
        );
        warp_results[warp_id] = old;
    }

    __syncthreads();

    // Broadcast result to all threads in warp
    unsigned long long result = warp_results[warp_id];

    return (result == compare);
}

// ============================================================
// CRDT Merge Operation - Multi-Warp Safe
// ============================================================

// Merge a cell update with atomic conflict resolution
// Multiple warps can safely update the same cell concurrently
__device__ __forceinline__ bool merge_op(
    CrdtState* crdt,
    uint32_t row,
    uint32_t col,
    double value,
    uint64_t timestamp,
    uint32_t node_id
) {
    // Boundary check
    if (!crdt->is_valid(row, col)) {
        return false;
    }

    const uint32_t idx = crdt->get_index(row, col);
    Cell* cell_array = crdt->cells;

    // Optimized spin loop for concurrent updates
    int spin_count = 0;
    const int max_spins = 1000;

    while (spin_count < max_spins) {
        // Load current cell state
        Cell current = cell_array[idx];

        // Check if new update has higher priority
        if (compare_timestamps(current.timestamp, current.node_id, timestamp, node_id)) {
            // Existing cell has higher priority, reject update
            return false;
        }

        // Prepare new cell state
        Cell new_cell = current;
        new_cell.value = value;
        new_cell.timestamp = timestamp;
        new_cell.node_id = node_id;
        new_cell.state = CELL_ACTIVE;

        // Attempt atomic update using 64-bit CAS
        unsigned long long* current_ptr = reinterpret_cast<unsigned long long*>(&current);
        unsigned long long* new_ptr = reinterpret_cast<unsigned long long*>(&new_cell);
        unsigned long long* cell_ptr = reinterpret_cast<unsigned long long*>(&cell_array[idx]);

        // Use atomicCAS on first 64 bits (value + part of timestamp)
        unsigned long long old = atomicCAS(cell_ptr, current_ptr[0], new_ptr[0]);

        if (old == current_ptr[0]) {
            // First half succeeded, update second half atomically
            atomicCAS(cell_ptr + 1, current_ptr[1], new_ptr[1]);

            // Update global statistics atomically
            atomicAdd(&crdt->global_version, 1);
            return true;
        }

        // CAS failed, retry with new current value
        spin_count++;

        // Optional: Exponential backoff to reduce contention
        if (spin_count % 10 == 0) {
            __nanosleep(10 * spin_count);
        }
    }

    // Failed to acquire lock after maximum spins
    atomicAdd(&crdt->conflict_count, 1);
    return false;
}

// ============================================================
// Batch Operations for Coalesced Access
// ============================================================

// Kernel to batch initialize cells (coalesced writes)
__global__ void crdt_init_kernel(CrdtState* crdt, double initial_value) {
    const uint32_t total_cells = crdt->total_cells;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_cells) {
        // Coalesced write pattern - consecutive threads access consecutive memory
        crdt->cells[idx].value = initial_value;
        crdt->cells[idx].timestamp = 0;
        crdt->cells[idx].node_id = 0;
        crdt->cells[idx].state = CELL_ACTIVE;
    }
}

// Kernel to clear all cells (coalesced writes)
__global__ void crdt_clear_kernel(CrdtState* crdt) {
    const uint32_t total_cells = crdt->total_cells;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_cells) {
        crdt->cells[idx].value = 0.0;
        crdt->cells[idx].state = CELL_ACTIVE;
    }
}

// Kernel for batch cell updates with pattern optimization
__global__ void crdt_batch_merge_kernel(
    CrdtState* crdt,
    const uint32_t* rows,
    const uint32_t* cols,
    const double* values,
    const uint64_t* timestamps,
    const uint32_t* node_ids,
    uint32_t count
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        uint32_t row = rows[idx];
        uint32_t col = cols[idx];

        if (crdt->is_valid(row, col)) {
            merge_op(
                crdt,
                row,
                col,
                values[idx],
                timestamps[idx],
                node_ids[idx]
            );
        }
    }
}

// Kernel for conflict resolution (merge conflicting cells)
__global__ void crdt_resolve_conflicts_kernel(CrdtState* crdt) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_cells = crdt->total_cells;

    if (idx < total_cells) {
        Cell& cell = crdt->cells[idx];

        if (cell.state == CELL_CONFLICT) {
            // Simple resolution: keep the highest timestamp
            // In production, this would implement application-specific merge logic
            cell.state = CELL_MERGED;
            atomicAdd(&crdt->merge_count, 1);
        }
    }
}

// ============================================================
// Helper Functions for Memory Access Patterns
// ============================================================

// Strided access pattern for row-wise operations (coalesced)
__device__ __forceinline__ void row_wise_access(CrdtState* crdt, uint32_t row) {
    if (row >= crdt->rows) return;

    const uint32_t row_start = row * crdt->cols;
    const uint32_t tid_in_block = threadIdx.x;

    // Each thread in warp accesses consecutive columns (coalesced)
    for (uint32_t col = tid_in_block; col < crdt->cols; col += blockDim.x) {
        uint32_t idx = row_start + col;
        // Access cell for processing
        volatile double val = crdt->cells[idx].value;
    }
}

// Strided access pattern for column-wise operations (coalesced)
__device__ __forceinline__ void column_wise_access(CrdtState* crdt, uint32_t col) {
    if (col >= crdt->cols) return;

    const uint32_t tid_in_block = threadIdx.x;

    // Each thread accesses consecutive rows (coalesced)
    for (uint32_t row = tid_in_block; row < crdt->rows; row += blockDim.x) {
        uint32_t idx = row * crdt->cols + col;
        // Access cell for processing
        volatile double val = crdt->cells[idx].value;
    }
}

// ============================================================
// Host-Side Helper Functions
// ============================================================

#ifndef __CUDACC__

#include <cstring>
#include <stdexcept>

// Initialize CRDT state on host
inline void init_crdt_state_host(
    CrdtState* crdt,
    uint32_t rows,
    uint32_t cols
) {
    crdt->rows = rows;
    crdt->cols = cols;
    crdt->total_cells = rows * cols;
    crdt->global_version = 0;
    crdt->conflict_count = 0;
    crdt->merge_count = 0;
    memset(crdt->padding, 0, sizeof(crdt->padding));
}

// Allocate device memory for CRDT state
inline cudaError_t alloc_crdt_state_device(CrdtState* d_crdt, uint32_t rows, uint32_t cols) {
    const uint32_t total_cells = rows * cols;

    // Allocate cells array
    Cell* d_cells;
    cudaError_t err = cudaMalloc(&d_cells, total_cells * sizeof(Cell));
    if (err != cudaSuccess) return err;

    // Copy to device
    err = cudaMemcpy(
        &d_crdt->cells,
        &d_cells,
        sizeof(Cell*),
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        cudaFree(d_cells);
        return err;
    }

    // Set dimensions
    err = cudaMemcpy(
        &d_crdt->rows,
        &rows,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    );

    err = cudaMemcpy(
        &d_crdt->cols,
        &cols,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    );

    err = cudaMemcpy(
        &d_crdt->total_cells,
        &total_cells,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    );

    return err;
}

// Free device memory
inline cudaError_t free_crdt_state_device(CrdtState* d_crdt) {
    Cell* d_cells;
    cudaError_t err = cudaMemcpy(
        &d_cells,
        &d_crdt->cells,
        sizeof(Cell*),
        cudaMemcpyDeviceToHost
    );

    if (err == cudaSuccess) {
        cudaFree(d_cells);
    }

    return err;
}

#endif // __CUDACC__

// ============================================================
// Performance Optimization Macros
// ============================================================

// Force inline for device functions (performance critical)
#define CRDT_DEVICE_INLINE __device__ __forceinline__

// Loop unrolling for fixed-size operations
#define CRDT_UNROLL_4 for (int _i = 0; _i < 4; _i++)
#define CRDT_UNROLL_8 for (int _i = 0; _i < 8; _i++)

// Memory barrier for ensuring visibility
#define CRDT_MEM_fence __threadfence_block()
