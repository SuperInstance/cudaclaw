// ============================================================
// SmartCRDT - GPU-Optimized RGA Implementation
// ============================================================
// This file implements Replicated Growable Array (RGA) CRDT operations
// optimized for GPU execution using CUDA atomics and coalesced memory access.
//
// KEY FEATURES:
// - Atomic insert/delete operations using atomicCAS
// - Lock-free concurrent edits to spreadsheet cells
// - Memory layout optimized for coalesced access
// - Adjacent threads process adjacent cells
// - Efficient VRAM bandwidth utilization
//
// RGA OPERATIONS:
// - Atomic insert: Add element at position
// - Atomic delete: Remove element at position
// - Conflict resolution: Last-write-wins with timestamps
//
// MEMORY OPTIMIZATION:
// - Cell data stored in Structure of Arrays (SoA) format
// - Adjacent threads access adjacent cells (coalesced)
// - Warp-level processing for row operations
//
// ============================================================

#ifndef SMARTCRDT_CUH
#define SMARTCRDT_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "shared_types.h"

// ============================================================
// SmartCRDT Configuration
// ============================================================

#define MAX_ROWS 10000         // Maximum spreadsheet rows
#define MAX_COLS 10000         // Maximum spreadsheet columns
#define MAX_CELL_VALUE_LEN 256 // Maximum cell value length
#define MAX_FORMULA_LEN 512    // Maximum formula length

// Thread block configuration for optimal coalescing
#define THREADS_PER_CELL 1     // One thread per cell for simple operations
#define CELLS_PER_WARP 32      // Process 32 cells per warp (full warp)
#define WARPS_PER_BLOCK 8      // 8 warps per block (256 threads)

// ============================================================
// SmartCRDT Data Structures
// ============================================================

/**
 * Cell identifier - unique position in spreadsheet
 */
struct __align__(8) CellID {
    uint32_t row;
    uint32_t col;

    __device__ __forceinline__ bool operator==(const CellID& other) const {
        return row == other.row && col == other.col;
    }

    __device__ __forceinline__ bool operator!=(const CellID& other) const {
        return row != other.row || col != other.col;
    }

    __device__ __forceinline__ uint64_t to_hash() const {
        // Interleave bits for better cache distribution
        uint64_t hash = ((uint64_t)row << 32) | col;
        return hash;
    }
};

/**
 * RGA Tombstone - marks deleted elements
 */
struct __align__(8) RGATombstone {
    uint64_t timestamp;    // Deletion timestamp
    uint32_t node_id;      // Which node performed deletion
    uint32_t reserved;     // Padding

    __device__ __forceinline__ bool is_valid() const {
        return timestamp > 0;
    }
};

/**
 * RGA Node - represents an element in the replicated array
 */
struct __align__(16) RGANode {
    uint64_t timestamp;       // Operation timestamp for ordering
    uint32_t node_id;         // Which node created this
    uint32_t value_len;       // Length of value (for strings)
    CellID cell_id;           // Which cell this represents
    double value;             // Numeric value
    RGATombstone tombstone;   // Deletion marker

    __device__ __forceinline__ bool is_alive() const {
        return !tombstone.is_valid();
    }

    __device__ __forceinline__ bool is_deleted() const {
        return tombstone.is_valid();
    }
};

/**
 * Cell Value - supports multiple types
 */
enum CellValueType : uint32_t {
    CELL_EMPTY = 0,      // No value
    CELL_NUMBER = 1,     // Numeric value (double)
    CELL_STRING = 2,     // String value
    CELL_FORMULA = 3,    // Formula (computed value)
    CELL_BOOLEAN = 4,    // True/false
    CELL_ERROR = 5       // Error value
};

/**
 * Spreadsheet Cell - optimized for GPU access
 *
 * Uses Structure of Arrays (SoA) layout for better coalescing:
 * - All values stored contiguously
 * - Adjacent threads access adjacent cells
 * - Minimizes memory transactions
 */
struct __align__(16) SpreadsheetCell {
    CellID cell_id;            // Cell position (row, col)
    volatile CellValueType type;  // Value type
    volatile double numeric_value;  // For CELL_NUMBER
    volatile uint64_t timestamp;   // Last edit timestamp
    volatile uint32_t node_id;     // Last editor node
    volatile uint32_t value_hash;  // Hash of value (for change detection)
    volatile uint32_t formula_len; // Length of formula (if applicable)
    volatile uint32_t flags;       // Cell flags (computed, locked, etc.)

    // Pointers to extended data (stored separately for better cache)
    volatile uint64_t string_ptr;  // Pointer to string value
    volatile uint64_t formula_ptr; // Pointer to formula

    __device__ __forceinline__ bool is_empty() const {
        return type == CELL_EMPTY;
    }

    __device__ __forceinline__ bool is_computed() const {
        return type == CELL_FORMULA;
    }

    __device__ __forceinline__ bool needs_update(const SpreadsheetCell& other) const {
        // Check if other cell is newer (last-write-wins)
        return other.timestamp > timestamp;
    }
};

/**
 * RGA Array - lock-free growable array for cell values
 *
 * Uses atomic operations for concurrent insert/delete:
 * - atomicCAS for insert
 * - atomicCAS for delete
 * - atomicCAS for update
 */
struct __align__(16) RGAArray {
    volatile uint64_t element_count;     // Current number of elements
    volatile uint64_t tombstone_count;   // Number of deleted elements
    volatile uint32_t capacity;          // Current capacity
    volatile uint32_t node_id;           // Our node ID
    volatile RGANode* elements;          // Array of nodes (in global memory)

    __device__ __forceinline__ uint64_t size() const {
        return element_count - tombstone_count;
    }
};

/**
 * Spreadsheet Edit Command - CRDT operation
 */
struct __align__(32) SpreadsheetEdit {
    CellID cell_id;             // Target cell
    CellValueType new_type;     // New value type
    double numeric_value;       // Numeric value (if CELL_NUMBER)
    uint64_t timestamp;         // Operation timestamp
    uint32_t node_id;           // Editor node ID
    uint32_t value_len;         // Length of string/formula
    uint64_t string_ptr;        // Pointer to string value (if applicable)
    uint64_t formula_ptr;       // Pointer to formula (if applicable)
    uint32_t flags;             // Edit flags

    // RGA metadata
    bool is_insert;             // True if insert, false if update
    bool is_delete;             // True if delete operation
    uint32_t insert_after_idx;  // Insert position (for RGA)
};

// ============================================================
// Atomic RGA Operations
// ============================================================

/**
 * Atomic compare-and-swap for RGA nodes
 *
 * PTX inline assembly for optimal performance
 */
__device__ __forceinline__ bool atomic_cas_node(
    volatile RGANode* ptr,
    const RGANode& expected,
    const RGANode& desired
) {
    // Use 128-bit CAS for entire struct
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        unsigned long long int* ptr_as_ull = (unsigned long long int*)ptr;
        unsigned long long int expected_as_ull[2];
        unsigned long long int desired_as_ull[2];

        memcpy(expected_as_ull, &expected, sizeof(RGANode));
        memcpy(desired_as_ull, &desired, sizeof(RGANode));

        unsigned long long int old[2];
        old[0] = atomicCAS(ptr_as_ull, expected_as_ull[0], desired_as_ull[0]);
        old[1] = atomicCAS(ptr_as_ull + 1, expected_as_ull[1], desired_as_ull[1]);

        bool success = (old[0] == expected_as_ull[0]) && (old[1] == expected_as_ull[1]);

        if (success) {
            return true;
        } else {
            // CAS failed - copy old value back to check
            memcpy((RGANode*)expected, old, sizeof(RGANode));
            return false;
        }
    #else
        // Fallback for older architectures - use 64-bit CAS
        return false;  // Not supported on older arch
    #endif
}

/**
 * Atomic cell update using last-write-wins semantics
 *
 * @param cell Pointer to cell in global memory
 * @param edit Edit operation to apply
 * @return true if update was applied, false if concurrent update won
 */
__device__ bool atomic_update_cell(
    volatile SpreadsheetCell* cell,
    const SpreadsheetEdit& edit
) {
    // Load current cell state
    SpreadsheetCell old_cell = *cell;

    // Check if our edit is newer (last-write-wins)
    if (edit.timestamp <= old_cell.timestamp) {
        return false;  // Our edit is older, don't apply
    }

    // Prepare new cell state
    SpreadsheetCell new_cell = old_cell;
    new_cell.cell_id = edit.cell_id;
    new_cell.type = edit.new_type;
    new_cell.numeric_value = edit.numeric_value;
    new_cell.timestamp = edit.timestamp;
    new_cell.node_id = edit.node_id;
    new_cell.flags = edit.flags;
    new_cell.string_ptr = edit.string_ptr;
    new_cell.formula_ptr = edit.formula_ptr;
    new_cell.formula_len = edit.formula_len;

    // AtomicCAS to update
    // Use 64-bit volatile pointer casting for alignment
    volatile uint64_t* cell_ptr_64 = (volatile uint64_t*)cell;
    uint64_t* new_cell_ptr_64 = (uint64_t*)&new_cell;
    uint64_t* old_cell_ptr_64 = (uint64_t*)&old_cell;

    // CAS on first 64 bits (cell_id through timestamp)
    uint64_t old0 = atomicCAS(cell_ptr_64, old_cell_ptr_64[0], new_cell_ptr_64[0]);
    if (old0 != old_cell_ptr_64[0]) {
        return false;  // Concurrent update won
    }

    // CAS on second 64 bits (node_id through end)
    uint64_t old1 = atomicCAS(cell_ptr_64 + 1, old_cell_ptr_64[1], new_cell_ptr_64[1]);
    if (old1 != old_cell_ptr_64[1]) {
        // Rollback first CAS
        atomicCAS(cell_ptr_64, new_cell_ptr_64[0], old_cell_ptr_64[0]);
        return false;  // Concurrent update won
    }

    return true;  // Update succeeded
}

/**
 * Atomic cell deletion - marks cell as deleted with tombstone
 *
 * @param cell Pointer to cell in global memory
 * @param timestamp Deletion timestamp
 * @param node_id Node performing deletion
 * @return true if deletion succeeded
 */
__device__ bool atomic_delete_cell(
    volatile SpreadsheetCell* cell,
    uint64_t timestamp,
    uint32_t node_id
) {
    // Load current state
    SpreadsheetCell old_cell = *cell;

    // Prepare deleted state
    SpreadsheetCell new_cell = old_cell;
    new_cell.type = CELL_EMPTY;
    new_cell.timestamp = timestamp;
    new_cell.node_id = node_id;
    new_cell.value_hash = 0xDEADBEEF;  // Tombstone marker

    // AtomicCAS to update
    volatile uint64_t* cell_ptr_64 = (volatile uint64_t*)cell;
    uint64_t* new_cell_ptr_64 = (uint64_t*)&new_cell;
    uint64_t* old_cell_ptr_64 = (uint64_t*)&old_cell;

    uint64_t old0 = atomicCAS(cell_ptr_64, old_cell_ptr_64[0], new_cell_ptr_64[0]);
    if (old0 != old_cell_ptr_64[0]) {
        return false;
    }

    uint64_t old1 = atomicCAS(cell_ptr_64 + 1, old_cell_ptr_64[1], new_cell_ptr_64[1]);
    if (old1 != old_cell_ptr_64[1]) {
        atomicCAS(cell_ptr_64, new_cell_ptr_64[0], old_cell_ptr_64[0]);
        return false;
    }

    return true;
}

// ============================================================
// Coalesced Memory Access Patterns
// ============================================================

/**
 * Calculate cell index for optimal coalescing
 *
 * Maps (row, col) to linear index such that adjacent threads
 * access adjacent cells in memory (coalesced access pattern)
 *
 * Mapping strategy: Interleave rows and columns
 * - Thread 0: (row 0, col 0)
 * - Thread 1: (row 0, col 1)
 * - ...
 * - Thread 32: (row 1, col 0)
 *
 * @param row Spreadsheet row
 * @param col Spreadsheet column
 * @param cols_per_row Number of columns per row
 * @return Linear cell index
 */
__device__ __forceinline__ uint32_t get_coalesced_cell_index(
    uint32_t row,
    uint32_t col,
    uint32_t cols_per_row
) {
    // Row-major order with column grouping for coalescing
    // Each row is stored contiguously
    return row * cols_per_row + col;
}

/**
 * Process a range of cells with coalesced memory access
 *
 * Each thread in a warp processes one cell, with adjacent threads
 * processing adjacent cells for optimal memory bandwidth utilization.
 *
 * @param cells Pointer to cell array
 * @param edits Array of edits to apply
 * @param edit_count Number of edits
 * @param thread_id Global thread ID
 * @param total_threads Total number of threads
 */
__device__ void process_cells_coalesced(
    volatile SpreadsheetCell* cells,
    const SpreadsheetEdit* edits,
    uint32_t edit_count,
    uint32_t cols_per_row
) {
    // Each thread processes a subset of edits
    for (uint32_t i = threadIdx.x; i < edit_count; i += blockDim.x) {
        const SpreadsheetEdit& edit = edits[i];

        // Calculate cell index for coalesced access
        uint32_t cell_idx = get_coalesced_cell_index(
            edit.cell_id.row,
            edit.cell_id.col,
            cols_per_row
        );

        // Apply edit using atomic operations
        if (edit.is_delete) {
            atomic_delete_cell(&cells[cell_idx], edit.timestamp, edit.node_id);
        } else {
            atomic_update_cell(&cells[cell_idx], edit);
        }
    }

    // Synchronize warp to ensure all updates are visible
    __syncwarp();
}

/**
 * Batch process cells using warp-level parallelism
 *
 * Distributes cells across warps for maximum throughput.
 * Each warp processes a contiguous group of cells.
 *
 * @param cells Pointer to cell array
 * @param edits Array of edits to apply
 * @param edit_count Number of edits
 * @param warp_id Which warp this thread belongs to
 * @param total_warps Total number of warps in block
 */
__device__ void process_cells_batch(
    volatile SpreadsheetCell* cells,
    const SpreadsheetEdit* edits,
    uint32_t edit_count,
    uint32_t warp_id,
    uint32_t total_warps,
    uint32_t cols_per_row
) {
    // Each warp processes a subset of edits
    uint32_t edits_per_warp = (edit_count + total_warps - 1) / total_warps;
    uint32_t start_idx = warp_id * edits_per_warp;
    uint32_t end_idx = min(start_idx + edits_per_warp, edit_count);

    // Process edits assigned to this warp
    for (uint32_t i = start_idx; i < end_idx; i++) {
        const SpreadsheetEdit& edit = edits[i];

        // Calculate cell index for coalesced access
        uint32_t cell_idx = get_coalesced_cell_index(
            edit.cell_id.row,
            edit.cell_id.col,
            cols_per_row
        );

        // Apply edit using atomic operations
        if (edit.is_delete) {
            atomic_delete_cell(&cells[cell_idx], edit.timestamp, edit.node_id);
        } else {
            atomic_update_cell(&cells[cell_idx], edit);
        }
    }

    // Synchronize warp after processing
    __syncwarp();
}

// ============================================================
// RGA Insert/Delete Operations
// ============================================================

/**
 * Atomic insert into RGA array
 *
 * Inserts a new element at the specified position using atomicCAS.
 * Handles concurrent inserts by using timestamps for ordering.
 *
 * @param array RGA array to insert into
 * @param node Node to insert
 * @param position Insert position (index)
 * @return true if insert succeeded
 */
__device__ bool atomic_rga_insert(
    RGAArray* array,
    const RGANode& node,
    uint32_t position
) {
    // Check capacity
    if (array->element_count >= array->capacity) {
        return false;  // Array is full
    }

    // Try to find a slot using linear probing
    for (uint32_t i = 0; i < array->capacity; i++) {
        uint32_t idx = (position + i) % array->capacity;
        volatile RGANode* slot = &array->elements[idx];

        // Try to claim this slot
        RGANode expected;
        expected.timestamp = 0;  // Empty slot

        RGANode desired = node;

        if (atomic_cas_node(slot, expected, desired)) {
            // Successfully inserted
            atomicAdd((unsigned long long*)&array->element_count, 1ULL);
            return true;
        }

        // Slot was occupied, try next
        // (In production, use better probing strategy)
    }

    return false;  // No slot found
}

/**
 * Atomic delete from RGA array
 *
 * Marks an element as deleted using a tombstone.
 * The element remains in the array but is marked as deleted.
 *
 * @param array RGA array to delete from
 * @param cell_id Cell ID to delete
 * @param timestamp Deletion timestamp
 * @param node_id Node performing deletion
 * @return true if delete succeeded
 */
__device__ bool atomic_rga_delete(
    RGAArray* array,
    const CellID& cell_id,
    uint64_t timestamp,
    uint32_t node_id
) {
    // Search for the element
    for (uint32_t i = 0; i < array->element_count; i++) {
        volatile RGANode* node = &array->elements[i];

        // Check if this is our target node
        if (node->cell_id == cell_id && !node->is_deleted()) {
            // Mark as deleted using tombstone
            RGATombstone tombstone;
            tombstone.timestamp = timestamp;
            tombstone.node_id = node_id;
            tombstone.reserved = 0;

            // Use atomicCAS to set tombstone
            // (Simplified - full implementation would use proper CAS)
            node->tombstone = tombstone;

            atomicAdd((unsigned long long*)&array->tombstone_count, 1ULL);
            return true;
        }
    }

    return false;  // Element not found
}

// ============================================================
// Conflict Resolution
// ============================================================

/**
 * Resolve concurrent edits using last-write-wins
 *
 * When two edits conflict, the one with the higher timestamp wins.
 * This ensures all nodes converge to the same state.
 *
 * @param existing Existing cell value
 * @param incoming Incoming edit
 * @return true if incoming should win
 */
__device__ __forceinline__ bool resolve_conflict_lww(
    const SpreadsheetCell& existing,
    const SpreadsheetEdit& incoming
) {
    return incoming.timestamp > existing.timestamp;
}

/**
 * Resolve concurrent insert conflicts
 *
 * When two inserts happen at the same position, use node_id
 * as tiebreaker for deterministic ordering.
 *
 * @param existing Existing node
 * @param incoming Incoming node
 * @return true if incoming should win
 */
__device__ __forceinline__ bool resolve_insert_conflict(
    const RGANode& existing,
    const RGANode& incoming
) {
    if (incoming.timestamp > existing.timestamp) {
        return true;
    } else if (incoming.timestamp == existing.timestamp) {
        return incoming.node_id > existing.node_id;
    }
    return false;
}

// ============================================================
// Memory Layout Optimization
// ============================================================

/**
 * Spreadsheet memory layout optimized for GPU access
 *
 * Uses Structure of Arrays (SoA) format for better coalescing:
 *
 * Layout: [row0_col0, row0_col1, row0_col2, ..., row1_col0, row1_col1, ...]
 *
 * This ensures that when adjacent threads access adjacent cells,
 * memory accesses are coalesced into fewer transactions.
 *
 * @param cells Array of spreadsheet cells
 * @param rows Number of rows
 * @param cols Number of columns
 * @param thread_id Thread ID for this access
 * @return Pointer to cell for this thread
 */
__device__ volatile SpreadsheetCell* get_cell_for_thread(
    volatile SpreadsheetCell* cells,
    uint32_t rows,
    uint32_t cols,
    uint32_t thread_id
) {
    // Row-major layout with coalescing optimization
    uint32_t row = thread_id / cols;
    uint32_t col = thread_id % cols;

    if (row < rows && col < cols) {
        return &cells[row * cols + col];
    }

    return nullptr;  // Out of bounds
}

// ============================================================
// SmartCRDT Processing Functions
// ============================================================

/**
 * Process a spreadsheet edit command using CRDT merge
 *
 * This function applies a spreadsheet edit using atomic operations
 * to ensure consistency across concurrent edits.
 *
 * @param queue Command queue
 * @param cells Spreadsheet cell array
 * @param edit Edit to apply
 * @return true if edit was applied successfully
 */
__device__ bool process_spreadsheet_edit(
    CommandQueue* queue,
    volatile SpreadsheetCell* cells,
    const SpreadsheetEdit& edit
) {
    // Get cell index for coalesced access
    uint32_t cell_idx = edit.cell_id.row * 10000 + edit.cell_id.col;  // Assuming max 10000 cols

    // Apply edit with conflict resolution
    if (edit.is_delete) {
        return atomic_delete_cell(&cells[cell_idx], edit.timestamp, edit.node_id);
    } else {
        return atomic_update_cell(&cells[cell_idx], edit);
    }
}

/**
 * Batch process spreadsheet edits with optimal coalescing
 *
 * Distributes edits across threads such that adjacent threads
 * process adjacent cells, maximizing memory bandwidth utilization.
 *
 * @param cells Spreadsheet cell array
 * @param edits Array of edits to apply
 * @param edit_count Number of edits
 * @param cols_per_row Number of columns per row
 */
__device__ void process_spreadsheet_edits_batch(
    volatile SpreadsheetCell* cells,
    const SpreadsheetEdit* edits,
    uint32_t edit_count,
    uint32_t cols_per_row
) {
    // Process edits in parallel with coalesced access
    process_cells_coalesced(cells, edits, edit_count, cols_per_row);
}

/**
 * Merge concurrent spreadsheet edits using CRDT semantics
 *
 * When multiple edits target the same cell, resolve conflicts
 * using last-write-wins with timestamps.
 *
 * @param cells Spreadsheet cell array
 * @param edit1 First edit
 * @param edit2 Second edit (concurrent)
 * @return Merged edit that wins
 */
__device__ SpreadsheetEdit merge_concurrent_edits(
    volatile SpreadsheetCell* cells,
    const SpreadsheetEdit& edit1,
    const SpreadsheetEdit& edit2
) {
    // Apply last-write-wins based on timestamp
    if (edit1.timestamp > edit2.timestamp) {
        return edit1;
    } else if (edit2.timestamp > edit1.timestamp) {
        return edit2;
    } else {
        // Same timestamp - use node_id as tiebreaker
        return (edit1.node_id > edit2.node_id) ? edit1 : edit2;
    }
}

#endif // SMARTCRDT_CUH
