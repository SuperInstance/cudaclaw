// ============================================================
// Persistent Worker Kernel - Simplified Command Processing
// ============================================================
// This file implements a persistent GPU kernel that continuously
// polls the CommandQueue and processes commands using the new
// binary interface defined in shared_types.h.
//
// KEY FEATURES:
// - Persistent kernel with while(queue->is_running) loop
// - Single thread (thread 0, block 0) manages queue
// - Lock-free ring buffer with volatile head/tail indices
// - __threadfence_system() for PCIe memory consistency
// - SmartCRDT integration for spreadsheet operations
//
// ARCHITECTURE:
// - Thread 0, Block 0: Queue manager and command processor
// - Other threads: Available for future parallel processing
// - Single-producer (Rust) / single-consumer (GPU) pattern
//
// MEMORY MODEL:
// - CommandQueue allocated in Unified Memory
// - Rust writes to queue->head (volatile)
// - GPU reads queue->head and writes to queue->tail (volatile)
// - __threadfence_system() ensures PCIe visibility
//
// ============================================================

#include <cuda_runtime.h>
#include "shared_types.h"
#include "smartcrdt.cuh"

// ============================================================
// Configuration Constants
// ============================================================

#define QUEUE_SIZE 1024
#define POLL_DELAY_NS 1000     // 1 microsecond - balances latency and power

// ============================================================
// SmartCRDT Integration - Cell Edit Processing
// ============================================================

/**
 * Process an EDIT_CELL command using SmartCRDT merge logic
 *
 * This function handles the actual cell edit operation on the
 * spreadsheet using CRDT (Conflict-free Replicated Data Type)
 * merge semantics to ensure consistency across nodes.
 *
 * @param cell_id  - The cell identifier to edit
 * @param value    - The new value to set
 * @param timestamp - Edit timestamp for conflict resolution
 * @param node_id   - Which node made this edit
 */
__device__ void process_edit_cell(
    uint32_t cell_id,
    float value,
    uint64_t timestamp,
    uint32_t node_id
) {
    // Calculate cell index for coalesced memory access
    // This ensures efficient GPU memory access patterns
    uint32_t row = cell_id / MAX_COLS;
    uint32_t col = cell_id % MAX_COLS;
    uint32_t cell_idx = get_coalesced_cell_index(row, col, MAX_COLS);

    // Create a SpreadsheetEdit structure for CRDT merge
    SpreadsheetEdit edit;
    edit.cell_id.row = row;
    edit.cell_id.col = col;
    edit.new_type = CELL_NUMBER;  // Assuming numeric value
    edit.numeric_value = value;
    edit.timestamp = timestamp;
    edit.node_id = node_id;
    edit.is_delete = 0;  // This is an edit, not a delete
    edit.string_ptr = 0;  // No string data
    edit.formula_ptr = 0;  // No formula data
    edit.value_len = 0;
    edit.reserved = 0;

    // In a real implementation, we would have access to the actual
    // spreadsheet cell array here. For now, we'll demonstrate the
    // SmartCRDT merge logic.

    // Example: Access global spreadsheet cell array
    // extern __device__ SpreadsheetCell g_spreadsheet_cells[];
    // SpreadsheetCell* cell = &g_spreadsheet_cells[cell_idx];
    // atomic_update_cell(cell, edit);

    printf("[GPU] EDIT_CELL: cell[%u] = %.2f (ts=%lu, node=%u)\n",
           cell_id, value, timestamp, node_id);
}

/**
 * Process a SYNC_CRDT command to synchronize CRDT state
 *
 * This handles multi-node synchronization of CRDT state,
 * ensuring consistency across distributed spreadsheet instances.
 *
 * @param node_id     - Source node for synchronization
 * @param timestamp   - Sync timestamp
 * @param vector_size - Size of CRDT vector to sync
 */
__device__ void process_sync_crdt(
    uint32_t node_id,
    uint64_t timestamp,
    uint32_t vector_size
) {
    printf("[GPU] SYNC_CRDT: node=%u, ts=%lu, vector_size=%u\n",
           node_id, timestamp, vector_size);

    // In a real implementation, this would:
    // 1. Receive CRDT state vector from source node
    // 2. Merge local state with received state
    // 3. Resolve conflicts using timestamp ordering
    // 4. Update local cell values
    // 5. Broadcast updates to dependent cells
}

// ============================================================
// Main Persistent Worker Kernel
// ============================================================

/**
 * Persistent worker kernel that continuously processes commands from the queue
 *
 * This kernel implements a simple, efficient persistent worker pattern:
 * 1. Continuously polls the command queue
 * 2. Processes commands as they arrive
 * 3. Updates tail index after processing
 * 4. Shuts down gracefully when signaled
 *
 * THREAD ORGANIZATION:
 * - Thread 0, Block 0: Queue manager and command processor
 * - Other threads: Currently unused (available for future parallelization)
 *
 * LAUNCH CONFIGURATION:
 * - Blocks: 1 (single block for queue management)
 * - Threads: 1-32 (only thread 0 is currently used)
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void persistent_worker(CommandQueue* queue) {
    // Only thread 0 of block 0 manages the queue
    // This prevents contention and simplifies synchronization
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // Initialize kernel statistics
    printf("[GPU] Persistent worker kernel started\n");
    printf("[GPU] Queue at %p, buffer size: %d commands\n", queue, QUEUE_SIZE);

    // ============================================================
    // MAIN PERSISTENT LOOP
    // ============================================================

    // Continue until external running flag is set to false
    while (queue->is_running) {

        // ============================================================
        // MEMORY FENCE FOR PCIe VISIBILITY
        // ============================================================
        // __threadfence_system() ensures that:
        // 1. All previous GPU operations are visible to CPU
        // 2. We see the latest queue->head written by CPU (Rust)
        // 3. Stronger than __threadfence() - works across PCIe bus
        // This is critical for Unified Memory consistency
        __threadfence_system();

        // ============================================================
        // POLL QUEUE HEAD INDEX
        // ============================================================
        // Read volatile head and tail indices
        // head: write index (updated by CPU/Rust)
        // tail: read index (updated by GPU/kernel)
        uint32_t head = queue->head;
        uint32_t tail = queue->tail;

        // ============================================================
        // CHECK FOR COMMANDS
        // ============================================================
        if (head != tail) {
            // Queue has commands - process one

            // Calculate index in circular buffer
            // Using modulo 1024 for wrap-around
            uint32_t cmd_idx = tail % QUEUE_SIZE;

            // Fetch command from buffer
            Command cmd = queue->buffer[cmd_idx];

            // ============================================================
            // PROCESS COMMAND BASED ON TYPE
            // ============================================================
            switch (cmd.type) {
                case NOOP:
                    // No-operation - just acknowledge
                    printf("[GPU] NOOP: Command %u processed\n", cmd._padding);
                    break;

                case EDIT_CELL: {
                    // ============================================================
                    // SMARTCRDT MERGE LOGIC
                    // ============================================================
                    // Extract cell_id and value from command data
                    uint32_t cell_id = cmd.data.edit_cell.cell_id;
                    float value = cmd.data.edit_cell.value;

                    // Process the cell edit using SmartCRDT merge logic
                    process_edit_cell(
                        cell_id,
                        value,
                        0,  // timestamp (would be in command)
                        0   // node_id (would be in command)
                    );

                    printf("[GPU] EDIT_CELL: Processed cell %u with value %.2f\n",
                           cell_id, value);
                    break;
                }

                case SYNC_CRDT: {
                    // ============================================================
                    // CRDT SYNCHRONIZATION
                    // ============================================================
                    // Extract sync parameters from command data
                    uint32_t node_id = cmd.data.sync_crdt.node_id;
                    uint64_t timestamp = cmd.data.sync_crdt.timestamp;
                    uint32_t vector_size = cmd.data.sync_crdt.vector_size;

                    // Process CRDT synchronization
                    process_sync_crdt(node_id, timestamp, vector_size);
                    break;
                }

                case SHUTDOWN:
                    // ============================================================
                    // GRACEFUL SHUTDOWN
                    // ============================================================
                    printf("[GPU] SHUTDOWN: Gracefully shutting down...\n");

                    // Set is_running flag to false
                    // This signals other threads (if any) to exit
                    queue->is_running = false;

                    // Update final statistics
                    printf("[GPU] Final stats: sent=%llu, processed=%llu\n",
                           queue->commands_sent,
                           queue->commands_processed);
                    break;

                default:
                    // Unknown command type - log error
                    printf("[GPU] ERROR: Unknown command type %u\n", cmd.type);
                    break;
            }

            // ============================================================
            // ADVANCE TAIL (POP COMMAND)
            // ============================================================
            // Move tail forward to indicate command is processed
            // This signals to Rust that the slot is available for reuse
            tail = (tail + 1) % QUEUE_SIZE;
            queue->tail = tail;

            // Update statistics
            queue->commands_processed++;

            // ============================================================
            // ENSURE MEMORY CONSISTENCY
            // ============================================================
            // Memory fence to ensure CPU sees our tail update
            // This prevents race conditions where Rust might
            // overwrite commands we haven't processed yet
            __threadfence_system();

        } else {
            // ============================================================
            // QUEUE EMPTY - IDLE WAITING
            // ============================================================
            // No commands to process - yield to prevent SM burnout
            // Use __nanosleep() instead of busy-waiting for efficiency

            #if __CUDA_ARCH__ >= 700
                // Pascal and newer: use nanosleep for efficient waiting
                // POLL_DELAY_NS = 1000 nanoseconds = 1 microsecond
                // This prevents burning SM cycles while maintaining low latency
                __nanosleep(POLL_DELAY_NS);
            #else
                // Older architectures: use minimal sleep
                // Spin for a few cycles to yield
                for (volatile int i = 0; i < 100; i++) {
                    __threadfence_block();
                }
            #endif

            // Update idle statistics (optional)
            // queue->commands_sent++; // Could track polling cycles
        }

        // ============================================================
        // PERIODIC STATE UPDATE (optional)
        // ============================================================
        // Every 1024 iterations, update statistics
        // This prevents overflow and provides monitoring
        static __device__ uint64_t cycle_count = 0;
        if ((cycle_count++ & 1023) == 0) {
            // Update statistics periodically
            // queue->total_cycles++;
        }
    }

    // ============================================================
    // KERNEL SHUTDOWN
    // ============================================================
    printf("[GPU] Persistent worker kernel shutting down...\n");
    printf("[GPU] Final statistics:\n");
    printf("[GPU]   Commands sent:     %llu\n", queue->commands_sent);
    printf("[GPU]   Commands processed: %llu\n", queue->commands_processed);

    // Set final state to indicate completion
    queue->is_running = false;

    // Final memory fence to ensure CPU sees our final state
    __threadfence_system();
}

// ============================================================
// Helper Kernels (Optional)
// ============================================================

/**
 * Initialize the command queue for persistent worker operation
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void init_command_queue(CommandQueue* queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize queue state
        queue->head = 0;
        queue->tail = 0;
        queue->is_running = false;
        queue->commands_sent = 0;
        queue->commands_processed = 0;

        printf("[GPU] Command queue initialized\n");
    }
}

/**
 * Signal the persistent worker to start processing
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void start_persistent_worker(CommandQueue* queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->is_running = true;
        printf("[GPU] Persistent worker started\n");
    }
}

/**
 * Signal the persistent worker to shut down gracefully
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void stop_persistent_worker(CommandQueue* queue) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        queue->is_running = false;
        printf("[GPU] Persistent worker stop signal sent\n");
    }
}

// ============================================================
// Statistics and Monitoring Kernels
// ============================================================

/**
 * Get queue statistics from the GPU
 *
 * @param queue Pointer to CommandQueue
 * @param stats Output array for statistics (4 elements)
 */
extern "C" __global__ void get_queue_stats(CommandQueue* queue, uint64_t* stats) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stats[0] = queue->commands_sent;
        stats[1] = queue->commands_processed;
        stats[2] = queue->head;
        stats[3] = queue->tail;
    }
}

// ============================================================
// End of executor.cu
// ============================================================
