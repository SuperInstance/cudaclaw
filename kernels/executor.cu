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
// - SmartCRDT integration temporarily disabled for testing
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
// #include "smartcrdt.cuh"  // Temporarily disabled for testing

// ============================================================
// Configuration Constants
// ============================================================

#define QUEUE_SIZE 1024
#define POLL_DELAY_NS 100      // 100 nanoseconds - prevents thermal throttling while staying responsive

// ============================================================
// Cell Edit Processing (Simplified - without SmartCRDT)
// ============================================================

/**
 * Process an EDIT_CELL command
 *
 * This is a simplified version for testing the persistent kernel
 * architecture. SmartCRDT integration will be added back later.
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
    // Simplified cell processing - just log for now
    printf("[GPU] EDIT_CELL: cell[%u] = %.2f (ts=%llu, node=%u)\n",
           cell_id, value, (unsigned long long)timestamp, node_id);
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
    printf("[GPU] SYNC_CRDT: node=%u, ts=%llu, vector_size=%u\n",
           node_id, (unsigned long long)timestamp, vector_size);

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
 * This kernel implements a lock-free persistent worker pattern using
 * a single-producer (Rust/CPU) / single-consumer (GPU/thread 0) model:
 *
 * 1. Thread 0 of block 0 continuously polls the tail index
 * 2. Uses __threadfence_system() to guarantee PCIe bus visibility
 * 3. When Rust increments head on CPU, GPU sees it instantly across PCIe
 * 4. Processes commands and advances tail to signal completion
 * 5. Uses __nanosleep(100) when idle to prevent thermal throttling
 *
 * THREAD ORGANIZATION:
 * - Thread 0, Block 0: Queue manager and command processor
 * - Other threads: Available for future parallel processing
 *
 * LAUNCH CONFIGURATION:
 * - Blocks: 1 (single block for queue management)
 * - Threads: 256 (only thread 0 is currently used, rest available)
 *
 * PCIE MEMORY MODEL:
 * - CommandQueue allocated in Unified Memory
 * - Rust writes to queue->head (volatile write from CPU)
 * - GPU reads queue->head (volatile read across PCIe)
 * - GPU writes to queue->tail (volatile write from GPU)
 * - __threadfence_system() ensures full PCIe bus visibility
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void persistent_worker(CommandQueue* queue) {
    // ============================================================
    // THREAD RESTRICTION: Only Thread 0 of Block 0 manages queue
    // ============================================================
    // This prevents contention and simplifies synchronization
    // Future: Other threads could be used for parallel command processing
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // ============================================================
    // KERNEL INITIALIZATION
    // ============================================================
    printf("[GPU] ══════════════════════════════════════════════════════\n");
    printf("[GPU] Persistent Worker Kernel Started\n");
    printf("[GPU] ══════════════════════════════════════════════════════\n");
    printf("[GPU] Queue location: %p\n", queue);
    printf("[GPU] Buffer size: %d commands (%d bytes total)\n",
           QUEUE_SIZE, QUEUE_SIZE * (int)sizeof(Command));
    printf("[GPU] Polling strategy: Thread 0 continuously polls tail index\n");
    printf("[GPU] PCIe visibility: __threadfence_system() guarantees instant visibility\n");
    printf("[GPU] Idle strategy: __nanosleep(100) prevents thermal throttling\n");

    // ============================================================
    // MAIN PERSISTENT LOOP
    // ============================================================
    // Continue until external running flag is set to false
    while (queue->is_running) {

        // ============================================================
        // CRITICAL: MEMORY FENCE FOR PCIe BUS VISIBILITY
        // ============================================================
        // __threadfence_system() is THE KEY to cross-CPU/GPU communication:
        //
        // 1. Forces all previous GPU memory operations to complete
        // 2. Ensures all writes from GPU are visible to CPU (across PCIe)
        // 3. Ensures GPU sees the latest queue->head written by CPU (Rust)
        // 4. Stronger than __threadfence() - works across the PCIe bus
        // 5. Critical for lock-free producer/consumer pattern across PCIe
        //
        // Without this fence, the CPU and GPU could have inconsistent views
        // of the queue state, leading to lost commands or corruption.
        __threadfence_system();

        // ============================================================
        // ATOMIC POLLING: Read queue indices
        // ============================================================
        // Read volatile head and tail indices from Unified Memory
        // - head: write index (updated by CPU/Rust via volatile write)
        // - tail: read index (updated by this GPU thread)
        //
        // These reads are volatile to ensure we get fresh values across PCIe
        uint32_t head = queue->head;
        uint32_t tail = queue->tail;

        // ============================================================
        // CHECK FOR WORK: Does head != tail?
        // ============================================================
        if (head != tail) {
            // ============================================================
            // COMMAND AVAILABLE: Process it
            // ============================================================

            // Calculate index in circular buffer (wrap-around at QUEUE_SIZE)
            uint32_t cmd_idx = tail % QUEUE_SIZE;

            // Fetch command from buffer (in Unified Memory, visible to both CPU/GPU)
            Command cmd = queue->buffer[cmd_idx];

            // ============================================================
            // PROCESS COMMAND based on type
            // ============================================================
            switch (cmd.cmd_type) {
                case NOOP:
                    // No-operation - just acknowledge receipt
                    printf("[GPU] ✓ NOOP: Command %u processed\n", cmd.id);
                    break;

                case EDIT_CELL: {
                    // Extract cell_id and value from command data
                    uint32_t cell_id = cmd.id;
                    float value = cmd.data_a;

                    // Process the cell edit
                    process_edit_cell(
                        cell_id,
                        value,
                        cmd.timestamp,
                        0  // node_id (would be in command)
                    );

                    printf("[GPU] ✓ EDIT_CELL: Processed cell[%u] = %.2f\n", cell_id, value);
                    break;
                }

                case SYNC_CRDT: {
                    // Extract sync parameters
                    uint32_t node_id = cmd.id;
                    uint64_t timestamp = cmd.timestamp;
                    uint32_t vector_size = cmd.batch_count;

                    // Process CRDT synchronization
                    process_sync_crdt(node_id, timestamp, vector_size);
                    break;
                }

                case SHUTDOWN:
                    // ============================================================
                    // GRACEFUL SHUTDOWN SEQUENCE
                    // ============================================================
                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                    printf("[GPU] SHUTDOWN command received\n");
                    printf("[GPU] Initiating graceful shutdown...\n");

                    // Set is_running flag to false (breaks the while loop)
                    queue->is_running = false;

                    // Update final statistics
                    printf("[GPU] Final statistics:\n");
                    printf("[GPU]   Commands sent:     %llu\n", queue->commands_sent);
                    printf("[GPU]   Commands processed: %llu\n", queue->commands_processed);
                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                    break;

                default:
                    // Unknown command type - log error
                    printf("[GPU] ✗ ERROR: Unknown command type %u\n", cmd.cmd_type);
                    break;
            }

            // ============================================================
            // ADVANCE TAIL: Signal command completion to CPU
            // ============================================================
            // Move tail forward to indicate command is processed
            // This signals to Rust that the slot is available for reuse
            tail = (tail + 1) % QUEUE_SIZE;
            queue->tail = tail;

            // Update statistics
            queue->commands_processed++;

            // ============================================================
            // CRITICAL: ENSURE CPU SEES OUR TAIL UPDATE
            // ============================================================
            // Memory fence to ensure CPU sees our tail update immediately
            // This prevents race conditions where Rust might overwrite
            // commands we haven't processed yet (buffer corruption)
            __threadfence_system();

        } else {
            // ============================================================
            // QUEUE EMPTY: Prevent thermal throttling with __nanosleep
            // ============================================================
            // No commands to process - sleep briefly to:
            // 1. Prevent GPU SM from burning cycles (thermal throttling)
            // 2. Maintain low latency for incoming commands (100ns)
            // 3. Reduce power consumption while staying responsive
            //
            // __nanosleep(100) = 100 nanoseconds sleep
            // - Fast enough for low-latency response (<1µs overhead)
            // - Long enough to prevent thermal issues
            // - Much more efficient than busy-waiting

            #if __CUDA_ARCH__ >= 700
                // Pascal and newer: use nanosleep for efficient waiting
                // POLL_DELAY_NS = 100 nanoseconds
                __nanosleep(POLL_DELAY_NS);
            #else
                // Older architectures: spin briefly to yield
                for (volatile int i = 0; i < 100; i++) {
                    __threadfence_block();
                }
            #endif
        }

        // ============================================================
        // OPTIONAL: Periodic health check (every 1024 iterations)
        // ============================================================
        static __device__ uint64_t cycle_count = 0;
        if ((cycle_count++ & 1023) == 0) {
            // Could add periodic health checks or statistics here
            // For now, just prevents cycle_count overflow
        }
    }

    // ============================================================
    // KERNEL EXIT: Final cleanup
    // ============================================================
    printf("[GPU] ══════════════════════════════════════════════════════\n");
    printf("[GPU] Persistent Worker Kernel Exited\n");
    printf("[GPU] ══════════════════════════════════════════════════════\n");

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
