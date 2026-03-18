// ============================================================
// Persistent Worker Kernel - Warp-Parallel Lock-Free SPSC Queue
// ============================================================
// This file implements a persistent GPU kernel that continuously
// polls the CommandQueue and processes commands using lock-free
// SPSC queue with warp-level parallelism.
//
// KEY OPTIMIZATIONS:
// - Lock-free SPSC queue with volatile operations
// - Zero cudaDeviceSynchronize() calls in hot path
// - __threadfence_system() for PCIe visibility
// - Volatile writes from Rust for immediate GPU visibility
// - __shfl_sync() warp broadcast for parallel command processing
// - Ultra-low-latency dispatch-to-execution (< 5 microseconds target)
//
// ARCHITECTURE:
// - Block 0, Warp 0 (threads 0-31): Persistent polling kernel
// - Lane 0 (Thread 0): Queue manager, polls head index
// - Lanes 1-31: Receive commands via __shfl_sync broadcast
// - All 32 lanes process commands in parallel
// - Single-producer (Rust) / single-consumer (GPU) pattern
//
// MEMORY MODEL:
// - CommandQueue allocated in Unified Memory
// - Rust writes to queue->head using volatile writes
// - GPU Thread 0 reads queue->head with __threadfence_system()
// - GPU Thread 0 writes to queue->tail after processing
// - __shfl_sync() distributes command data within warp
//
// ============================================================

#include <cuda_runtime.h>
#include "shared_types.h"

// ============================================================
// Configuration Constants
// ============================================================

#define QUEUE_SIZE 1024
#define POLL_DELAY_NS 100      // 100 nanoseconds - prevents thermal throttling while staying responsive
#define WARP_SIZE 32
#define FULL_WARP_MASK 0xFFFFFFFF

// Sentinel values for warp broadcast control flow
#define WARP_SIGNAL_NO_WORK  0
#define WARP_SIGNAL_HAS_WORK 1
#define WARP_SIGNAL_SHUTDOWN  2

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
// Warp-Level Shuffle Helpers
// ============================================================

/**
 * Broadcast a uint64_t value from lane 0 to all lanes in the warp.
 * __shfl_sync operates on 32-bit values, so we split the 64-bit
 * value into high/low halves.
 */
__device__ __forceinline__ uint64_t shfl_broadcast_u64(
    uint64_t val,
    int src_lane
) {
    uint32_t lo = (uint32_t)(val & 0xFFFFFFFF);
    uint32_t hi = (uint32_t)(val >> 32);
    lo = __shfl_sync(FULL_WARP_MASK, lo, src_lane);
    hi = __shfl_sync(FULL_WARP_MASK, hi, src_lane);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/**
 * Broadcast a float value from lane 0 to all lanes in the warp.
 * Uses the float overload of __shfl_sync.
 */
__device__ __forceinline__ float shfl_broadcast_f32(
    float val,
    int src_lane
) {
    return __shfl_sync(FULL_WARP_MASK, val, src_lane);
}

// ============================================================
// Main Persistent Worker Kernel (Warp-Parallel)
// ============================================================

/**
 * Persistent worker kernel that continuously processes commands from the queue
 * using warp-level parallelism.
 *
 * ARCHITECTURE:
 * - Block 0, Warp 0 (lanes 0-31) run the persistent polling loop.
 * - Lane 0 (Thread 0) is the "queue manager":
 *     1. Calls __threadfence_system() to see the latest head written by Rust.
 *     2. Reads queue->head and queue->tail (volatile reads across PCIe).
 *     3. When a new command is detected, fetches it from the ring buffer.
 *     4. After processing, advances queue->tail and calls __threadfence_system()
 *        so the CPU sees the update immediately.
 * - Lane 0 broadcasts every field of the fetched Command to lanes 1-31
 *   using __shfl_sync(), giving all 32 threads an identical copy in registers.
 * - All 32 lanes then execute the command handler in parallel.
 *   For EDIT_CELL this means each lane can process a different sub-cell;
 *   for SYNC_CRDT each lane can merge a different slice of the CRDT vector.
 * - When no work is available, all lanes sleep via __nanosleep(100 ns)
 *   to prevent thermal throttling while staying responsive.
 *
 * LAUNCH CONFIGURATION:
 * - Blocks: 1
 * - Threads per block: 32 (one full warp)
 *
 * @param queue Pointer to CommandQueue in Unified Memory
 */
extern "C" __global__ void persistent_worker(CommandQueue* queue) {
    // ============================================================
    // THREAD RESTRICTION: Only Block 0, Warp 0 participates
    // ============================================================
    if (blockIdx.x != 0) return;

    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;

    // Only the first warp (threads 0-31) runs the persistent loop
    if (warp_id != 0) return;

    // ============================================================
    // KERNEL INITIALIZATION (lane 0 only)
    // ============================================================
    if (lane_id == 0) {
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        printf("[GPU] Persistent Worker Kernel Started (Warp-Parallel)\n");
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        printf("[GPU] Queue location: %p\n", queue);
        printf("[GPU] Buffer size: %d commands (%d bytes total)\n",
               QUEUE_SIZE, QUEUE_SIZE * (int)sizeof(Command));
        printf("[GPU] Warp lanes active: %d\n", WARP_SIZE);
        printf("[GPU] Polling strategy: Lane 0 polls, __shfl_sync broadcasts\n");
        printf("[GPU] PCIe visibility: __threadfence_system() on lane 0\n");
        printf("[GPU] Target latency: < 5 microseconds dispatch-to-execution\n");
    }

    // ============================================================
    // MAIN PERSISTENT LOOP (all 32 lanes participate)
    // ============================================================
    while (true) {
        // ============================================================
        // PHASE 1: Lane 0 polls the queue
        // ============================================================
        // Lane 0 issues __threadfence_system() and reads head/tail.
        // Other lanes wait at the __shfl_sync barrier below.
        uint32_t signal = WARP_SIGNAL_NO_WORK;

        // Command fields — only meaningful on lane 0 until broadcast
        uint32_t cmd_type    = 0;
        uint32_t cmd_id      = 0;
        uint64_t cmd_ts      = 0;
        float    cmd_data_a  = 0.0f;
        float    cmd_data_b  = 0.0f;
        float    cmd_result  = 0.0f;
        uint64_t cmd_batch   = 0;
        uint32_t cmd_bcount  = 0;
        uint32_t cmd_rcode   = 0;

        if (lane_id == 0) {
            // ============================================================
            // CRITICAL: MEMORY FENCE FOR PCIe BUS VISIBILITY
            // ============================================================
            // __threadfence_system() ensures we see the latest queue->head
            // written by the Rust host across the PCIe bus.  This is the
            // ONLY synchronization mechanism — no cudaDeviceSynchronize().
            __threadfence_system();

            // Check if kernel should keep running
            if (!queue->is_running) {
                signal = WARP_SIGNAL_SHUTDOWN;
            } else {
                // Volatile polling: read queue indices
                uint32_t head = queue->head;
                uint32_t tail = queue->tail;

                if (head != tail) {
                    // Command available — fetch from ring buffer
                    uint32_t cmd_idx = tail % QUEUE_SIZE;
                    Command cmd = queue->buffer[cmd_idx];

                    // Extract all fields for warp broadcast
                    cmd_type   = cmd.cmd_type;
                    cmd_id     = cmd.id;
                    cmd_ts     = cmd.timestamp;
                    cmd_data_a = cmd.data_a;
                    cmd_data_b = cmd.data_b;
                    cmd_result = cmd.result;
                    cmd_batch  = cmd.batch_data;
                    cmd_bcount = cmd.batch_count;
                    cmd_rcode  = cmd.result_code;

                    signal = WARP_SIGNAL_HAS_WORK;
                }
            }
        }

        // ============================================================
        // PHASE 2: Broadcast control signal to all lanes
        // ============================================================
        signal = __shfl_sync(FULL_WARP_MASK, signal, 0);

        // ============================================================
        // Handle shutdown: all lanes exit together
        // ============================================================
        if (signal == WARP_SIGNAL_SHUTDOWN) {
            if (lane_id == 0) {
                printf("[GPU] ══════════════════════════════════════════════════════\n");
                printf("[GPU] SHUTDOWN: is_running == false detected\n");
                printf("[GPU] Final statistics:\n");
                printf("[GPU]   Commands sent:      %llu\n",
                       (unsigned long long)queue->commands_sent);
                printf("[GPU]   Commands processed: %llu\n",
                       (unsigned long long)queue->commands_processed);
                printf("[GPU] ══════════════════════════════════════════════════════\n");
            }
            break;  // All 32 lanes break out of the while loop
        }

        // ============================================================
        // Handle idle: no commands available, sleep to avoid throttling
        // ============================================================
        if (signal == WARP_SIGNAL_NO_WORK) {
            #if __CUDA_ARCH__ >= 700
                __nanosleep(POLL_DELAY_NS);
            #else
                for (volatile int i = 0; i < 100; i++) {
                    __threadfence_block();
                }
            #endif
            continue;
        }

        // ============================================================
        // PHASE 3: Broadcast command fields from lane 0 to all lanes
        // ============================================================
        // Each __shfl_sync call distributes one register-width value
        // from lane 0 to every lane in the warp (~1 cycle each).
        cmd_type   = __shfl_sync(FULL_WARP_MASK, cmd_type,   0);
        cmd_id     = __shfl_sync(FULL_WARP_MASK, cmd_id,     0);
        cmd_ts     = shfl_broadcast_u64(cmd_ts, 0);
        cmd_data_a = shfl_broadcast_f32(cmd_data_a, 0);
        cmd_data_b = shfl_broadcast_f32(cmd_data_b, 0);
        cmd_result = shfl_broadcast_f32(cmd_result, 0);
        cmd_batch  = shfl_broadcast_u64(cmd_batch, 0);
        cmd_bcount = __shfl_sync(FULL_WARP_MASK, cmd_bcount, 0);
        cmd_rcode  = __shfl_sync(FULL_WARP_MASK, cmd_rcode,  0);

        // ============================================================
        // PHASE 4: Parallel command processing (all 32 lanes)
        // ============================================================
        // Every lane now holds an identical copy of the command in
        // registers.  The command handler can partition work across
        // lanes using lane_id.
        switch (cmd_type) {
            case NOOP:
                // No-operation — lane 0 acknowledges
                if (lane_id == 0) {
                    printf("[GPU] NOOP: Command %u processed (warp-parallel)\n",
                           cmd_id);
                }
                break;

            case EDIT_CELL: {
                // All 32 lanes have the cell edit data.
                // Lane 0 performs the primary edit; other lanes can
                // process dependent/adjacent cells in parallel.
                uint32_t cell_id = cmd_id;
                float value = cmd_data_a;

                if (lane_id == 0) {
                    // Primary cell edit
                    process_edit_cell(cell_id, value, cmd_ts, 0);
                } else {
                    // Lanes 1-31: process adjacent cells in parallel.
                    // Each lane handles cell_id + lane_id if within grid.
                    // (Actual bounds checking would use grid dimensions.)
                    uint32_t adjacent_cell = cell_id + lane_id;
                    // Placeholder for adjacent-cell recalculation;
                    // in production this would trigger formula re-eval.
                    (void)adjacent_cell;
                }

                if (lane_id == 0) {
                    printf("[GPU] EDIT_CELL: cell[%u] = %.2f "
                           "(warp-parallel, 32 lanes)\n", cell_id, value);
                }
                break;
            }

            case SYNC_CRDT: {
                // CRDT synchronization — partition the vector across lanes.
                // Each lane merges a different slice of the CRDT vector.
                uint32_t node_id     = cmd_id;
                uint32_t vector_size = cmd_bcount;

                // Each lane processes vector_size/32 elements
                uint32_t slice_size  = (vector_size + WARP_SIZE - 1) / WARP_SIZE;
                uint32_t slice_start = lane_id * slice_size;
                uint32_t slice_end   = slice_start + slice_size;
                if (slice_end > vector_size) slice_end = vector_size;

                // Each lane would merge its slice here
                // (Placeholder — real impl reads from CRDT state)
                (void)slice_start;
                (void)slice_end;

                if (lane_id == 0) {
                    process_sync_crdt(node_id, cmd_ts, vector_size);
                }
                break;
            }

            case SHUTDOWN:
                // SHUTDOWN command in the queue itself
                if (lane_id == 0) {
                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                    printf("[GPU] SHUTDOWN command received\n");
                    printf("[GPU] Initiating graceful shutdown...\n");
                    queue->is_running = false;
                    printf("[GPU] Final statistics:\n");
                    printf("[GPU]   Commands sent:      %llu\n",
                           (unsigned long long)queue->commands_sent);
                    printf("[GPU]   Commands processed: %llu\n",
                           (unsigned long long)queue->commands_processed);
                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                }
                break;

            default:
                if (lane_id == 0) {
                    printf("[GPU] ERROR: Unknown command type %u\n", cmd_type);
                }
                break;
        }

        // ============================================================
        // PHASE 5: Lane 0 advances tail and updates statistics
        // ============================================================
        if (lane_id == 0) {
            uint32_t tail = queue->tail;
            // Use monotonic (unwrapped) tail to match the host's monotonic
            // head counter. Buffer indexing uses tail % QUEUE_SIZE (already
            // done at line 231). Without this, after QUEUE_SIZE commands
            // the wrapped tail (0) != monotonic head (1024+), causing the
            // GPU to replay stale commands indefinitely. (BUG_0001 fix)
            queue->tail = tail + 1;
            queue->commands_processed++;

            // Memory fence so CPU sees our tail update immediately.
            // Prevents Rust from overwriting unprocessed commands.
            __threadfence_system();
        }
    }

    // ============================================================
    // KERNEL EXIT: Final cleanup (lane 0 only)
    // ============================================================
    if (lane_id == 0) {
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        printf("[GPU] Persistent Worker Kernel Exited (Warp-Parallel)\n");
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        __threadfence_system();
    }
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
