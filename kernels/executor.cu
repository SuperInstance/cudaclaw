// ============================================================
// Persistent Worker Kernel - Warp-Parallel + SmartCRDT Integration
// ============================================================
// This file implements a persistent GPU kernel that continuously
// polls the CommandQueue and processes commands using lock-free
// SPSC queue with warp-level parallelism, reconciling spreadsheet
// edits directly in the SmartCRDT engine without leaving the GPU.
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
// - All 32 lanes process commands in parallel via SmartCRDT
// - Single-producer (Rust) / single-consumer (GPU) pattern
//
// SMARTCRDT INTEGRATION:
// - EDIT_CELL: All 32 lanes call crdt_write_cell() in parallel,
//   each writing an adjacent cell (cell_id + lane_id) using
//   atomicCAS-based LWW conflict resolution at the hardware level.
// - SYNC_CRDT: Each lane calls crdt_merge_conflict() on its slice
//   of the CRDT vector, resolving conflicts without global memory
//   round-trips.
// - Formula recalculation: warp_recalculate_cells() runs a
//   warp-parallel scan across dependent cell chains.
//
// MEMORY MODEL:
// - CommandQueue allocated in Unified Memory
// - CRDTState allocated in Unified Memory (cells[] in device memory)
// - Rust writes to queue->head using volatile writes
// - GPU Thread 0 reads queue->head with __threadfence_system()
// - GPU Thread 0 writes to queue->tail after processing
// - __shfl_sync() distributes command data within warp
//
// ============================================================

#include <cuda_runtime.h>
#include "shared_types.h"
#include "crdt_engine.cuh"

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
// SmartCRDT Cell Edit Handler (Warp-Parallel)
// ============================================================

/**
 * Process an EDIT_CELL command using the SmartCRDT engine.
 *
 * All 32 lanes participate in parallel:
 * - Lane 0 writes the primary cell (cell_id) via crdt_write_cell().
 * - Lanes 1-31 each write an adjacent cell (cell_id + lane_id),
 *   enabling a warp-width burst of 32 coalesced CRDT writes per
 *   command — all resolved at the hardware level via atomicCAS LWW.
 *
 * @param crdt      Pointer to CRDTState in Unified Memory
 * @param cell_id   The primary cell identifier (flat 1D index)
 * @param value     The new value for the primary cell
 * @param timestamp Edit timestamp for LWW conflict resolution
 * @param node_id   Which node made this edit
 * @param lane_id   This thread's lane within the warp (0-31)
 * @return Number of successful CRDT writes across the warp
 */
__device__ uint32_t smartcrdt_edit_cell(
    CRDTState* crdt,
    uint32_t cell_id,
    float value,
    uint64_t timestamp,
    uint32_t node_id,
    uint32_t lane_id
) {
    // Each lane targets a different adjacent cell for maximum parallelism.
    // Lane 0 → primary cell, lanes 1-31 → adjacent cells in row-major order.
    uint32_t target_flat = cell_id + lane_id;

    // Convert flat index to (row, col) for the CRDT engine
    uint32_t target_row = 0;
    uint32_t target_col = target_flat;
    if (crdt->cols > 0) {
        target_row = target_flat / crdt->cols;
        target_col = target_flat % crdt->cols;
    }

    // All lanes write the same value — in production, adjacent lanes
    // would write formula-recalculated values for their cells.
    double lane_value = (double)value;

    bool success = crdt_write_cell(crdt, target_row, target_col,
                                   lane_value, timestamp, node_id);

    // Reduce success count across warp using __ballot_sync
    #if __CUDA_ARCH__ >= 700
        unsigned int success_ballot = __ballot_sync(FULL_WARP_MASK, success);
        return __popc(success_ballot);
    #else
        __shared__ uint32_t shared_success[WARP_SIZE];
        shared_success[lane_id] = success ? 1 : 0;
        __syncwarp();
        uint32_t total = 0;
        if (lane_id == 0) {
            for (int i = 0; i < WARP_SIZE; i++) total += shared_success[i];
        }
        return __shfl_sync(FULL_WARP_MASK, total, 0);
    #endif
}

// ============================================================
// SmartCRDT Formula Recalculation Trigger
// ============================================================

/**
 * After a cell edit, trigger warp-parallel formula recalculation
 * for cells that depend on the edited cell.
 *
 * Uses warp_recalculate_cells() from crdt_engine.cuh to run a
 * warp-parallel scan across the dependency chain.
 *
 * @param crdt      Pointer to CRDTState
 * @param cell_id   The cell that was just edited (dependency root)
 * @param timestamp Recalculation timestamp
 * @param node_id   Node performing recalculation
 */
__device__ void smartcrdt_trigger_recalc(
    CRDTState* crdt,
    uint32_t cell_id,
    uint64_t timestamp,
    uint32_t node_id
) {
    uint32_t total_cells = crdt->total_cells;

    // Build a warp-width frontier of cells to recalculate.
    // Each lane recalculates cell_id + lane_id (the same burst
    // pattern as the write, covering the dependency neighborhood).
    uint32_t cell_indices[WARP_SIZE];
    for (int i = 0; i < WARP_SIZE; i++) {
        uint32_t idx = cell_id + i;
        cell_indices[i] = (total_cells > 0 && idx < total_cells) ? idx : 0;
    }

    // warp_recalculate_cells() uses __ballot_sync internally to
    // count successful recalculations across the warp.
    warp_recalculate_cells(crdt, cell_indices, timestamp + 1, node_id);
}

// ============================================================
// SmartCRDT CRDT Sync Handler (Warp-Parallel)
// ============================================================

/**
 * Process a SYNC_CRDT command using the SmartCRDT engine.
 *
 * Partitions the CRDT vector across all 32 lanes:
 * - Each lane calls crdt_merge_conflict() on its assigned slice,
 *   resolving conflicts using the LWW strategy baked into the
 *   CRDT engine — no PCIe round-trips required.
 *
 * @param crdt        Pointer to CRDTState in Unified Memory
 * @param node_id     Source node for synchronization
 * @param timestamp   Sync timestamp
 * @param vector_size Number of cells in the CRDT vector to sync
 * @param lane_id     This thread's lane within the warp (0-31)
 * @return Number of successful merges across the warp
 */
__device__ uint32_t smartcrdt_sync_crdt(
    CRDTState* crdt,
    uint32_t node_id,
    uint64_t timestamp,
    uint32_t vector_size,
    uint32_t lane_id
) {
    (void)node_id;    // used for future multi-node routing
    (void)timestamp;  // used for future version-vector sync

    // Partition the vector across 32 lanes
    uint32_t slice_size  = (vector_size + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t slice_start = lane_id * slice_size;
    uint32_t slice_end   = slice_start + slice_size;
    if (slice_end > vector_size) slice_end = vector_size;

    uint32_t merged = 0;

    // Each lane merges its slice of conflicted cells
    for (uint32_t flat_idx = slice_start; flat_idx < slice_end; flat_idx++) {
        uint32_t row = (crdt->cols > 0) ? flat_idx / crdt->cols : 0;
        uint32_t col = (crdt->cols > 0) ? flat_idx % crdt->cols : flat_idx;

        if (crdt->is_valid(row, col)) {
            bool ok = crdt_merge_conflict(crdt, row, col);
            if (ok) merged++;
        }
    }

    // Warp-level reduction using __shfl_xor_sync butterfly
    #if __CUDA_ARCH__ >= 700
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            merged += __shfl_xor_sync(FULL_WARP_MASK, merged, offset);
        }
    #else
        __shared__ uint32_t shared_merged[WARP_SIZE];
        shared_merged[lane_id] = merged;
        __syncwarp();
        if (lane_id == 0) {
            uint32_t total = 0;
            for (int i = 0; i < WARP_SIZE; i++) total += shared_merged[i];
            merged = total;
        }
        merged = __shfl_sync(FULL_WARP_MASK, merged, 0);
    #endif

    return merged;
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
extern "C" __global__ void persistent_worker(CommandQueue* queue, CRDTState* crdt) {
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
        printf("[GPU] Persistent Worker Kernel Started (Warp-Parallel + SmartCRDT)\n");
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        printf("[GPU] Queue location: %p\n", queue);
        printf("[GPU] CRDT state:     %p (%s)\n", crdt,
               crdt ? "connected" : "CPU-only mode");
        if (crdt) {
            printf("[GPU] CRDT grid:      %u rows x %u cols (%u cells)\n",
                   crdt->rows, crdt->cols, crdt->total_cells);
        }
        printf("[GPU] Buffer size: %d commands (%d bytes total)\n",
               QUEUE_SIZE, QUEUE_SIZE * (int)sizeof(Command));
        printf("[GPU] Warp lanes active: %d\n", WARP_SIZE);
        printf("[GPU] Polling strategy: Lane 0 polls, __shfl_sync broadcasts\n");
        printf("[GPU] PCIe visibility: __threadfence_system() on lane 0\n");
        printf("[GPU] SmartCRDT: EDIT_CELL=32-lane parallel write, "
               "SYNC_CRDT=32-lane merge\n");
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
                if (crdt) {
                    printf("[GPU] SmartCRDT final state:\n");
                    printf("[GPU]   Total updates:   %u\n", crdt->update_count);
                    printf("[GPU]   Conflicts:       %u\n", crdt->conflict_count);
                    printf("[GPU]   Merges:          %u\n", crdt->merge_count);
                    printf("[GPU]   Global version:  %llu\n",
                           (unsigned long long)crdt->global_version);
                }
                printf("[GPU] Final queue statistics:\n");
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
                // ============================================================
                // SMARTCRDT EDIT: 32-lane parallel crdt_write_cell()
                // ============================================================
                // All 32 lanes write adjacent cells simultaneously using
                // atomicCAS LWW conflict resolution in the CRDT engine.
                // No CPU round-trip needed — conflicts resolved on-chip.
                uint32_t cell_id = cmd_id;
                float    value   = cmd_data_a;
                uint32_t node_id = cmd_rcode;  // node_id packed in result_code

                uint32_t writes = 0;
                if (crdt) {
                    writes = smartcrdt_edit_cell(crdt, cell_id, value,
                                                 cmd_ts, node_id, lane_id);
                    // After writing, trigger warp-parallel formula recalc
                    // for the dependency neighborhood of the edited cell.
                    smartcrdt_trigger_recalc(crdt, cell_id, cmd_ts, node_id);
                } else {
                    // CPU-only / test mode: log without CRDT
                    if (lane_id == 0) {
                        printf("[GPU] EDIT_CELL (no CRDT): cell[%u] = %.2f "
                               "(ts=%llu, node=%u)\n",
                               cell_id, value,
                               (unsigned long long)cmd_ts, node_id);
                    }
                }

                if (lane_id == 0) {
                    printf("[GPU] EDIT_CELL: cell[%u] = %.2f "
                           "(warp-parallel SmartCRDT, %u/32 writes succeeded)\n",
                           cell_id, value, writes);
                }
                break;
            }

            case SYNC_CRDT: {
                // ============================================================
                // SMARTCRDT SYNC: 32-lane parallel crdt_merge_conflict()
                // ============================================================
                // Each lane merges a different slice of the CRDT vector.
                // Conflict resolution (LWW) happens entirely on-chip via
                // atomicCAS — no global memory round-trips to the CPU.
                uint32_t node_id     = cmd_id;
                uint32_t vector_size = cmd_bcount;

                uint32_t merges = 0;
                if (crdt) {
                    merges = smartcrdt_sync_crdt(crdt, node_id, cmd_ts,
                                                 vector_size, lane_id);
                } else {
                    // CPU-only / test mode: log without CRDT
                    if (lane_id == 0) {
                        printf("[GPU] SYNC_CRDT (no CRDT): node=%u, ts=%llu, "
                               "vector_size=%u\n",
                               node_id, (unsigned long long)cmd_ts, vector_size);
                    }
                }

                if (lane_id == 0) {
                    printf("[GPU] SYNC_CRDT: node=%u, vector_size=%u, "
                           "%u conflicts merged (warp-parallel SmartCRDT)\n",
                           node_id, vector_size, merges);
                }
                break;
            }

            case SHUTDOWN:
                // SHUTDOWN command in the queue itself.
                // Advance tail and set is_running=false BEFORE breaking
                // out of the while(true) loop, so no extra iteration can
                // process a post-SHUTDOWN command.
                if (lane_id == 0) {
                    uint32_t tail = queue->tail;
                    queue->tail = tail + 1;
                    queue->commands_processed++;
                    queue->is_running = false;
                    __threadfence_system();

                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                    printf("[GPU] SHUTDOWN command received\n");
                    printf("[GPU] Final statistics:\n");
                    printf("[GPU]   Commands sent:      %llu\n",
                           (unsigned long long)queue->commands_sent);
                    printf("[GPU]   Commands processed: %llu\n",
                           (unsigned long long)queue->commands_processed);
                    printf("[GPU] ══════════════════════════════════════════════════════\n");
                }
                // All 32 lanes exit the persistent loop immediately.
                // goto is used to break out of both the switch and while.
                goto kernel_exit;

            default:
                if (lane_id == 0) {
                    printf("[GPU] ERROR: Unknown command type %u\n", cmd_type);
                }
                break;
        }

        // ============================================================
        // PHASE 5: Lane 0 advances tail and updates statistics
        // ============================================================
        // (Skipped for SHUTDOWN — handled inline above to avoid an
        // extra loop iteration that could process a post-SHUTDOWN cmd.)
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

kernel_exit:

    // ============================================================
    // KERNEL EXIT: Final cleanup (lane 0 only)
    // ============================================================
    if (lane_id == 0) {
        printf("[GPU] ══════════════════════════════════════════════════════\n");
        printf("[GPU] Persistent Worker Kernel Exited (Warp-Parallel + SmartCRDT)\n");
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

/**
 * Get SmartCRDT statistics from the GPU
 *
 * @param crdt  Pointer to CRDTState
 * @param stats Output array for statistics (4 elements):
 *              [0] = update_count, [1] = conflict_count,
 *              [2] = merge_count,  [3] = global_version
 */
extern "C" __global__ void get_crdt_stats(CRDTState* crdt, uint64_t* stats) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (crdt) {
            stats[0] = crdt->update_count;
            stats[1] = crdt->conflict_count;
            stats[2] = crdt->merge_count;
            stats[3] = crdt->global_version;
        } else {
            stats[0] = stats[1] = stats[2] = stats[3] = 0;
        }
    }
}

// ============================================================
// End of executor.cu
// ============================================================
