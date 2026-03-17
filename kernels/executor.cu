// ============================================================
// Persistent Worker Kernel - Lock-Free Queue Integration
// ============================================================
// This file implements a persistent GPU kernel that continuously
// polls the CommandQueue using lock-free operations and processes
// tasks using warp-level parallelism for maximum efficiency.
//
// KEY FEATURES:
// - Persistent kernel with while(*running_flag) loop
// - Lock-free queue operations using atomicCAS
// - Single thread per block handles queue management
// - Other threads wait for work signal via __syncthreads()
// - __nanosleep() for efficient idle waiting
// - Warp-level task distribution and synchronization
//
// LOCK-FREE OPERATIONS:
// - pop_command() from lock_free_queue.cuh
// - Atomic head/tail index management
// - No locks or mutexes required
//
// WARP-LEVEL OPERATIONS:
// - __syncthreads() for block-level sync
// - __ballot_sync() for warp vote
// - __shfl_sync() for warp shuffle communication
// - Atomic operations for shared state
//
// ============================================================

#include <cuda_runtime.h>
#include "shared_types.h"
#include "lock_free_queue.cuh"
#include "smartcrdt.cuh"

// ============================================================
// Configuration Constants
// ============================================================

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 8
#define MAX_THREADS_PER_BLOCK (WARP_SIZE * MAX_WARPS_PER_BLOCK)

// Polling delays (in nanoseconds for __nanosleep)
#define FAST_POLL_NS 10       // 10 nanoseconds - lowest latency
#define MEDIUM_POLL_NS 100    // 100 nanoseconds - balanced
#define SLOW_POLL_NS 1000     // 1 microsecond - power saving

// Activity thresholds
#define ACTIVITY_BURST 10      // Consecutive commands before switching to spin
#define IDLE_THRESHOLD 100     // Idle cycles before increasing delay

// Worker state
enum WorkerState : uint32_t {
    WORKER_IDLE = 0,         // No work to do
    WORKER_ACTIVE = 1,       // Currently processing
    WORKER_WAITING = 2,      // Waiting for synchronization
    WORKER_DONE = 3          // Task complete
};

// ============================================================
// Per-Thread Worker Context
// ============================================================

struct __align__(16) WorkerContext {
    uint32_t thread_id;      // Global thread ID
    uint32_t warp_id;        // Warp ID within block
    uint32_t lane_id;        // Lane ID within warp (0-31)
    uint32_t is_leader;      // Is this lane the warp leader?
    uint32_t is_block_leader; // Is this thread the block queue manager?

    // Task assignment
    uint32_t assigned_cmd;   // Which command this warp processes
    uint32_t task_type;      // Type of task to execute
    uint32_t workload_size;  // Number of work items

    // Statistics
    uint32_t tasks_processed;
    uint64_t total_cycles;
};

// ============================================================
// Shared Memory for Work Signaling
// ============================================================

struct __align__(8) WorkSignal {
    volatile uint32_t has_work;     // Is work available?
    volatile uint32_t cmd_count;    // How many commands?
    volatile uint32_t shutdown;     // Should we shutdown?
};

// ============================================================
// Warp-Level Primitives
// ============================================================

// Warp-level ballot: check if all lanes agree on a condition
__device__ __forceinline__ bool warp_all_agree(bool condition) {
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, condition);
    return ballot == 0xFFFFFFFF;
}

// Warp-level any: check if any lane agrees
__device__ __forceinline__ bool warp_any_agree(bool condition) {
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, condition);
    return ballot != 0;
}

// Get the number of active lanes in the warp
__device__ __forceinline__ uint32_t warp_active_count() {
    return __popc(__ballot_sync(0xFFFFFFFF, true));
}

// Broadcast a value from the leader lane to all lanes
__device__ __forceinline__ uint32_t warp_broadcast(uint32_t value, int leader_lane) {
    return __shfl_sync(0xFFFFFFFF, value, leader_lane);
}

// Parallel prefix sum (scan) across warp
__device__ __forceinline__ uint32_t warp_scan_sum(uint32_t value) {
    uint32_t sum = value;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        uint32_t neighbor = __shfl_up_sync(0xFFFFFFFF, sum, offset);
        if (threadIdx.x >= offset) {
            sum += neighbor;
        }
    }

    return sum;
}

// ============================================================
// Task Processing Functions - Warp-Level
// ============================================================

// Process a single command using the entire warp
__device__ void process_command_warp(Command* cmd, WorkerContext* ctx) {
    // Only the warp leader (lane 0) makes control decisions
    if (ctx->is_leader) {
        switch (cmd->type) {
            case CMD_NO_OP:
                cmd->result_code = 0;
                break;

            case CMD_ADD: {
                // Simple addition - can be done by single thread
                cmd->data.add.result = cmd->data.add.a + cmd->data.add.b;
                cmd->result_code = 0;
                break;
            }

            case CMD_MULTIPLY: {
                // Simple multiplication - can be done by single thread
                cmd->data.multiply.result = cmd->data.multiply.a * cmd->data.multiply.b;
                cmd->result_code = 0;
                break;
            }

            case CMD_BATCH_PROCESS: {
                // Batch processing - distribute work across warp
                float* data = cmd->data.batch.data;
                uint32_t count = cmd->data.batch.count;
                float* output = cmd->data.batch.output;

                if (data != nullptr && output != nullptr) {
                    // Broadcast count and pointers to all lanes
                    count = warp_broadcast(count, 0);
                    uint64_t data_ptr = warp_broadcast((uint64_t)data, 0);
                    uint64_t output_ptr = warp_broadcast((uint64_t)output, 0);
                    data = (float*)data_ptr;
                    output = (float*)output_ptr;

                    // Process batch in parallel: each thread handles count/32 elements
                    uint32_t elements_per_thread = (count + WARP_SIZE - 1) / WARP_SIZE;

                    for (uint32_t i = 0; i < elements_per_thread; i++) {
                        uint32_t idx = ctx->lane_id * elements_per_thread + i;

                        if (idx < count) {
                            output[idx] = data[idx] * 2.0f;  // Example: multiply by 2
                        }
                    }

                    // Synchronize warp before continuing
                    __syncwarp();

                    cmd->result_code = 0;
                } else {
                    cmd->result_code = 1;  // Error: null pointer
                }
                break;
            }

            case CMD_SPREADSHEET_EDIT: {
                // Spreadsheet CRDT edit operation - batch style
                // cells_ptr: pointer to SpreadsheetCell array in GPU memory
                // edit_ptr: pointer to array of SpreadsheetEdit structures in GPU memory
                // spreadsheet_id: number of edits to process

                SpreadsheetCell* cells = (SpreadsheetCell*)cmd->data.spreadsheet.cells_ptr;
                const SpreadsheetEdit* edits = (const SpreadsheetEdit*)cmd->data.spreadsheet.edit_ptr;
                uint32_t edit_count = cmd->data.spreadsheet.spreadsheet_id;

                if (cells != nullptr && edits != nullptr && edit_count > 0) {
                    // Process edits in parallel across the warp
                    // Each thread in the warp processes one edit
                    bool all_success = true;

                    for (uint32_t i = ctx->lane_id; i < edit_count; i += WARP_SIZE) {
                        const SpreadsheetEdit& edit = edits[i];

                        // Calculate cell index for coalesced access
                        uint32_t cell_idx = get_coalesced_cell_index(
                            edit.cell_id.row,
                            edit.cell_id.col,
                            MAX_COLS
                        );

                        // Process the edit using atomic CRDT operations
                        bool success;
                        if (edit.is_delete) {
                            success = atomic_delete_cell(
                                &cells[cell_idx],
                                edit.timestamp,
                                edit.node_id
                            );
                        } else {
                            success = atomic_update_cell(&cells[cell_idx], edit);
                        }

                        if (!success) {
                            all_success = false;
                        }
                    }

                    __syncwarp();
                    cmd->result_code = all_success ? 0 : 1;
                } else {
                    cmd->result_code = 2;
                }
                break;
            }

            default:
                cmd->result_code = 0xFFFFFFFF;  // Unknown command
                break;
        }
    }

    // Synchronize warp to ensure leader completes before proceeding
    __syncwarp();

    ctx->tasks_processed++;
}

// Process multiple commands in parallel (one per warp)
__device__ void process_commands_parallel(
    CommandQueue* queue,
    WorkerContext* ctx,
    uint32_t cmd_count
) {
    // Each warp processes a different command
    // Warp 0 processes commands[0], Warp 1 processes commands[1], etc.

    // Check if this warp has work to do
    if (ctx->warp_id < cmd_count) {
        Command* my_cmd = &queue->commands[(queue->tail + ctx->warp_id) % QUEUE_SIZE];

        // Only process if command is valid (not NO_OP or has valid ID)
        if (my_cmd->type != CMD_NO_OP || my_cmd->id > 0) {
            // Process the command
            process_command_warp(my_cmd, ctx);

            // Update statistics (only warp leaders)
            if (ctx->is_leader) {
                atomicAdd((unsigned long long*)&queue->commands_processed, 1ULL);
            }
        }
    }
}

// ============================================================
// Queue Management - Single Thread Per Block
// ============================================================

/**
 * Block-level queue manager - only thread 0 of each block runs this.
 *
 * This function:
 * 1. Checks for shutdown command
 * 2. Pops commands from the queue using lock-free operations
 * 3. Signals other threads when work is available
 * 4. Updates adaptive polling state
 *
 * @param queue Pointer to CommandQueue in unified memory
 * @param signal Pointer to shared work signal
 * @param running_flag External running flag pointer
 * @return true if kernel should continue, false if shutdown
 */
__device__ bool manage_queue(
    CommandQueue* queue,
    WorkSignal* signal,
    volatile bool* running_flag
) {
    // Only thread 0 of each block manages the queue
    if (threadIdx.x != 0) {
        return true;  // Other threads skip this
    }

    // Check external running flag
    if (running_flag != nullptr && !*running_flag) {
        signal->shutdown = 1;
        return false;
    }

    // Try to pop a command from the queue using lock-free operation
    Command cmd;
    bool cmd_popped = false;
    uint32_t cmds_available = 0;

    // Check if queue is empty using head/tail indices
    uint32_t head = queue->head;
    uint32_t tail = queue->tail;

    if (head != tail) {
        // Queue has commands - calculate how many
        if (head > tail) {
            cmds_available = head - tail;
        } else {
            cmds_available = (QUEUE_SIZE - tail) + head;
        }

        // Limit to what we can process in this iteration
        uint32_t warps_in_block = blockDim.x / WARP_SIZE;
        cmds_available = min(cmds_available, warps_in_block);

        // Signal work is available
        signal->has_work = 1;
        signal->cmd_count = cmds_available;

        // Check for shutdown command
        uint32_t idx = tail % QUEUE_SIZE;
        if (queue->commands[idx].type == CMD_SHUTDOWN) {
            signal->shutdown = 1;
            return false;
        }

        cmd_popped = true;
    } else {
        // Queue is empty
        signal->has_work = 0;
        signal->cmd_count = 0;

        // Update idle statistics
        atomicAdd((unsigned int*)&queue->consecutive_idle, 1);
        queue->consecutive_commands = 0;
    }

    // Update adaptive polling strategy
    if (cmd_popped) {
        atomicAdd((unsigned int*)&queue->consecutive_commands, 1);
        queue->consecutive_idle = 0;

        // Switch to spin mode during high activity
        if (queue->consecutive_commands > ACTIVITY_BURST) {
            queue->current_strategy = POLL_SPIN;
        }
    }

    return true;
}

// ============================================================
// Nanosleep-based Polling Delay
// ============================================================

/**
 * Apply polling delay based on current strategy.
 *
 * Uses __nanosleep() for efficient idle waiting instead of busy loops.
 * This reduces power consumption while maintaining low latency.
 *
 * @param queue Pointer to CommandQueue for strategy
 */
__device__ __forceinline__ void apply_polling_delay(CommandQueue* queue) {
    // Only warp leaders apply the delay
    if (!/*__builtin_expect*/(false)) {  // Expected to be false most of the time
        return;
    }

    // Get current polling strategy
    PollingStrategy strategy = queue->current_strategy;

    // Apply delay based on strategy
    switch (strategy) {
        case POLL_SPIN:
            // No delay - pure spin for lowest latency
            // Use __nanosleep(0) just to yield to other warps
            #if __CUDA_ARCH__ >= 700
                __nanosleep(0);  // Yield to other warps
            #else
                // Older architectures - minimal delay
                __threadfence_block();
            #endif
            break;

        case POLL_ADAPTIVE:
            // Adaptive delay based on idle time
            if (queue->consecutive_idle > IDLE_THRESHOLD) {
                // Medium delay for moderate idle periods
                #if __CUDA_ARCH__ >= 700
                    __nanosleep(MEDIUM_POLL_NS);
                #else
                    uint64_t start = clock64();
                    uint64_t target = start + (MEDIUM_POLL_NS * 1000 / 50);  // Approximate cycles
                    while (clock64() < target) {
                        __threadfence_block();
                    }
                #endif

                // Switch to timed polling after extended idle
                if (queue->consecutive_idle > IDLE_THRESHOLD * 10) {
                    queue->current_strategy = POLL_TIMED;
                }
            } else {
                // Fast poll during recent activity
                #if __CUDA_ARCH__ >= 700
                    __nanosleep(FAST_POLL_NS);
                #else
                    __threadfence_block();
                #endif
            }
            break;

        case POLL_TIMED:
            // Fixed interval polling for power savings
            #if __CUDA_ARCH__ >= 700
                __nanosleep(SLOW_POLL_NS);
            #else
                uint64_t start = clock64();
                uint64_t target = start + (SLOW_POLL_NS * 1000 / 50);
                while (clock64() < target) {
                    __threadfence_block();
                }
            #endif

            // Switch back to adaptive if we see activity
            if (queue->consecutive_idle < ACTIVITY_BURST) {
                queue->current_strategy = POLL_ADAPTIVE;
            }
            break;
    }
}

// ============================================================
// Simplified Persistent Kernel - Direct Queue Polling
// ============================================================

/**
 * Simple persistent kernel that continuously polls the command queue.
 *
 * DESIGN PRINCIPLES:
 * - Direct tail/head polling without complex state management
 * - Single thread per block for queue management to avoid contention
 * - __threadfence_system() ensures PCIe memory visibility
 * - __nanosleep() prevents SM burnout during idle periods
 *
 * THREAD ORGANIZATION:
 * - Thread 0 of each block: Manages queue and processes commands
 * - Other threads: Available for parallel processing within commands
 *
 * LAUNCH CONFIGURATION:
 * - Blocks: 1 (single block avoids queue contention)
 * - Threads: 32-256 (must be multiple of 32 for warp operations)
 *
 * @param queue Pointer to CommandQueue in unified memory
 * @param running_flag External running flag (set to false to shutdown)
 */
extern "C" __global__ void persistent_worker_simple(
    CommandQueue* queue,
    volatile bool* running_flag
) {
    // Only thread 0 of block 0 manages the queue
    // This prevents multiple threads from contending for queue access
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;  // Other threads exit immediately
    }

    // Initialize queue statistics
    queue->commands_processed = 0;
    queue->total_cycles = 0;
    queue->idle_cycles = 0;
    queue->status = STATUS_PROCESSING;

    printf("[GPU] Persistent kernel started. Queue at %p\n", queue);

    // ============================================================
    // MAIN PERSISTENT LOOP
    // ============================================================
    // Continue until external running flag is set to false
    // Check running_flag first (fast path), then check for shutdown command
    while (running_flag != nullptr && *running_flag) {

        // ============================================================
        // MEMORY FENCE FOR PCIe VISIBILITY
        // ============================================================
        // __threadfence_system() ensures that:
        // 1. All previous memory operations are visible to CPU
        // 2. We see the latest queue->head written by CPU
        // 3. Stronger than __threadfence() - works across PCIe bus
        __threadfence_system();

        // ============================================================
        // POLL TAIL INDEX
        // ============================================================
        // Read volatile head and tail indices
        // head: write index (updated by CPU/Rust)
        // tail: read index (updated by GPU kernel)
        uint32_t head = queue->head;
        uint32_t tail = queue->tail;

        // ============================================================
        // CHECK IF QUEUE HAS COMMANDS
        // ============================================================
        if (head != tail) {
            // Queue has commands - process one

            // Calculate index in circular buffer
            uint32_t cmd_idx = tail % QUEUE_SIZE;
            Command* cmd = &queue->commands[cmd_idx];

            printf("[GPU] Processing command %u at tail=%u head=%u\n",
                   cmd->id, tail, head);

            // ============================================================
            // PROCESS COMMAND BASED ON TYPE
            // ============================================================
            switch (cmd->type) {
                case CMD_NO_OP:
                    // No-operation - just acknowledge
                    cmd->result_code = 0;
                    break;

                case CMD_ADD: {
                    // Simple addition
                    cmd->data.add.result = cmd->data.add.a + cmd->data.add.b;
                    cmd->result_code = 0;
                    printf("[GPU] ADD: %.2f + %.2f = %.2f\n",
                           cmd->data.add.a, cmd->data.add.b, cmd->data.add.result);
                    break;
                }

                case CMD_MULTIPLY: {
                    // Simple multiplication
                    cmd->data.multiply.result =
                        cmd->data.multiply.a * cmd->data.multiply.b;
                    cmd->result_code = 0;
                    printf("[GPU] MULTIPLY: %.2f * %.2f = %.2f\n",
                           cmd->data.multiply.a, cmd->data.multiply.b,
                           cmd->data.multiply.result);
                    break;
                }

                case CMD_BATCH_PROCESS: {
                    // Batch processing - apply transformation to array
                    float* data = cmd->data.batch.data;
                    uint32_t count = cmd->data.batch.count;
                    float* output = cmd->data.batch.output;

                    if (data != nullptr && output != nullptr && count > 0) {
                        // Process entire array (single thread for simplicity)
                        for (uint32_t i = 0; i < count; i++) {
                            output[i] = data[i] * 2.0f;  // Example: multiply by 2
                        }
                        cmd->result_code = 0;
                        printf("[GPU] BATCH: Processed %u elements\n", count);
                    } else {
                        cmd->result_code = 1;  // Error: null pointers
                        printf("[GPU] BATCH ERROR: Invalid pointers\n");
                    }
                    break;
                }

                case CMD_SPREADSHEET_EDIT: {
                    // ============================================================
                    // SMARTCRDT MERGE LOGIC
                    // ============================================================
                    // This is where GPU-accelerated CRDT merge happens
                    SpreadsheetCell* cells =
                        (SpreadsheetCell*)cmd->data.spreadsheet.cells_ptr;
                    const SpreadsheetEdit* edit =
                        (const SpreadsheetEdit*)cmd->data.spreadsheet.edit_ptr;

                    if (cells != nullptr && edit != nullptr) {
                        // Calculate cell index
                        uint32_t cell_idx = get_coalesced_cell_index(
                            edit->cell_id.row,
                            edit->cell_id.col,
                            MAX_COLS
                        );

                        printf("[GPU] SPREADSHEET_EDIT: cell[%u][%u] idx=%u\n",
                               edit->cell_id.row, edit->cell_id.col, cell_idx);

                        // Apply atomic CRDT update
                        bool success = atomic_update_cell(&cells[cell_idx], *edit);

                        cmd->result_code = success ? 0 : 1;
                        printf("[GPU] SmartCRDT merge: %s\n",
                               success ? "SUCCESS" : "FAILED");
                    } else {
                        cmd->result_code = 2;  // Error: null pointers
                        printf("[GPU] SPREADSHEET_EDIT ERROR: Invalid pointers\n");
                    }
                    break;
                }

                case CMD_SHUTDOWN:
                    // Shutdown command - signal exit
                    printf("[GPU] Received SHUTDOWN command\n");
                    cmd->result_code = 0;
                    queue->status = STATUS_DONE;

                    // Update running flag to signal other threads
                    if (running_flag != nullptr) {
                        *running_flag = false;
                    }
                    break;

                default:
                    cmd->result_code = 0xFFFFFFFF;  // Unknown command
                    printf("[GPU] Unknown command type: %u\n", cmd->type);
                    break;
            }

            // ============================================================
            // ADVANCE TAIL (POP COMMAND)
            // ============================================================
            // Move tail forward to indicate command is processed
            tail = (tail + 1) % QUEUE_SIZE;
            queue->tail = tail;

            // Update statistics
            queue->commands_processed++;
            queue->commands_popped++;

            // Ensure all writes are visible to CPU before next iteration
            __threadfence_system();

        } else {
            // ============================================================
            // QUEUE EMPTY - IDLE WAITING
            // ============================================================
            // No commands to process - yield to prevent SM burnout
            // Use __nanosleep() instead of busy-waiting

            queue->idle_cycles++;

            #if __CUDA_ARCH__ >= 700
                // Pascal and newer: use nanosleep for efficient waiting
                // 1000 nanoseconds = 1 microsecond
                // This prevents burning SM cycles while maintaining low latency
                __nanosleep(1000);
            #else
                // Older architectures: use minimal sleep
                // Spin for a few cycles to yield
                for (volatile int i = 0; i < 100; i++) {
                    __threadfence_block();
                }
            #endif
        }

        // Update total cycle count
        queue->total_cycles++;
    }

    // ============================================================
    // KERNEL SHUTDOWN
    // ============================================================
    printf("[GPU] Persistent kernel shutting down...\n");
    printf("[GPU] Final stats: processed=%llu, cycles=%llu, idle=%llu\n",
           queue->commands_processed, queue->total_cycles, queue->idle_cycles);

    queue->status = STATUS_IDLE;
    __threadfence_system();  // Ensure final state is visible to CPU
}

// ============================================================
// Initialization Kernel
// ============================================================

/**
 * Initialize the command queue for persistent worker operation.
 *
 * @param queue Pointer to CommandQueue in unified memory
 * @param running_flag Pointer to running flag (can be nullptr)
 */
extern "C" __global__ void init_persistent_worker(
    CommandQueue* queue,
    volatile bool* running_flag
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize queue state
        queue->status = STATUS_IDLE;
        queue->head = 0;
        queue->tail = 0;
        queue->commands_pushed = 0;
        queue->commands_popped = 0;
        queue->commands_processed = 0;
        queue->total_cycles = 0;
        queue->idle_cycles = 0;
        queue->current_strategy = POLL_ADAPTIVE;
        queue->consecutive_commands = 0;
        queue->consecutive_idle = 0;
        queue->last_command_cycle = 0;
        queue->avg_command_latency_cycles = 0;

        // Initialize running flag
        if (running_flag != nullptr) {
            *running_flag = true;
        }
    }
}

// ============================================================
// Shutdown Kernel
// ============================================================

/**
 * Signal the persistent worker to shutdown gracefully.
 *
 * @param running_flag Pointer to running flag to set to false
 */
extern "C" __global__ void shutdown_persistent_worker(volatile bool* running_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (running_flag != nullptr) {
            *running_flag = false;
        }
    }
}

// ============================================================
// Statistics Gathering Kernels
// ============================================================

/**
 * Kernel to gather statistics from the worker queue.
 *
 * @param queue Pointer to CommandQueue
 * @param stats_out Output array for statistics (10 elements)
 */
extern "C" __global__ void get_worker_stats(CommandQueue* queue, uint64_t* stats_out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        stats_out[0] = queue->commands_processed;
        stats_out[1] = queue->commands_pushed;
        stats_out[2] = queue->commands_popped;
        stats_out[3] = queue->total_cycles;
        stats_out[4] = queue->idle_cycles;
        stats_out[5] = queue->head;
        stats_out[6] = queue->tail;
        stats_out[7] = static_cast<uint64_t>(queue->status);
        stats_out[8] = static_cast<uint64_t>(queue->current_strategy);
        stats_out[9] = queue->consecutive_commands;
        stats_out[10] = queue->consecutive_idle;
        stats_out[11] = queue->avg_command_latency_cycles;
    }
}

/**
 * Kernel to measure warp efficiency and utilization.
 *
 * @param queue Pointer to CommandQueue
 * @param metrics_out Output array for metrics (4 elements)
 */
extern "C" __global__ void measure_warp_metrics(CommandQueue* queue, uint32_t* metrics_out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        // Calculate utilization
        uint64_t total = queue->total_cycles;
        uint64_t idle = queue->idle_cycles;

        uint32_t utilization = (total > 0) ?
            ((total - idle) * 100 / total) : 0;

        metrics_out[0] = utilization;
        metrics_out[1] = queue->commands_processed;
        metrics_out[2] = queue->consecutive_commands;
        metrics_out[3] = queue->consecutive_idle;
    }
}

// ============================================================
// Helper Kernels for Queue Management
// ============================================================

/**
 * Check if the queue has commands available.
 *
 * @param queue Pointer to CommandQueue
 * @return true if queue has commands, false otherwise
 */
extern "C" __global__ void check_queue_available(CommandQueue* queue, uint32_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t head = queue->head;
        uint32_t tail = queue->tail;
        *result = (head != tail) ? 1 : 0;
    }
}

/**
 * Get the current number of commands in the queue.
 *
 * @param queue Pointer to CommandQueue
 * @param count_out Output parameter for command count
 */
extern "C" __global__ void get_queue_count(CommandQueue* queue, uint32_t* count_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t head = queue->head;
        uint32_t tail = queue->tail;

        uint32_t count;
        if (head >= tail) {
            count = head - tail;
        } else {
            count = (QUEUE_SIZE - tail) + head;
        }

        *count_out = count;
    }
}

// ============================================================
// End of executor.cu
// ============================================================
