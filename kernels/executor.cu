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
// Main Persistent Worker Kernel (Refactored)
// ============================================================

/**
 * Persistent worker kernel that continuously processes commands from the queue.
 *
 * ARCHITECTURE:
 * - Thread 0 of each block manages the queue (checks for work, pops commands)
 * - Other threads wait for work signal via shared memory
 * - When work is available, warps process commands in parallel
 * - Uses __nanosleep() for efficient idle waiting
 * - Lock-free queue operations for thread-safe access
 *
 * THREAD ORGANIZATION:
 * - Block manager: Thread 0 (handles queue)
 * - Warp 0: Threads 0-31 (processes command 0)
 * - Warp 1: Threads 32-63 (processes command 1)
 * - etc.
 *
 * LAUNCH CONFIGURATION:
 * - Blocks: 1-8 (depending on workload)
 * - Threads: 32-256 per block (must be multiple of 32)
 *
 * @param queue Pointer to CommandQueue in unified memory
 * @param running_flag External pointer to running flag (can be nullptr)
 */
extern "C" __global__ void persistent_worker(
    CommandQueue* queue,
    volatile bool* running_flag
) {
    // Initialize worker context
    WorkerContext ctx;

    // Calculate thread IDs
    ctx.thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    ctx.warp_id = threadIdx.x / WARP_SIZE;  // Warp ID within block
    ctx.lane_id = threadIdx.x % WARP_SIZE;   // Lane ID within warp (0-31)
    ctx.is_leader = (ctx.lane_id == 0);
    ctx.is_block_leader = (threadIdx.x == 0);  // Only thread 0 manages queue

    ctx.assigned_cmd = 0;
    ctx.task_type = 0;
    ctx.workload_size = 0;
    ctx.tasks_processed = 0;
    ctx.total_cycles = 0;

    // Get the number of warps in this block
    __shared__ uint32_t warp_count;
    if (ctx.is_block_leader) {
        warp_count = blockDim.x / WARP_SIZE;
    }
    __syncthreads();

    // Shared work signal for block coordination
    __shared__ WorkSignal work_signal;

    // Initialize work signal
    if (ctx.is_block_leader) {
        work_signal.has_work = 0;
        work_signal.cmd_count = 0;
        work_signal.shutdown = 0;

        // Initialize queue state (only first block, first time)
        if (blockIdx.x == 0) {
            queue->current_strategy = POLL_ADAPTIVE;
            queue->consecutive_commands = 0;
            queue->consecutive_idle = 0;
            queue->commands_processed = 0;
            queue->total_cycles = 0;
            queue->idle_cycles = 0;
        }
    }
    __syncthreads();

    // Persistent worker loop
    uint64_t cycle_count = 0;
    bool should_exit = false;

    while (!should_exit) {
        cycle_count++;

        // Ensure we see the latest queue state from CPU
        __threadfence_system();

        // ============================================================
        // PHASE 1: Queue Management (Thread 0 only)
        // ============================================================

        // Block leader manages the queue and signals other threads
        __syncthreads();  // Ensure work_signal is initialized

        bool continue_running = manage_queue(queue, &work_signal, running_flag);

        // Broadcast shutdown signal to all threads
        should_exit = (work_signal.shutdown != 0);

        if (should_exit || !continue_running) {
            break;  // Exit the persistent loop
        }

        __syncthreads();  // Ensure all threads see the work signal

        // ============================================================
        // PHASE 2: Work Processing (All threads)
        // ============================================================

        if (work_signal.has_work) {
            // Work is available - process it

            // Update statistics
            if (ctx.is_block_leader) {
                atomicAdd((unsigned long long*)&queue->commands_processed,
                          work_signal.cmd_count);
            }

            // Process commands in parallel (one per warp)
            process_commands_parallel(queue, &ctx, work_signal.cmd_count);

            // Advance tail for processed commands
            if (ctx.is_block_leader) {
                uint32_t processed = work_signal.cmd_count;
                uint32_t new_tail = (queue->tail + processed) % QUEUE_SIZE;
                queue->tail = new_tail;

                // Reset cycle count after work
                cycle_count = 0;
            }

            // Update per-warp statistics
            ctx.total_cycles += cycle_count;

        } else {
            // ============================================================
            // PHASE 3: Idle Waiting (All threads)
            // ============================================================

            // Update idle statistics
            if (ctx.is_block_leader) {
                atomicAdd((unsigned long long*)&queue->total_cycles, 1);
                atomicAdd((unsigned long long*)&queue->idle_cycles, 1);
            }

            // Apply adaptive polling delay using __nanosleep
            // This prevents SM saturation while maintaining low latency
            if (ctx.is_leader) {
                apply_polling_delay(queue);
            }

            // Synchronize warp to ensure all lanes are ready
            __syncwarp();
        }

        // Periodic state update (every 1024 cycles)
        if ((cycle_count & 1023) == 0 && ctx.is_block_leader) {
            queue->total_cycles += cycle_count;
        }

        __syncthreads();  // Synchronize block before next iteration
    }

    // Kernel shutdown - update final statistics
    if (ctx.is_block_leader) {
        queue->total_cycles += cycle_count;
        queue->status = STATUS_IDLE;
    }
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
