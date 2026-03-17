// ============================================================
// Persistent Worker Kernel - Warp-Level Parallelism
// ============================================================
// This file implements a persistent GPU kernel that continuously
// polls the CommandQueue and processes tasks using warp-level
// parallelism for maximum efficiency.
//
// KEY FEATURES:
// - Persistent kernel with while(running) loop
// - Warp-level task distribution and synchronization
// - Efficient workload balancing across 32 threads
// - Support for parallel command processing
//
// WARP-LEVEL OPERATIONS:
// - __syncthreads() for block-level sync
// - __ballot() for warp vote
// - __shfl_*() for warp shuffle communication
// - Atomic operations for shared state
//
// ============================================================

#include <cuda_runtime.h>
#include "shared_types.h"

// ============================================================
// Configuration Constants
// ============================================================

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 8
#define MAX_THREADS_PER_BLOCK (WARP_SIZE * MAX_WARPS_PER_BLOCK)

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

    // Task assignment
    uint32_t assigned_cmd;   // Which command this warp processes
    uint32_t task_type;      // Type of task to execute
    uint32_t workload_size;  // Number of work items

    // Statistics
    uint32_t tasks_processed;
    uint64_t total_cycles;
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
    uint32_t warp_count
) {
    // Each warp processes a different command
    // Warp 0 processes commands[tail], Warp 1 processes commands[tail+1], etc.

    uint32_t base_idx = queue->tail;
    uint32_t my_cmd_idx = base_idx + ctx->warp_id;

    // Check if this warp has work to do
    if (my_cmd_idx < base_idx + warp_count && my_cmd_idx < QUEUE_SIZE) {
        Command* my_cmd = &queue->commands[my_cmd_idx % QUEUE_SIZE];

        // Only process if command is ready
        if (my_cmd->type != CMD_NO_OP || my_cmd->id > 0) {
            // Mark as processing
            if (ctx->is_leader) {
                // Atomic operation to update status
                // Only one warp needs to do this
                if (ctx->warp_id == 0) {
                    queue->status = STATUS_PROCESSING;
                }
            }

            __syncwarp();  // Synchronize warp

            // Process the command
            process_command_warp(my_cmd, ctx);

            // Mark as done (only warp 0 does this)
            if (ctx->is_leader && ctx->warp_id == 0) {
                queue->status = STATUS_DONE;
            }
        }
    }
}

// ============================================================
// Queue Management - Warp-Level
// ============================================================

// Check for available work in the queue (warp-level optimization)
__device__ __forceinline__ bool has_work_available(CommandQueue* queue) {
    // Use warp-level ballot to check status efficiently
    bool ready = (queue->status == STATUS_READY);

    // Only one thread per warp needs to read the status
    // but all threads in the warp need to agree
    return warp_all_agree(ready);
}

// Get the number of available commands (warp-level scan)
__device__ __forceinline__ uint32_t get_available_command_count(CommandQueue* queue) {
    // Calculate how many commands are available in the circular buffer
    uint32_t head = queue->head;
    uint32_t tail = queue->tail;

    uint32_t count;
    if (head >= tail) {
        count = head - tail;
    } else {
        count = (QUEUE_SIZE - tail) + head;
    }

    // Broadcast to all lanes in warp
    return warp_broadcast(count, 0);
}

// ============================================================
// Statistics Collection - Warp-Level
// ============================================================

// Update queue statistics (only called by warp leaders)
__device__ void update_queue_stats(
    CommandQueue* queue,
    WorkerContext* ctx,
    uint64_t cycle_count
) {
    if (!ctx->is_leader) {
        return;  // Only leaders update statistics
    }

    // Use atomic operations to safely update statistics
    atomicAdd((unsigned long long*)&queue->commands_processed, 1);
    atomicAdd((unsigned long long*)&queue->total_cycles, cycle_count);

    // Update idle time if we had to wait
    uint64_t idle_time = cycle_count - ctx->total_cycles;
    if (idle_time > 0) {
        atomicAdd((unsigned long long*)&queue->idle_cycles, idle_time);
    }
}

// ============================================================
// Main Persistent Worker Kernel
// ============================================================

/**
 * Persistent worker kernel that continuously processes commands from the queue.
 *
 * This kernel uses warp-level parallelism to efficiently process commands:
 * - Each warp (32 threads) works on one command
 * - Multiple warps can work on different commands in parallel
 * - Warp leader (lane 0) makes control decisions
 * - All lanes in warp participate in data processing
 *
 * Launch configuration:
 * - Blocks: 1 (can be extended to multiple blocks)
 * - Threads: 32-256 per block (must be multiple of 32)
 *
 * @param queue Pointer to CommandQueue in unified memory
 */
extern "C" __global__ void persistent_worker(CommandQueue* queue) {
    // Initialize worker context
    WorkerContext ctx;

    // Calculate thread IDs
    ctx.thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    ctx.warp_id = threadIdx.x / WARP_SIZE;  // Warp ID within block
    ctx.lane_id = threadIdx.x % WARP_SIZE;   // Lane ID within warp (0-31)
    ctx.is_leader = (ctx.lane_id == 0);

    ctx.assigned_cmd = 0;
    ctx.task_type = 0;
    ctx.workload_size = 0;
    ctx.tasks_processed = 0;
    ctx.total_cycles = 0;

    // Get the number of warps in this block
    __shared__ uint32_t warp_count;
    if (threadIdx.x == 0) {
        warp_count = blockDim.x / WARP_SIZE;
    }
    __syncthreads();

    // Initialize queue state (only first block does this)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        queue->current_strategy = POLL_SPIN;
        queue->consecutive_commands = 0;
        queue->consecutive_idle = 0;
        queue->commands_processed = 0;
        queue->total_cycles = 0;
        queue->idle_cycles = 0;
    }

    // Persistent worker loop
    uint64_t cycle_count = 0;
    bool running = true;

    while (running) {
        cycle_count++;

        // Ensure we see the latest queue status from CPU
        __threadfence_system();

        // Warp-level check for available work
        // Only warp leaders need to check, but we sync the decision
        bool work_available = false;

        if (ctx.is_leader) {
            work_available = (queue->status == STATUS_READY);
        }

        // Broadcast the decision to all lanes in the warp
        work_available = __ballot_sync(0xFFFFFFFF, work_available) != 0;

        if (work_available) {
            // Work is available - process it

            // Count how many commands are ready
            uint32_t available_count = 0;
            if (ctx.is_leader) {
                available_count = get_available_command_count(queue);
            }
            available_count = warp_broadcast(available_count, 0);

            // Limit parallelism to available warps and commands
            uint32_t warps_to_use = min(available_count, warp_count);

            // Process commands in parallel (one per warp)
            process_commands_parallel(queue, &ctx, warps_to_use);

            // Update statistics
            update_queue_stats(queue, &ctx, cycle_count);

            // Update adaptive polling state
            if (ctx.is_leader && ctx.warp_id == 0) {
                atomicAdd((unsigned int*)&queue->consecutive_commands, 1);
                queue->consecutive_idle = 0;

                // Switch to spin mode during high activity
                if (queue->consecutive_commands > ACTIVITY_BURST) {
                    queue->current_strategy = POLL_SPIN;
                }

                // Check for shutdown command
                uint32_t tail_idx = queue->tail;
                if (queue->commands[tail_idx].type == CMD_SHUTDOWN) {
                    running = false;
                }
            }

            // Reset cycle count after work
            cycle_count = 0;

        } else {
            // No work available - update idle statistics

            if (ctx.is_leader && ctx.warp_id == 0) {
                atomicAdd((unsigned int*)&queue->consecutive_idle, 1);
                queue->consecutive_commands = 0;
            }

            // Apply adaptive polling delay
            if (ctx.is_leader) {
                PollingStrategy strategy = queue->current_strategy;

                switch (strategy) {
                    case POLL_SPIN:
                        // No delay - pure spin
                        break;

                    case POLL_ADAPTIVE:
                        // Adaptive delay based on idle time
                        if (queue->consecutive_idle > IDLE_THRESHOLD) {
                            // Use timed polling after being idle for a while
                            if (queue->consecutive_idle > IDLE_THRESHOLD * 10) {
                                // Switch to timed polling
                                queue->current_strategy = POLL_TIMED;
                            }
                            // Medium delay
                            uint64_t start = clock64();
                            uint64_t target = start + MEDIUM_POLL_CYCLES;
                            while (clock64() < target) {
                                __threadfence_block();
                            }
                        } else {
                            // Fast poll
                            uint64_t start = clock64();
                            uint64_t target = start + FAST_POLL_CYCLES;
                            while (clock64() < target) {
                                __threadfence_block();
                            }
                        }
                        break;

                    case POLL_TIMED:
                        // Fixed interval polling
                        uint64_t start = clock64();
                        uint64_t target = start + SLOW_POLL_CYCLES;
                        while (clock64() < target) {
                            __threadfence_block();
                        }

                        // Switch back to adaptive if we see activity
                        if (queue->consecutive_idle < ACTIVITY_BURST) {
                            queue->current_strategy = POLL_ADAPTIVE;
                        }
                        break;
                }
            }

            // Synchronize warp to ensure all lanes are ready
            __syncwarp();
        }

        // Periodic state update (every 1024 cycles)
        if ((cycle_count & 1023) == 0 && ctx.is_leader && ctx.warp_id == 0) {
            queue->total_cycles += cycle_count;
        }
    }

    // Kernel shutdown - update final statistics
    if (ctx.is_leader && ctx.warp_id == 0) {
        queue->total_cycles += cycle_count;
        queue->status = STATUS_IDLE;
    }
}

// ============================================================
// Multi-Block Persistent Worker Variant
// ============================================================

/**
 * Multi-block variant of the persistent worker.
 *
 * This version allows multiple thread blocks to work on different
 * parts of the command queue, enabling higher throughput.
 *
 * Launch configuration:
 * - Blocks: 1-8 (depending on workload)
 * - Threads: 32-256 per block
 *
 * @param queue Pointer to CommandQueue in unified memory
 * @param block_id Which block this is (for workload distribution)
 * @param total_blocks Total number of blocks
 */
extern "C" __global__ void persistent_worker_multiblock(
    CommandQueue* queue,
    uint32_t block_id,
    uint32_t total_blocks
) {
    // Similar to single-block version, but each block handles
    // a subset of the command queue

    WorkerContext ctx;
    ctx.thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    ctx.warp_id = threadIdx.x / WARP_SIZE;
    ctx.lane_id = threadIdx.x % WARP_SIZE;
    ctx.is_leader = (ctx.lane_id == 0);
    ctx.tasks_processed = 0;
    ctx.total_cycles = 0;

    // Calculate which portion of the queue this block handles
    uint32_t commands_per_block = (QUEUE_SIZE + total_blocks - 1) / total_blocks;
    uint32_t start_idx = block_id * commands_per_block;
    uint32_t end_idx = min(start_idx + commands_per_block, QUEUE_SIZE);

    bool running = true;
    uint64_t cycle_count = 0;

    while (running) {
        cycle_count++;

        __threadfence_system();

        // Check for work in this block's portion of the queue
        bool has_work = false;

        if (ctx.is_leader) {
            // Check if any command in our range is ready
            for (uint32_t i = start_idx; i < end_idx; i++) {
                uint32_t idx = i % QUEUE_SIZE;
                if (queue->commands[idx].type != CMD_NO_OP ||
                    queue->commands[idx].result_code != 0) {
                    has_work = true;
                    break;
                }
            }
        }

        has_work = __ballot_sync(0xFFFFFFFF, has_work) != 0;

        if (has_work) {
            // Process commands in our range
            for (uint32_t i = start_idx; i < end_idx; i++) {
                uint32_t idx = i % QUEUE_SIZE;
                Command* cmd = &queue->commands[idx];

                // Only process non-NOOP commands
                if (cmd->type != CMD_NO_OP || cmd->id > 0) {
                    process_command_warp(cmd, &ctx);

                    // Update statistics
                    if (ctx.is_leader && ctx.warp_id == 0) {
                        atomicAdd((unsigned long long*)&queue->commands_processed, 1);
                    }

                    // Check for shutdown
                    if (cmd->type == CMD_SHUTDOWN) {
                        running = false;
                        break;
                    }
                }
            }

            cycle_count = 0;
        } else {
            // Adaptive delay (same as single-block version)
            if (ctx.is_leader) {
                PollingStrategy strategy = queue->current_strategy;

                if (strategy != POLL_SPIN) {
                    uint32_t delay_cycles = (strategy == POLL_TIMED) ?
                        SLOW_POLL_CYCLES : MEDIUM_POLL_CYCLES;

                    uint64_t start = clock64();
                    uint64_t target = start + delay_cycles;
                    while (clock64() < target) {
                        __threadfence_block();
                    }
                }
            }

            __syncwarp();
        }
    }
}

// ============================================================
// Worker Statistics Kernel
// ============================================================

/**
 * Kernel to gather statistics from the worker queue.
 *
 * @param queue Pointer to CommandQueue
 * @param stats_out Output array for statistics
 */
extern "C" __global__ void get_worker_stats(CommandQueue* queue, uint64_t* stats_out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        stats_out[0] = queue->commands_processed;
        stats_out[1] = queue->total_cycles;
        stats_out[2] = queue->idle_cycles;
        stats_out[3] = queue->head;
        stats_out[4] = queue->tail;
        stats_out[5] = static_cast<uint64_t>(queue->status);
        stats_out[6] = static_cast<uint64_t>(queue->current_strategy);
        stats_out[7] = queue->consecutive_commands;
        stats_out[8] = queue->consecutive_idle;
        stats_out[9] = queue->avg_command_latency_cycles;
    }
}

// ============================================================
// Performance Monitoring
// ============================================================

/**
 * Kernel to measure warp efficiency and utilization.
 *
 * @param queue Pointer to CommandQueue
 * @param metrics_out Output array for metrics
 */
extern "C" __global__ void measure_warp_metrics(CommandQueue* queue, uint32_t* metrics_out) {
    // This would be implemented to measure:
    // - Warp utilization percentage
    // - Average warp execution time
    // - Memory transaction efficiency
    // - Divergence statistics

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
