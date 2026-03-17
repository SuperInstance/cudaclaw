// ============================================================
// Lock-Free Command Queue - CUDA Device Functions
// ============================================================
// This file contains the CUDA device functions for the lock-free
// CommandQueue that is shared between Rust host and GPU device.
//
// The queue uses a circular buffer design with atomic head/tail indices
// to enable concurrent push (Rust) and pop (GPU) operations without
// explicit locking.
//
// ARCHITECTURE:
// - Unified Memory: Both CPU and GPU can access the same memory
// - Lock-Free: Uses atomicCAS for concurrent access
// - Circular Buffer: Fixed-size array with wraparound
// - Producer-Consumer: Rust pushes, GPU pops
//
// ============================================================

#ifndef LOCK_FREE_QUEUE_CUH
#define LOCK_FREE_QUEUE_CUH

#include <cuda_runtime.h>
#include <cstdint>

#include "shared_types.h"

// ============================================================
// Atomic Operations for Lock-Free Queue
// ============================================================

// Atomic compare-and-swap wrapper for CUDA
__device__ inline bool atomic_compare_exchange_uint32(
    volatile uint32_t* ptr,
    uint32_t expected,
    uint32_t desired
) {
    return atomicCAS(ptr, expected, desired) == expected;
}

// Atomic fetch-add wrapper
__device__ inline uint32_t atomic_fetch_add_uint32(
    volatile uint32_t* ptr,
    uint32_t value
) {
    return atomicAdd(ptr, value);
}

// Memory fence for ordering
__device__ inline void memory_fence() {
    __threadfence();
}

// ============================================================
// Queue State Management
// ============================================================

// Queue states for synchronization
enum QueueState : uint32_t {
    QUEUE_EMPTY = 0,       // No commands available
    QUEUE_PARTIAL = 1,     // Some commands available
    QUEUE_FULL = 2,        // No space for new commands
    QUEUE_LOCKED = 3,      // Queue being updated (future use)
};

// ============================================================
// Device Functions
// ============================================================

/**
 * Pop a command from the queue (GPU side)
 *
 * This function attempts to atomically pop a command from the queue.
 * It uses a CAS (Compare-And-Swap) loop to ensure thread-safe access.
 *
 * @param queue Pointer to the CommandQueue in unified memory
 * @param cmd Pointer to store the popped command
 * @return true if a command was successfully popped, false if queue was empty
 */
__device__ bool pop_command(CommandQueue* queue, Command* cmd) {
    // Get current tail index
    uint32_t tail = queue->tail;
    uint32_t head = queue->head;

    // Check if queue is empty
    if (head == tail) {
        return false;  // Queue is empty
    }

    // Calculate actual index in circular buffer
    uint32_t index = tail % QUEUE_SIZE;

    // Memory fence to ensure we see the latest data
    __threadfence();

    // Attempt to advance tail using CAS
    uint32_t new_tail = (tail + 1) % QUEUE_SIZE;

    // Use volatile read for head to get latest value
    uint32_t current_head = queue->head;

    // Double-check that queue is still not empty
    if (current_head == new_tail) {
        return false;  // Queue became empty
    }

    // Try to claim this slot by advancing tail
    if (atomic_compare_exchange_uint32(&queue->tail, tail, new_tail)) {
        // Successfully claimed this slot - read the command
        *cmd = queue->commands[index];

        // Update statistics
        atomicAdd((unsigned long long*)&queue->commands_popped, 1ULL);

        // Memory fence to ensure command is read before any further operations
        __threadfence();

        return true;
    }

    // CAS failed - another thread claimed this slot
    return false;
}

/**
 * Pop multiple commands from the queue (batch processing)
 *
 * This function attempts to pop multiple commands in a single operation,
 * which can be more efficient than popping one at a time.
 *
 * @param queue Pointer to the CommandQueue
 * @param cmds Array to store popped commands
 * @param max_cmds Maximum number of commands to pop
 * @return Actual number of commands popped
 */
__device__ uint32_t pop_commands_batch(
    CommandQueue* queue,
    Command* cmds,
    uint32_t max_cmds
) {
    uint32_t popped = 0;

    for (uint32_t i = 0; i < max_cmds; i++) {
        if (!pop_command(queue, &cmds[i])) {
            break;  // Queue is empty
        }
        popped++;
    }

    return popped;
}

/**
 * Get the current number of commands in the queue
 *
 * @param queue Pointer to the CommandQueue
 * @return Number of commands currently in queue
 */
__device__ uint32_t get_queue_size(CommandQueue* queue) {
    uint32_t head = queue->head;
    uint32_t tail = queue->tail;

    if (tail >= head) {
        return tail - head;
    } else {
        return QUEUE_SIZE - (head - tail);
    }
}

/**
 * Check if queue is empty
 *
 * @param queue Pointer to the CommandQueue
 * @return true if queue is empty
 */
__device__ bool is_queue_empty(CommandQueue* queue) {
    return queue->head == queue->tail;
}

/**
 * Check if queue is full
 *
 * @param queue Pointer to the CommandQueue
 * @return true if queue is full
 */
__device__ bool is_queue_full(CommandQueue* queue) {
    uint32_t next_head = (queue->head + 1) % QUEUE_SIZE;
    return next_head == queue->tail;
}

/**
 * Wait for command to become available (blocking)
 *
 * This function spins until a command is available. Use sparingly
 * as it consumes GPU cycles.
 *
 * @param queue Pointer to the CommandQueue
 * @param cmd Pointer to store the popped command
 * @param max_spins Maximum number of spin iterations (0 = infinite)
 * @return true if command was popped, false if timeout
 */
__device__ bool wait_for_command(CommandQueue* queue, Command* cmd, uint32_t max_spins) {
    uint32_t spins = 0;

    while (true) {
        if (pop_command(queue, cmd)) {
            return true;
        }

        if (max_spins > 0 && ++spins >= max_spins) {
            return false;  // Timeout
        }

        // Small delay to reduce GPU power consumption
        #ifdef __CUDACC_RTC__
            // In emulation mode, yield more aggressively
            #if __CUDA_ARCH__ >= 700
                __nanosleep(100);  // 100 nanoseconds
            #else
                __nanosleep(1000);  // 1 microsecond
            #endif
        #else
            // In device mode, minimal delay
            __nanosleep(10);  // 10 nanoseconds
        #endif
    }
}

/**
 * Batch wait for multiple commands
 *
 * Wait until the specified number of commands are available,
 * then pop them all at once.
 *
 * @param queue Pointer to the CommandQueue
 * @param cmds Array to store popped commands
 * @param min_cmds Minimum number of commands to wait for
 * @param max_spins Maximum number of spin iterations
 * @return Actual number of commands popped
 */
__device__ uint32_t wait_for_commands_batch(
    CommandQueue* queue,
    Command* cmds,
    uint32_t min_cmds,
    uint32_t max_spins
) {
    uint32_t spins = 0;

    while (true) {
        uint32_t available = get_queue_size(queue);

        if (available >= min_cmds) {
            uint32_t to_pop = (available < min_cmds * 2) ? available : min_cmds;
            return pop_commands_batch(queue, cmds, to_pop);
        }

        if (max_spins > 0 && ++spins >= max_spins) {
            return 0;  // Timeout
        }

        #ifdef __CUDACC_RTC__
            __nanosleep(100);
        #else
            __nanosleep(10);
        #endif
    }
}

// ============================================================
// Warp-Level Queue Operations
// ============================================================

/**
 * Warp-level pop command
 *
 * In a warp of 32 threads, only the lane 0 (leader) performs the
 * pop operation, then broadcasts the result to all other lanes.
 * This is useful when multiple threads in a warp need the same command.
 *
 * @param queue Pointer to the CommandQueue
 * @param cmd Pointer to store the popped command (all lanes)
 * @return true if command was successfully popped
 */
__device__ bool warp_pop_command(CommandQueue* queue, Command* cmd) {
    int lane_id = threadIdx.x % 32;
    bool success = false;

    if (lane_id == 0) {
        // Only lane 0 performs the pop
        success = pop_command(queue, cmd);
    }

    // Broadcast success flag to all lanes
    success = __shfl_sync(0xFFFFFFFF, success, 0);

    // Broadcast command to all lanes if successful
    if (success) {
        Command temp_cmd = *cmd;
        temp_cmd.type = (CommandType)__shfl_sync(0xFFFFFFFF, (uint32_t)temp_cmd.type, 0);
        temp_cmd.id = __shfl_sync(0xFFFFFFFF, temp_cmd.id, 0);
        temp_cmd.timestamp = __shfl_sync(0xFFFFFFFF, temp_cmd.timestamp, 0);
        temp_cmd.data.add.a = __shfl_sync(0xFFFFFFFF, temp_cmd.data.add.a, 0);
        temp_cmd.data.add.b = __shfl_sync(0xFFFFFFFF, temp_cmd.data.add.b, 0);
        temp_cmd.data.add.result = __shfl_sync(0xFFFFFFFF, temp_cmd.data.add.result, 0);
        temp_cmd.result_code = __shfl_sync(0xFFFFFFFF, temp_cmd.result_code, 0);
        *cmd = temp_cmd;
    }

    return success;
}

/**
 * Warp-level batch pop
 *
 * Pop multiple commands and distribute them across warp lanes.
 * Each lane gets a different command.
 *
 * @param queue Pointer to the CommandQueue
 * @param cmds Array to store popped commands (per-lane)
 * @return Number of commands popped by this warp
 */
__device__ uint32_t warp_pop_commands(
    CommandQueue* queue,
    Command* cmds
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    uint32_t total_popped = 0;

    // Only lane 0 of each warp performs coordination
    __shared__ uint32_t shared_counts[32];
    __shared__ uint32_t shared_start[32];

    if (lane_id == 0) {
        // Get queue size
        uint32_t available = get_queue_size(queue);
        uint32_t cmds_per_warp = (available + 31) / 32;  // Round up
        shared_start[warp_id] = cmds_per_warp * warp_id;
        shared_counts[warp_id] = cmds_per_warp;
    }

    __syncthreads();

    uint32_t warp_count = shared_start[warp_id];
    uint32_t warp_start_index = shared_start[warp_id];

    // Each lane pops its command
    if (lane_id < warp_count) {
        if (pop_command(queue, &cmds[lane_id])) {
            total_popped++;
        }
    }

    __syncthreads();

    // Broadcast total popped count to all lanes
    total_popped = __shfl_sync(0xFFFFFFFF, total_popped, 0);

    return total_popped;
}

#endif // LOCK_FREE_QUEUE_CUH
