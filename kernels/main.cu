// CudaClaw - Persistent GPU Kernel Executor
// Optimized for sub-10 microsecond latency with adaptive polling
// This kernel runs continuously on the GPU, processing commands from a queue in unified memory

#include <cuda_runtime.h>

// ============================================================
// Configuration Constants
// ============================================================

// Polling strategies
enum PollingStrategy : uint32_t {
    POLL_SPIN = 0,         // Tight spin - lowest latency, highest power
    POLL_ADAPTIVE = 1,     // Adaptive - balances latency and power
    POLL_TIMED = 2,        // Fixed interval - lower power, higher latency
};

// Polling intervals (in clock cycles, approximate)
// For reference: 1 GHz GPU = ~1 ns per cycle
#define SPIN_ITERATIONS        0       // No delay - pure spin
#define FAST_POLL_CYCLES       100     // ~100 ns delay
#define MEDIUM_POLL_CYCLES     1000    // ~1 µs delay
#define SLOW_POLL_CYCLES       10000   // ~10 µs delay

// Adaptive polling thresholds
#define IDLE_THRESHOLD         1000    // Cycles before switching to slower poll
#define ACTIVITY_BURST         10      // Consecutive commands before fast poll

// Status flags for the command queue
enum QueueStatus : uint32_t {
    STATUS_IDLE = 0,       // No command to process
    STATUS_READY = 1,      // Command ready to process
    STATUS_PROCESSING = 2, // Currently processing a command
    STATUS_DONE = 3,       // Command processing complete
    STATUS_ERROR = 4       // Error occurred
};

// Command types
enum CommandType : uint32_t {
    CMD_NO_OP = 0,        // No-operation - for latency testing
    CMD_SHUTDOWN = 1,     // Shutdown the persistent kernel
    CMD_ADD = 2,          // Add two numbers
    CMD_MULTIPLY = 3,     // Multiply two numbers
    CMD_BATCH_PROCESS = 4 // Batch process data
};

// ============================================================
// Data Structures
// ============================================================

// Command structure (optimized for cache line alignment)
struct __align__(32) Command {
    CommandType type;     // Type of command to execute
    uint32_t id;          // Unique command ID
    uint64_t timestamp;   // Host timestamp when command was issued

    // Command-specific data (union-like usage)
    union {
        struct {          // For CMD_NO_OP
            uint32_t padding;
        } noop;

        struct {          // For CMD_ADD
            float a;
            float b;
            float result;
        } add;

        struct {          // For CMD_MULTIPLY
            float a;
            float b;
            float result;
        } multiply;

        struct {          // For CMD_BATCH_PROCESS
            float* data;
            uint32_t count;
            float* output;
        } batch;
    } data;

    uint32_t result_code; // Result code (0 = success)
};

// Command Queue in Unified Memory with adaptive polling
struct __align__(128) CommandQueue {
    volatile QueueStatus status;  // Current status (use volatile for GPU-CPU sync)

    // Circular buffer for commands
    static const uint32_t QUEUE_SIZE = 16;
    Command commands[QUEUE_SIZE];

    volatile uint32_t head;  // Index where host writes new commands
    volatile uint32_t tail;  // Index where GPU reads commands

    // Statistics
    volatile uint64_t commands_processed;
    volatile uint64_t total_cycles;
    volatile uint64_t idle_cycles;

    // Adaptive polling state
    volatile PollingStrategy current_strategy;
    volatile uint32_t consecutive_commands;
    volatile uint32_t consecutive_idle;

    // Performance metrics
    volatile uint64_t last_command_cycle;
    volatile uint64_t avg_command_latency_cycles;

    // Padding to prevent false sharing and ensure cache line alignment
    uint8_t padding[64];
};

// ============================================================
// Device Functions - Polling Optimization
// ============================================================

// Optimized busy-wait with cycle counting
__device__ __forceinline__ void busy_wait_cycles(uint32_t cycles) {
    if (cycles == 0) return;

    uint64_t start = clock64();
    uint64_t target = start + cycles;

    // Tight spin loop for minimal latency
    while (clock64() < target) {
        // Optional: Add memory barrier to ensure visibility
        __threadfence_block();
    }
}

// Adaptive polling delay - adjusts based on recent activity
__device__ void adaptive_poll_delay(CommandQueue* queue) {
    uint32_t delay_cycles = 0;

    // Check current strategy
    PollingStrategy strategy = queue->current_strategy;

    switch (strategy) {
        case POLL_SPIN:
            // No delay - pure spin for lowest latency
            delay_cycles = SPIN_ITERATIONS;
            break;

        case POLL_ADAPTIVE:
            // Adjust based on activity patterns
            if (queue->consecutive_idle > IDLE_THRESHOLD) {
                // Been idle for a while - slow down polling
                delay_cycles = MEDIUM_POLL_CYCLES;

                // Consider switching to timed polling
                if (queue->consecutive_idle > IDLE_THRESHOLD * 10) {
                    atomicExch(reinterpret_cast<unsigned int*>(&queue->current_strategy), POLL_TIMED);
                }
            } else if (queue->consecutive_commands > ACTIVITY_BURST) {
                // High activity - switch to fast spin
                delay_cycles = SPIN_ITERATIONS;
            } else {
                // Mixed activity - use fast poll
                delay_cycles = FAST_POLL_CYCLES;
            }
            break;

        case POLL_TIMED:
            // Fixed interval polling - lowest power
            delay_cycles = SLOW_POLL_CYCLES;

            // Switch back to adaptive if we see activity
            if (queue->consecutive_commands > ACTIVITY_BURST) {
                atomicExch(reinterpret_cast<unsigned int*>(&queue->current_strategy), POLL_ADAPTIVE);
                delay_cycles = SPIN_ITERATIONS;
            }
            break;
    }

    // Apply the delay
    if (delay_cycles > 0) {
        busy_wait_cycles(delay_cycles);
        queue->idle_cycles += delay_cycles;
    }
}

// Optimized memory fence for unified memory visibility
__device__ __forceinline__ void ensure_visibility() {
    // Ensure all writes to unified memory are visible to CPU
    __threadfence_system();

    // Memory barrier within the warp
    __syncthreads();
}

// ============================================================
// Command Processing
// ============================================================

// Device function to process a single command
__device__ void process_command(Command* cmd) {
    switch (cmd->type) {
        case CMD_NO_OP:
            // Pure latency test - just acknowledge receipt
            cmd->result_code = 0;
            break;

        case CMD_ADD:
            cmd->data.add.result = cmd->data.add.a + cmd->data.add.b;
            cmd->result_code = 0;
            break;

        case CMD_MULTIPLY:
            cmd->data.multiply.result = cmd->data.multiply.a * cmd->data.multiply.b;
            cmd->result_code = 0;
            break;

        case CMD_BATCH_PROCESS:
            // Simple batch processing: multiply each element by 2
            if (cmd->data.batch.data != nullptr && cmd->data.batch.output != nullptr) {
                for (uint32_t i = 0; i < cmd->data.batch.count; i++) {
                    cmd->data.batch.output[i] = cmd->data.batch.data[i] * 2.0f;
                }
                cmd->result_code = 0;
            } else {
                cmd->result_code = 1; // Error: null pointer
            }
            break;

        case CMD_SHUTDOWN:
            cmd->result_code = 0;
            break;

        default:
            cmd->result_code = 0xFFFFFFFF; // Unknown command
            break;
    }
}

// ============================================================
// Optimized Persistent Kernel
// ============================================================

// Persistent kernel that continuously polls the command queue with adaptive polling
extern "C" __global__ void cuda_claw_executor(CommandQueue* queue) {
    // Get global thread ID (only thread 0 in block 0 does the processing)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only one thread processes commands (single-threaded executor for simplicity)
    if (tid != 0) {
        return;
    }

    // Initialize adaptive polling state
    queue->current_strategy = POLL_SPIN;  // Start with spin for fastest initial response
    queue->consecutive_commands = 0;
    queue->consecutive_idle = 0;
    queue->idle_cycles = 0;

    // Persistent loop - keeps running until shutdown command
    uint64_t cycle_count = 0;
    uint64_t last_activity_cycle = 0;

    while (true) {
        cycle_count++;

        // Memory barrier to ensure we see the latest status from CPU
        ensure_visibility();

        // Check if there's a command ready (volatile read ensures we get fresh value)
        volatile QueueStatus current_status = queue->status;

        if (current_status == STATUS_READY) {
            // Command available - process it immediately
            uint64_t command_start = clock64();

            // Get the command to process
            uint32_t idx = queue->tail;

            // Process the command
            queue->status = STATUS_PROCESSING;
            ensure_visibility();  // Ensure CPU sees status change

            process_command(&queue->commands[idx]);

            // Update statistics
            queue->commands_processed++;
            queue->consecutive_commands++;
            queue->consecutive_idle = 0;  // Reset idle counter

            // Calculate command latency
            uint64_t command_end = clock64();
            uint64_t latency = command_end - command_start;
            queue->last_command_cycle = command_end;
            queue->avg_command_latency_cycles = latency;

            // Mark command as complete
            queue->status = STATUS_DONE;
            ensure_visibility();  // Critical: Ensure CPU sees DONE status immediately

            // Move tail forward (circular buffer)
            queue->tail = (queue->tail + 1) % CommandQueue::QUEUE_SIZE;

            last_activity_cycle = cycle_count;

            // Switch to spin mode during high activity
            if (queue->consecutive_commands > ACTIVITY_BURST) {
                queue->current_strategy = POLL_SPIN;
            }

            // Check for shutdown command
            if (queue->commands[idx].type == CMD_SHUTDOWN) {
                queue->total_cycles = cycle_count;
                break; // Exit the persistent loop
            }
        } else {
            // No command available - update idle statistics
            queue->consecutive_idle++;
            queue->consecutive_commands = 0;  // Reset command counter

            // Apply adaptive polling delay
            adaptive_poll_delay(queue);
        }

        // Optional: Periodic state updates (every 1000 cycles)
        if ((cycle_count & 1023) == 0) {
            queue->total_cycles = cycle_count;
        }
    }
}

// ============================================================
// High-Performance Variant (Pure Spin-Wait)
// ============================================================

// Ultra-low latency variant - pure spin wait, no adaptive polling
// Use this when latency is more important than power consumption
extern "C" __global__ void cuda_claw_executor_spin(CommandQueue* queue) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid != 0) {
        return;
    }

    uint64_t cycle_count = 0;

    // Pure spin loop - no delays, maximum responsiveness
    while (true) {
        cycle_count++;

        // Minimal overhead status check
        if (queue->status == STATUS_READY) {
            uint32_t idx = queue->tail;

            // Process command immediately
            queue->status = STATUS_PROCESSING;
            process_command(&queue->commands[idx]);

            queue->commands_processed++;

            // Critical: Ensure CPU sees the status change immediately
            __threadfence_system();
            queue->status = STATUS_DONE;
            __threadfence_system();  // Double fence for maximum visibility

            queue->tail = (queue->tail + 1) % CommandQueue::QUEUE_SIZE;

            if (queue->commands[idx].type == CMD_SHUTDOWN) {
                queue->total_cycles = cycle_count;
                break;
            }
        }

        // No delay - pure spin
    }
}

// ============================================================
// Power-Optimized Variant (Timed Polling)
// ============================================================

// Lower power variant - fixed interval polling
// Use this when power consumption is more important than latency
extern "C" __global__ void cuda_claw_executor_timed(CommandQueue* queue) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid != 0) {
        return;
    }

    uint64_t cycle_count = 0;

    // Timed polling loop - fixed delay between checks
    while (true) {
        cycle_count++;

        // Check for command
        if (queue->status == STATUS_READY) {
            uint32_t idx = queue->tail;

            queue->status = STATUS_PROCESSING;
            process_command(&queue->commands[idx]);

            queue->commands_processed++;
            __threadfence_system();
            queue->status = STATUS_DONE;
            __threadfence_system();

            queue->tail = (queue->tail + 1) % CommandQueue::QUEUE_SIZE;

            if (queue->commands[idx].type == CMD_SHUTDOWN) {
                queue->total_cycles = cycle_count;
                break;
            }
        }

        // Fixed delay for power savings
        busy_wait_cycles(SLOW_POLL_CYCLES);
    }
}

// ============================================================
// Helper Kernels
// ============================================================

// Helper kernel to initialize the queue
extern "C" __global__ void init_command_queue(CommandQueue* queue) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        queue->status = STATUS_IDLE;
        queue->head = 0;
        queue->tail = 0;
        queue->commands_processed = 0;
        queue->total_cycles = 0;
        queue->idle_cycles = 0;
        queue->current_strategy = POLL_SPIN;
        queue->consecutive_commands = 0;
        queue->consecutive_idle = 0;
        queue->last_command_cycle = 0;
        queue->avg_command_latency_cycles = 0;

        // Initialize all commands to NO_OP
        for (uint32_t i = 0; i < CommandQueue::QUEUE_SIZE; i++) {
            queue->commands[i].type = CMD_NO_OP;
            queue->commands[i].id = i;
            queue->commands[i].timestamp = 0;
            queue->commands[i].result_code = 0;
        }
    }
}

// Helper kernel to get queue statistics
extern "C" __global__ void get_queue_stats(CommandQueue* queue, uint64_t* stats_out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        stats_out[0] = queue->commands_processed;
        stats_out[1] = queue->total_cycles;
        stats_out[2] = queue->head;
        stats_out[3] = queue->tail;
        stats_out[4] = static_cast<uint64_t>(queue->status);
        stats_out[5] = queue->idle_cycles;
        stats_out[6] = static_cast<uint64_t>(queue->current_strategy);
        stats_out[7] = queue->consecutive_commands;
        stats_out[8] = queue->consecutive_idle;
        stats_out[9] = queue->avg_command_latency_cycles;
    }
}

// Helper kernel to set polling strategy
extern "C" __global__ void set_polling_strategy(CommandQueue* queue, PollingStrategy strategy) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        queue->current_strategy = strategy;
        // Reset counters when strategy changes
        queue->consecutive_commands = 0;
        queue->consecutive_idle = 0;
    }
}
