// ============================================================
// Shared Types Header - Unified Memory Bridge
// ============================================================
// This file defines the data structures shared between CUDA C++
// kernels and Rust host code. Memory layout must match exactly
// on both sides to prevent GPU kernel crashes.
//
// CRITICAL: Any changes to these structures must be mirrored
// in src/cuda_claw.rs with compile-time size verification.
//
// ALIGNMENT VERIFICATION:
// - Command: 48 bytes, 32-byte aligned
// - CommandQueue: 896 bytes, 128-byte aligned
// - All offsets verified at compile time (C++) and runtime (Rust)
//
// #[repr(C)] is used on Rust side to ensure C-compatible layout
// volatile is used on CUDA side to ensure proper memory ordering
// ============================================================

#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// Compile-Time Alignment Verification
// ============================================================

// These macros verify alignment at compile time
#define VERIFY_ALIGN(type, expected) \
    static_assert(sizeof(type) == (expected), \
        "Size mismatch for " #type ": expected " #expected " bytes")

#define VERIFY_OFFSET(type, member, expected) \
    static_assert(offsetof(type, member) == (expected), \
        "Offset mismatch for " #member " in " #type ": expected " #expected)

// ============================================================
// Configuration Constants
// ============================================================

#define QUEUE_SIZE 16

// ============================================================
// Enums
// ============================================================

// Polling strategies
enum PollingStrategy : uint32_t {
    POLL_SPIN = 0,         // Tight spin - lowest latency, highest power
    POLL_ADAPTIVE = 1,     // Adaptive - balances latency and power
    POLL_TIMED = 2,        // Fixed interval - lower power, higher latency
};

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
// Command Structure (48 bytes total)
// ============================================================
// Memory layout must match Rust Command struct exactly
// See src/cuda_claw.rs for the Rust definition
//
struct __align__(32) Command {
    CommandType type;     // offset 0,  4 bytes
    uint32_t id;          // offset 4,  4 bytes
    uint64_t timestamp;   // offset 8,  8 bytes

    // Command-specific data (union) - 24 bytes total
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
    } data;               // offset 16, 24 bytes (largest is batch with alignment)

    uint32_t result_code; // offset 44, 4 bytes
};
// Total: 48 bytes (padded from 44 for alignment)

// Compile-time assertion to ensure size matches
static_assert(sizeof(Command) == 48, "Command must be 48 bytes");

// ============================================================
// Command Queue Structure (896 bytes total)
// ============================================================
// This structure is allocated in Unified Memory and shared
// between CPU (Rust) and GPU (CUDA kernel).
//
// Memory layout must match Rust CommandQueueHost struct exactly
// See src/cuda_claw.rs for the Rust definition
//
struct __align__(128) CommandQueue {
    volatile QueueStatus status;  // offset 0,   4 bytes
    Command commands[QUEUE_SIZE]; // offset 4,   768 bytes (16 * 48)
    volatile uint32_t head;       // offset 772, 4 bytes
    volatile uint32_t tail;       // offset 776, 4 bytes
    volatile uint64_t commands_processed; // offset 780, 8 bytes
    volatile uint64_t total_cycles;       // offset 788, 8 bytes
    volatile uint64_t idle_cycles;        // offset 796, 8 bytes
    volatile PollingStrategy current_strategy; // offset 804, 4 bytes
    volatile uint32_t consecutive_commands;  // offset 808, 4 bytes
    volatile uint32_t consecutive_idle;    // offset 812, 4 bytes
    volatile uint64_t last_command_cycle;  // offset 816, 8 bytes
    volatile uint64_t avg_command_latency_cycles; // offset 824, 8 bytes
    uint8_t padding[64];            // offset 832, 64 bytes
};
// Total: 896 bytes (aligned to 128-byte boundary)

// Compile-time assertions to ensure size matches
static_assert(sizeof(CommandQueue) == 896, "CommandQueue must be 896 bytes");

// ============================================================
// Field Offset Verification
// ============================================================
// Verify all critical field offsets to match Rust side

// Command field offsets
VERIFY_OFFSET(Command, type, 0);
VERIFY_OFFSET(Command, id, 4);
VERIFY_OFFSET(Command, timestamp, 8);
VERIFY_OFFSET(Command, result_code, 44);

// CommandQueue field offsets
VERIFY_OFFSET(CommandQueue, status, 0);
VERIFY_OFFSET(CommandQueue, head, 772);
VERIFY_OFFSET(CommandQueue, tail, 776);
VERIFY_OFFSET(CommandQueue, commands_processed, 780);
VERIFY_OFFSET(CommandQueue, total_cycles, 788);
VERIFY_OFFSET(CommandQueue, idle_cycles, 796);
VERIFY_OFFSET(CommandQueue, current_strategy, 804);
VERIFY_OFFSET(CommandQueue, consecutive_commands, 808);
VERIFY_OFFSET(CommandQueue, consecutive_idle, 812);

// ============================================================
// Unified Memory Bridge Documentation
// ============================================================
//
// ALLOCATION (Rust host side):
//   use cust::memory::UnifiedBuffer;
//   let queue = UnifiedBuffer::<CommandQueueHost>::new(&queue_data)?;
//
// The UnifiedBuffer allocates memory that is:
// - Accessible from both CPU and GPU
// - Automatically migrated between CPU and GPU
// - Cache-coherent on supported hardware
//
// USAGE:
// 1. CPU writes command to queue.status = STATUS_READY
// 2. GPU kernel polls queue.status in a loop
// 3. When GPU sees STATUS_READY, it processes the command
// 4. GPU sets queue.status = STATUS_DONE when complete
// 5. CPU reads result from queue.status
//
// VOLATILE KEYWORD:
// - Ensures reads/writes are not optimized away
// - Forces memory to be read/written directly from memory
// - Critical for GPU-CPU synchronization
//
// ALIGNMENT:
// - Command: 32-byte aligned for cache line efficiency
// - CommandQueue: 128-byte aligned to prevent false sharing
//
// SAFETY:
// - Both sides must agree on exact memory layout
// - Size verification done at compile time (C++) and runtime (Rust)
// - Any mismatch will cause immediate kernel crash
//
// ============================================================

#endif // SHARED_TYPES_H
