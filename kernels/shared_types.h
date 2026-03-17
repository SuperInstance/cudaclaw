// ============================================================
// Shared Types Header - Binary Interface for Rust-GPU Communication
// ============================================================
// This file defines the exact binary interface between Rust host code
// and CUDA GPU kernels. Memory layout MUST match exactly on both sides.
//
// CRITICAL REQUIREMENTS:
// - All structs use __align__(16) for 64-bit system compatibility
// - volatile keyword ensures proper memory ordering across PCIe
// - Union layout must match Rust's enum representation
// - Any changes require updates to src/bridge.rs and src/cuda_claw.rs
//
// ALIGNMENT GUARANTEES:
// - 16-byte alignment matches Rust's default struct alignment
// - Prevents misaligned access penalties on GPU
// - Ensures cache-line efficiency
//
// ============================================================

#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// CommandType Enum
// ============================================================
// Defines the types of commands that can be sent from Rust to GPU
// Must match Rust's CommandType enum exactly

enum CommandType : uint32_t {
    NOOP = 0,          // No-operation - for latency testing
    EDIT_CELL = 1,     // Edit a single cell with new value
    SYNC_CRDT = 2,     // Synchronize CRDT state between nodes
    SHUTDOWN = 3       // Signal GPU to shut down gracefully
};

// ============================================================
// Command Struct - 48 bytes total, 16-byte aligned
// ============================================================
// Represents a single command in the queue
// Uses union for payload to minimize memory footprint

struct __align__(16) Command {
    CommandType type;           // offset 0,  4 bytes  - Command type
    uint32_t _padding;          // offset 4,  4 bytes  - Alignment padding

    // Union for command-specific data - 40 bytes
    union {
        // For NOOP commands - no data needed
        struct {
            uint8_t _noop_padding[40];
        } noop;

        // For EDIT_CELL commands - cell identifier and new value
        struct {
            uint32_t cell_id;    // offset 8,  4 bytes  - Cell index to edit
            float value;         // offset 12, 4 bytes  - New value for cell
            uint8_t _edit_padding[32];
        } edit_cell;

        // For SYNC_CRDT commands - CRDT synchronization data
        struct {
            uint32_t node_id;    // offset 8,  4 bytes  - Source node ID
            uint64_t timestamp;  // offset 12, 8 bytes  - Sync timestamp
            uint32_t vector_size;// offset 20, 4 bytes  - CRDT vector size
            uint8_t _sync_padding[24];
        } sync_crdt;

        // For SHUTDOWN commands - optional shutdown code
        struct {
            uint32_t exit_code;   // offset 8,  4 bytes  - Shutdown exit code
            uint8_t _shutdown_padding[32];
        } shutdown;
    } data;                      // offset 8,  40 bytes total
};
// Total: 48 bytes (16-byte aligned)

// Compile-time size verification
static_assert(sizeof(Command) == 48, "Command must be 48 bytes");
static_assert(alignof(Command) == 16, "Command must be 16-byte aligned");

// ============================================================
// CommandQueue Struct - 16-byte aligned
// ============================================================
// Main communication structure between Rust and GPU
// Uses lock-free ring buffer pattern with volatile indices

struct __align__(16) CommandQueue {
    // ============================================================
    // RING BUFFER
    // ============================================================
    // Fixed-size circular buffer for command storage
    // Total size: 1024 commands × 48 bytes = 48 KB
    Command buffer[1024];          // offset 0,    48,992 bytes - Command ring buffer

    // ============================================================
    // QUEUE INDICES (volatile for cross-CPU/GPU visibility)
    // ============================================================
    volatile uint32_t head;         // offset 48,992, 4 bytes - Write index (Rust writes)
    volatile uint32_t tail;         // offset 48,996, 4 bytes - Read index (GPU reads)

    // ============================================================
    // CONTROL FLAGS (volatile)
    // ============================================================
    volatile bool is_running;       // offset 49,000, 1 byte   - Kernel running flag
    uint8_t _padding[3];            // offset 49,001, 3 bytes   - Alignment padding

    // ============================================================
    // STATISTICS (optional, for monitoring)
    // ============================================================
    volatile uint64_t commands_sent;     // offset 49,004, 8 bytes - Total commands sent
    volatile uint64_t commands_processed; // offset 49,012, 8 bytes - Total commands processed
    uint8_t _stats_padding[8];      // offset 49,020, 8 bytes   - Future expansion
};

// Total size: 49,028 bytes (aligned to 16-byte boundary)
// Compile-time verification
static_assert(sizeof(CommandQueue) == 49028, "CommandQueue size mismatch");
static_assert(alignof(CommandQueue) == 16, "CommandQueue must be 16-byte aligned");

// ============================================================
// FIELD OFFSET VERIFICATION
// ============================================================
// Verify critical field offsets to ensure Rust/CUDA compatibility

// Command field offsets
static_assert(offsetof(Command, type) == 0, "Command type offset mismatch");
static_assert(offsetof(Command, data) == 8, "Command data offset mismatch");

// CommandQueue field offsets
static_assert(offsetof(CommandQueue, buffer) == 0, "CommandQueue buffer offset mismatch");
static_assert(offsetof(CommandQueue, head) == 48992, "CommandQueue head offset mismatch");
static_assert(offsetof(CommandQueue, tail) == 48996, "CommandQueue tail offset mismatch");
static_assert(offsetof(CommandQueue, is_running) == 49000, "CommandQueue is_running offset mismatch");

// ============================================================
// MEMORY LAYOUT DOCUMENTATION
// ============================================================
//
// COMMAND STRUCTURE (48 bytes):
// ┌─────────────────────────────────────────────────────────┐
// │ offset  │ field           │ size  │ description           │
// ├─────────────────────────────────────────────────────────┤
// │ 0       │ type            │ 4     │ CommandType enum     │
// │ 4       │ padding         │ 4     │ Alignment padding    │
// │ 8       │ data.union      │ 40    │ Command-specific data │
// │         │                 │       │                       │
// │         │ EDIT_CELL:      │       │                       │
// │         │  - cell_id      │ 4     │ Cell index            │
// │         │  - value        │ 4     │ New cell value        │
// │         │  - padding      │ 32    │ Alignment             │
// │         │                 │       │                       │
// │         │ SYNC_CRDT:      │       │                       │
// │         │  - node_id      │ 4     │ Source node           │
// │         │  - timestamp    │ 8     │ Sync timestamp        │
// │         │  - vector_size  │ 4     │ CRDT vector size      │
// │         │  - padding      │ 24    │ Alignment             │
// └─────────────────────────────────────────────────────────┘
//
// COMMAND QUEUE STRUCTURE (49,028 bytes):
// ┌─────────────────────────────────────────────────────────┐
// │ offset      │ field              │ size    │ description │
// ├─────────────────────────────────────────────────────────┤
// │ 0           │ buffer[1024]       │ 48,992  │ Ring buffer │
// │ 48,992      │ head               │ 4       │ Write index │
// │ 48,996      │ tail               │ 4       │ Read index  │
// │ 49,000      │ is_running         │ 1       │ Running flag│
// │ 49,001      │ padding            │ 3       │ Alignment   │
// │ 49,004      │ commands_sent      │ 8       │ Statistics  │
// │ 49,012      │ commands_processed │ 8       │ Statistics  │
// │ 49,020      │ padding            │ 8       │ Reserved    │
// └─────────────────────────────────────────────────────────┘
//
// ============================================================
// USAGE PATTERNS
// ============================================================
//
// RUST (CPU) SIDE - Command Submission:
//   1. Write command to queue.buffer[queue.head]
//   2. volatile_write(queue.head++)  // Signal GPU
//   3. Set queue.is_running = true to start kernel
//
// GPU (CUDA) SIDE - Command Processing:
//   1. __threadfence_system()  // Ensure we see latest writes
//   2. Read queue.head and queue.tail (volatile reads)
//   3. if (head != tail) {
//   4.     Process queue.buffer[tail]
//   5.     volatile_write(queue.tail++)  // Advance tail
//   6. }
//   7. Check queue.is_running for shutdown
//
// SHUTDOWN SEQUENCE:
//   1. Rust: queue.is_running = false
//   2. GPU: Detects is_running == false
//   3. GPU: Exits persistent loop
//   4. Rust: Waits for kernel completion
//
// ============================================================
// ALIGNMENT REQUIREMENTS
// ============================================================
//
// Why __align__(16)?
// - Matches Rust's default struct alignment on 64-bit systems
// - Prevents misaligned access penalties on GPU
// - Ensures cache-line efficiency (64-byte cache lines)
// - Required for proper atomic operations
//
// Volatile Keyword Usage:
// - head/tail: Both CPU and GPU must see latest values immediately
// - is_running: Shutdown signal must propagate immediately
// - statistics: Counters must be visible across PCIe bus
//
// Memory Ordering:
// - CPU uses volatile writes (ptr::write_volatile in Rust)
// - GPU uses volatile reads with __threadfence_system()
// - Ensures proper ordering across PCIe bus
//
// ============================================================
// RUST COMPATIBILITY
// ============================================================
//
// The Rust side must use #[repr(C, align(16))] on matching structs:
//
// #[repr(C, align(16))]
// pub struct Command {
//     pub cmd_type: u32,
//     pub _padding: u32,
//     pub data: CommandData,  // Union representation
// }
//
// #[repr(C, align(16))]
// pub struct CommandQueue {
//     pub buffer: [Command; 1024],
//     pub head: u32,           // volatile on CUDA side
//     pub tail: u32,           // volatile on CUDA side
//     pub is_running: bool,    // volatile on CUDA side
//     pub _padding: [u8; 3],
//     pub commands_sent: u64,
//     pub commands_processed: u64,
//     pub _stats_padding: [u8; 8],
// }
//
// ============================================================

#endif // SHARED_TYPES_H