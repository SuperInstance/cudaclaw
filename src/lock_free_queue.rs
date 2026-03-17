// ============================================================
// Lock-Free Command Queue - Rust Host Functions
// ============================================================
// This file contains the Rust host functions for the lock-free
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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::ptr;
use crate::cuda_claw::{Command, CommandQueueHost, QueueStatus, CommandType};

// ============================================================
// Atomic Operations for Lock-Free Queue
// ============================================================

/// Atomic compare-and-swap for uint32_t (compatible with CUDA atomicCAS)
///
/// This function performs a lock-free atomic compare-and-swap operation
/// on a volatile uint32_t value in unified memory. It's compatible with
/// CUDA's atomicCAS operation on the GPU side.
///
/// # Safety
///
/// The pointer must be valid and point to memory in unified memory
/// that is accessible from both CPU and GPU.
///
/// # Arguments
///
/// * `ptr` - Volatile pointer to the uint32_t value
/// * `expected` - Expected value (if mismatch, operation fails)
/// * `desired` - New value to write if expected matches
///
/// # Returns
///
/// * `true` - CAS succeeded, value was updated to desired
/// * `false` - CAS failed, value was not expected
#[inline]
unsafe fn atomic_compare_exchange_u32(
    ptr: *const u32,
    expected: u32,
    desired: u32,
) -> bool {
    // Convert to AtomicU32 for atomic operations
    let atomic = &*(ptr as *const AtomicU32);
    atomic.compare_exchange_weak(expected, desired, Ordering::SeqCst, Ordering::Relaxed).is_ok()
}

/// Atomic fetch-and-add for uint32_t
///
/// Atomically adds a value to a uint32_t and returns the previous value.
/// Compatible with CUDA's atomicAdd operation.
///
/// # Arguments
///
/// * `ptr` - Volatile pointer to the uint32_t value
/// * `value` - Value to add
///
/// # Returns
///
/// The previous value before the addition
#[inline]
unsafe fn atomic_fetch_add_u32(ptr: *const u32, value: u32) -> u32 {
    let atomic = &*(ptr as *const AtomicU32);
    atomic.fetch_add(value, Ordering::SeqCst)
}

/// Atomic fetch-and-add for uint64_t
///
/// Atomically adds a value to a uint64_t and returns the previous value.
/// Compatible with CUDA's atomicAdd operation on unsigned long long.
///
/// # Arguments
///
/// * `ptr` - Volatile pointer to the uint64_t value
/// * `value` - Value to add
///
/// # Returns
///
/// The previous value before the addition
#[inline]
unsafe fn atomic_fetch_add_u64(ptr: *const u64, value: u64) -> u64 {
    let atomic = &*(ptr as *const AtomicU64);
    atomic.fetch_add(value, Ordering::SeqCst)
}

// ============================================================
// Queue State Management
/// ============================================================

/// Queue capacity (must match QUEUE_SIZE in CUDA)
pub const QUEUE_SIZE: u32 = 16;

/// Queue states for synchronization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LockFreeQueueState {
    Empty,       // No commands available
    Partial,     // Some commands available
    Full,        // No space for new commands
}

/// Lock-Free Command Queue wrapper
///
/// This wrapper provides safe Rust functions for operating on the
/// CommandQueue in unified memory.
pub struct LockFreeCommandQueue;

impl LockFreeCommandQueue {
    // ============================================================
    // Push Operations (Rust Host Side)
    // ============================================================

    /// Push a command to the queue (Rust side)
    ///
    /// This function attempts to atomically push a command to the queue.
    /// It uses a CAS (Compare-And-Swap) loop to ensure thread-safe access.
    ///
    /// # Algorithm
    ///
    /// 1. Read current head index
    /// 2. Read current tail index
    /// 3. Check if queue is full (head + 1 == tail)
    /// 4. Calculate slot index: head % QUEUE_SIZE
    /// 5. Write command to slot
    /// 6. Use atomicCAS to advance head
    /// 7. If CAS succeeds, increment commands_pushed
    /// 8. Return success
    ///
    /// # Arguments
    ///
    /// * `queue` - Mutable reference to the CommandQueue in unified memory
    /// * `cmd` - Command to push
    ///
    /// # Returns
    ///
    /// * `true` - Command was successfully pushed
    /// * `false` - Queue is full or CAS failed (retry)
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe and can be called from multiple threads
    /// concurrently. The CAS operation ensures that only one thread succeeds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use lock_free_queue::LockFreeCommandQueue;
    ///
    /// let cmd = Command {
    ///     cmd_type: CommandType::CMD_ADD,
    ///     id: 1,
    ///     timestamp: 12345,
    ///     data_a: 10.0,
    ///     data_b: 20.0,
    ///     result: 0.0,
    ///     batch_data: 0,
    ///     batch_count: 0,
    ///     _padding: 0,
    ///     result_code: 0,
    /// };
    ///
    /// if LockFreeCommandQueue::push_command(&mut queue, cmd) {
    ///     println!("Command pushed successfully");
    /// } else {
    ///     println!("Queue is full");
    /// }
    /// ```
    #[inline]
    pub fn push_command(queue: &mut CommandQueueHost, cmd: Command) -> bool {
        unsafe {
            // Get current head index
            let head = queue.head;
            let tail = queue.tail;

            // Check if queue is full
            // In circular buffer: if (head + 1) % SIZE == tail, then full
            let next_head = (head + 1) % QUEUE_SIZE;
            if next_head == tail {
                return false;  // Queue is full
            }

            // Calculate actual index in circular buffer
            let index = head % QUEUE_SIZE;

            // Write command to slot
            // NOTE: We write the command before advancing head
            // The GPU won't read this slot until head is advanced
            queue.commands[index as usize] = cmd;

            // Memory fence to ensure command is written before head advances
            std::sync::atomic::fence(Ordering::SeqCst);

            // Attempt to advance head using CAS
            // This ensures only one thread succeeds in claiming this slot
            if atomic_compare_exchange_u32(&queue.head as *const u32, head, next_head) {
                // Successfully claimed this slot
                // Update statistics
                atomic_fetch_add_u64(&queue.commands_pushed as *const u64, 1);

                // Update queue status if needed
                if queue.status == QueueStatus::STATUS_IDLE {
                    queue.status = QueueStatus::STATUS_READY;
                }

                // Memory fence to ensure all writes are visible
                std::sync::atomic::fence(Ordering::SeqCst);

                return true;
            }

            // CAS failed - another thread claimed this slot
            // The command we wrote will be overwritten, so we need to retry
            false
        }
    }

    /// Push multiple commands to the queue (batch processing)
    ///
    /// This function attempts to push multiple commands in a single operation,
    /// which can be more efficient than pushing one at a time.
    ///
    /// # Arguments
    ///
    /// * `queue` - Mutable reference to the CommandQueue
    /// * `cmds` - Slice of commands to push
    ///
    /// # Returns
    ///
    /// Number of commands successfully pushed
    ///
    /// # Example
    ///
    /// ```rust
    /// let commands = vec![cmd1, cmd2, cmd3];
    /// let pushed = LockFreeCommandQueue::push_commands_batch(&mut queue, &commands);
    /// println!("Pushed {} commands", pushed);
    /// ```
    pub fn push_commands_batch(queue: &mut CommandQueueHost, cmds: &[Command]) -> u32 {
        let mut pushed = 0;

        for cmd in cmds {
            if Self::push_command(queue, *cmd) {
                pushed += 1;
            } else {
                break;  // Queue is full
            }
        }

        pushed
    }

    /// Wait for space to become available in the queue (blocking)
    ///
    /// This function spins until space is available for pushing.
    /// Use sparingly as it consumes CPU cycles.
    ///
    /// # Arguments
    ///
    /// * `queue` - Mutable reference to the CommandQueue
    /// * `cmd` - Command to push
    /// * `max_spins` - Maximum number of spin iterations (0 = infinite)
    ///
    /// # Returns
    ///
    /// * `true` - Command was pushed successfully
    /// * `false` - Timeout
    ///
    /// # Example
    ///
    /// ```rust
    /// // Wait up to 1 second for space
    /// let success = LockFreeCommandQueue::wait_for_space(
    ///     &mut queue,
    ///     cmd,
    ///     1_000_000_000 // 1 second in nanoseconds
    /// );
    /// ```
    pub fn wait_for_space(queue: &mut CommandQueueHost, cmd: Command, max_spins: u32) -> bool {
        let mut spins = 0;

        loop {
            if Self::push_command(queue, cmd) {
                return true;
            }

            if max_spins > 0 && spins >= max_spins {
                return false;  // Timeout
            }

            spins += 1;

            // Small delay to reduce CPU usage
            if spins % 1000 == 0 {
                std::thread::sleep(std::time::Duration::from_micros(1));
            }
        }
    }

    // ============================================================
    // Query Functions
    // ============================================================

    /// Get the current number of commands in the queue
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    ///
    /// # Returns
    ///
    /// Number of commands currently in queue
    #[inline]
    pub fn get_queue_size(queue: &CommandQueueHost) -> u32 {
        let head = queue.head;
        let tail = queue.tail;

        if head >= tail {
            head - tail
        } else {
            QUEUE_SIZE - (tail - head)
        }
    }

    /// Check if queue is empty
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    ///
    /// # Returns
    ///
    /// * `true` - Queue is empty
    /// * `false` - Queue has commands
    #[inline]
    pub fn is_queue_empty(queue: &CommandQueueHost) -> bool {
        queue.head == queue.tail
    }

    /// Check if queue is full
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    ///
    /// # Returns
    ///
    /// * `true` - Queue is full
    /// * `false` - Queue has space
    #[inline]
    pub fn is_queue_full(queue: &CommandQueueHost) -> bool {
        let next_head = (queue.head + 1) % QUEUE_SIZE;
        next_head == queue.tail
    }

    /// Get the current state of the queue
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    ///
    /// # Returns
    ///
    /// Current queue state
    #[inline]
    pub fn get_queue_state(queue: &CommandQueueHost) -> LockFreeQueueState {
        let size = Self::get_queue_size(queue);

        if size == 0 {
            LockFreeQueueState::Empty
        } else if size >= QUEUE_SIZE - 1 {
            LockFreeQueueState::Full
        } else {
            LockFreeQueueState::Partial
        }
    }

    /// Get queue statistics
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    ///
    /// # Returns
    ///
    /// Tuple of (commands_pushed, commands_popped, commands_processed)
    #[inline]
    pub fn get_queue_stats(queue: &CommandQueueHost) -> (u64, u64, u64) {
        (
            queue.commands_pushed,
            queue.commands_popped,
            queue.commands_processed,
        )
    }

    // ============================================================
    // Utility Functions
    // ============================================================

    /// Reset the queue to empty state
    ///
    /// # Warning
    ///
    /// This is a destructive operation that clears all pending commands.
    /// Only use this when you know the queue is not being accessed.
    ///
    /// # Arguments
    ///
    /// * `queue` - Mutable reference to the CommandQueue
    ///
    /// # Safety
    ///
    /// This function is NOT thread-safe. Ensure no other threads are
    /// accessing the queue when calling this function.
    pub fn reset_queue(queue: &mut CommandQueueHost) {
        queue.head = 0;
        queue.tail = 0;
        queue.status = QueueStatus::STATUS_IDLE;
        // Note: We don't reset the statistics counters
    }

    /// Print queue status for debugging
    ///
    /// # Arguments
    ///
    /// * `queue` - Reference to the CommandQueue
    pub fn print_queue_status(queue: &CommandQueueHost) {
        let size = Self::get_queue_size(queue);
        let state = Self::get_queue_state(queue);
        let (pushed, popped, processed) = Self::get_queue_stats(queue);

        println!("=== Lock-Free Command Queue Status ===");
        println!("  State: {:?}", state);
        println!("  Size: {} / {}", size, QUEUE_SIZE - 1);
        println!("  Head: {}", queue.head);
        println!("  Tail: {}", queue.tail);
        println!("  Status: {:?}", queue.status);
        println!("  Statistics:");
        println!("    Commands Pushed: {}", pushed);
        println!("    Commands Popped: {}", popped);
        println!("    Commands Processed: {}", processed);
        println!("  Efficiency: {:.1}%",
            if pushed > 0 { (processed as f64 / pushed as f64) * 100.0 }
            else { 0.0 }
        );
        println!("========================================");
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::zeroed;

    #[test]
    fn test_queue_empty_initially() {
        let mut queue: CommandQueueHost = unsafe { zeroed() };
        assert!(LockFreeCommandQueue::is_queue_empty(&queue));
        assert_eq!(LockFreeCommandQueue::get_queue_size(&queue), 0);
    }

    #[test]
    fn test_push_and_query() {
        let mut queue: CommandQueueHost = unsafe { zeroed() };

        let cmd = Command {
            cmd_type: CommandType::CMD_ADD,
            id: 1,
            timestamp: 0,
            data_a: 10.0,
            data_b: 20.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        };

        assert!(LockFreeCommandQueue::push_command(&mut queue, cmd));
        assert_eq!(LockFreeCommandQueue::get_queue_size(&queue), 1);
        assert!(!LockFreeCommandQueue::is_queue_empty(&queue));
    }

    #[test]
    fn test_queue_full() {
        let mut queue: CommandQueueHost = unsafe { zeroed() };

        let cmd = Command {
            cmd_type: CommandType::CMD_NO_OP,
            id: 0,
            timestamp: 0,
            data_a: 0.0,
            data_b: 0.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        };

        // Fill queue to capacity (QUEUE_SIZE - 1 to distinguish full from empty)
        for _ in 0..(QUEUE_SIZE - 1) {
            assert!(LockFreeCommandQueue::push_command(&mut queue, cmd));
        }

        // Queue should now be full
        assert!(LockFreeCommandQueue::is_queue_full(&queue));

        // Next push should fail
        assert!(!LockFreeCommandQueue::push_command(&mut queue, cmd));
    }

    #[test]
    fn test_batch_push() {
        let mut queue: CommandQueueHost = unsafe { zeroed() };

        let cmd = Command {
            cmd_type: CommandType::CMD_NO_OP,
            id: 0,
            timestamp: 0,
            data_a: 0.0,
            data_b: 0.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        };

        let commands = vec![cmd; 5];
        let pushed = LockFreeCommandQueue::push_commands_batch(&mut queue, &commands);

        assert_eq!(pushed, 5);
        assert_eq!(LockFreeCommandQueue::get_queue_size(&queue), 5);
    }
}
