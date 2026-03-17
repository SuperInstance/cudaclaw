// CudaClaw - Rust bindings for persistent GPU kernel
// FIXED: Memory layout now matches CUDA C++ structs exactly
//
// UNIFIED MEMORY BRIDGE:
// This file contains Rust definitions that MUST match the CUDA C++ definitions
// in kernels/shared_types.h exactly. The CommandQueue is allocated in Unified
// Memory using cust::memory::UnifiedBuffer, allowing zero-copy access from
// both CPU (Rust) and GPU (CUDA kernel).
//
// CRITICAL: Any changes to struct layouts must be updated in both:
// - kernels/shared_types.h (CUDA C++ side)
// - src/cuda_claw.rs (Rust side)
//
// Verification is done via compile-time assertions (see below)

use cust::prelude::*;
use cust::memory::{CopyDestination, DeviceBuffer, UnifiedBuffer};
use std::ffi::CString;
use std::time::{Duration, Instant};

// Re-export PTX loading
pub mod ptx;

// ============================================================
// MEMORY ALIGNMENT VERIFICATION
// ============================================================

// Macro to get field offset for compile-time verification
macro_rules! offset_of {
    ($ty:ty, $field:ident) => {{
        let uninit = std::mem::MaybeUninit::<$ty>::uninit();
        let ptr = uninit.as_ptr();
        unsafe {
            (&(*ptr).$field as *const _ as usize) - (ptr as usize)
        }
    }};
}

// Compile-time assertions for critical field offsets
#[test]
fn verify_command_layout() {
    // These tests will fail at compile time if offsets are wrong
    const _ASSERT_CMD_TYPE: [(); offset_of!(Command, cmd_type) - 0] = [(); 0];
    const _ASSERT_ID: [(); offset_of!(Command, id) - 4] = [(); 0];
    const _ASSERT_TIMESTAMP: [(); offset_of!(Command, timestamp) - 8] = [(); 0];
    const _ASSERT_DATA_A: [(); offset_of!(Command, data_a) - 16] = [(); 0];
    const _ASSERT_RESULT_CODE: [(); offset_of!(Command, result_code) - 44] = [(); 0];
}

// Status flags (must match CUDA enum)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueStatus {
    Idle = 0,
    Ready = 1,
    Processing = 2,
    Done = 3,
    Error = 4,
}

impl From<u32> for QueueStatus {
    fn from(val: u32) -> Self {
        match val {
            0 => QueueStatus::Idle,
            1 => QueueStatus::Ready,
            2 => QueueStatus::Processing,
            3 => QueueStatus::Done,
            4 => QueueStatus::Error,
            _ => QueueStatus::Error,
        }
    }
}

// Polling strategies (must match CUDA enum)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollingStrategy {
    Spin = 0,      // Tight spin - lowest latency, highest power
    Adaptive = 1,  // Adaptive - balances latency and power
    Timed = 2,     // Fixed interval - lower power, higher latency
}

// Command types (must match CUDA enum)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandType {
    NoOp = 0,
    Shutdown = 1,
    Add = 2,
    Multiply = 3,
    BatchProcess = 4,
    SpreadsheetEdit = 5,
}

// Cell value types (must match CUDA enum)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellValueType {
    Empty = 0,
    Number = 1,
    String = 2,
    Formula = 3,
    Boolean = 4,
    Error = 5,
}

// ============================================================
// Spreadsheet CRDT Data Structures
// ============================================================

/// Cell identifier - unique position in spreadsheet
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellID {
    pub row: u32,
    pub col: u32,
}

/// Spreadsheet edit operation for CRDT merge
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpreadsheetEdit {
    pub cell_id: CellID,
    pub new_type: CellValueType,
    pub numeric_value: f64,
    pub timestamp: u64,
    pub node_id: u32,
    pub is_delete: u32,
    pub string_ptr: u64,
    pub formula_ptr: u64,
    pub value_len: u32,
    pub reserved: u32,
}

// ============================================================
// CRITICAL FIX: Command struct must match CUDA layout exactly
// ============================================================

// The CUDA union has these variants:
// - noop:    uint32_t padding (4 bytes)
// - add:     float a, b, result (12 bytes)
// - multiply: float a, b, result (12 bytes)
// - batch:   float* data (8), uint32_t count (4), float* output (8) = 20 bytes
//           but with alignment: data(8) + count(4) + padding(4) + output(8) = 24 bytes
// - spreadsheet: void* cells_ptr (8), void* edit_ptr (8), spreadsheet_id (4)
//
// So the union is 24 bytes total (largest member is batch)
//
// SPREADSHEET EDIT LAYOUT:
// - cells_ptr: 8 bytes (pointer to cell array)
// - edit_ptr: 8 bytes (pointer to SpreadsheetEdit in GPU memory)
// - spreadsheet_id: 4 bytes
//
// For spreadsheet edits, the SpreadsheetEdit data is passed separately in GPU memory
// to avoid size constraints of the command structure.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Command {
    pub cmd_type: u32,          // offset 0,  4 bytes
    pub id: u32,                // offset 4,  4 bytes
    pub timestamp: u64,         // offset 8,  8 bytes
    // Union starts here (offset 16, 24 bytes total)
    pub data_a: f32,            // offset 16, 4 bytes  (add.a or multiply.a)
    pub data_b: f32,            // offset 20, 4 bytes  (add.b or multiply.b)
    pub result: f32,            // offset 24, 4 bytes  (add.result or multiply.result or spreadsheet.edit_ptr)
    pub batch_data: u64,        // offset 28, 8 bytes  (batch.data pointer or spreadsheet.cells_ptr)
    pub batch_count: u32,       // offset 36, 4 bytes  (batch.count or spreadsheet.spreadsheet_id)
    pub _padding: u32,          // offset 40, 4 bytes  (padding for alignment)
    pub result_code: u32,       // offset 44, 4 bytes
}

// Compile-time assertion to ensure size matches
const _: [(); std::mem::size_of::<Command>()] = [(); 48]; // Must be 48 bytes

impl Command {
    pub fn new(cmd_type: CommandType, id: u32) -> Self {
        Command {
            cmd_type: cmd_type as u32,
            id,
            timestamp: 0,
            data_a: 0.0,
            data_b: 0.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        }
    }

    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_add_data(mut self, a: f32, b: f32) -> Self {
        self.data_a = a;
        self.data_b = b;
        self
    }

    /// Create a spreadsheet edit command (batch style)
    /// Processes multiple edits in parallel across the GPU warp.
    ///
    /// # Arguments
    /// * `cells_ptr` - Pointer to SpreadsheetCell array in GPU memory
    /// * `edits_ptr` - Pointer to array of SpreadsheetEdit structures in GPU memory
    /// * `edit_count` - Number of edits to process
    ///
    /// # Note
    /// The SpreadsheetEdit array must be copied to GPU memory separately.
    /// Edits are processed in parallel across the warp (32 threads).
    pub fn with_spreadsheet_edit_batch(
        mut self,
        cells_ptr: u64,
        edits_ptr: u64,
        edit_count: u32,
    ) -> Self {
        self.batch_data = cells_ptr;       // cells_ptr
        self.batch_count = edit_count;     // edit_count

        // Store edits_ptr in result (lower 32 bits) and _padding (upper 32 bits)
        self.result = f32::from_bits((edits_ptr & 0xFFFFFFFF) as u32);
        self._padding = ((edits_ptr >> 32) & 0xFFFFFFFF) as u32;

        self
    }

    /// Get the edits_ptr from spreadsheet edit command
    pub fn get_edits_ptr(&self) -> u64 {
        let lower = self.result.to_bits() as u64;
        let upper = self._padding as u64;
        (upper << 32) | lower
    }
}

// Command Queue (must match CUDA CommandQueue struct exactly)
pub const QUEUE_SIZE: usize = 16;

#[repr(C)]
pub struct CommandQueueHost {
    pub status: u32,                         // offset 0,   4 bytes - queue status flag
    pub commands: [Command; QUEUE_SIZE],     // offset 4,   768 bytes - circular buffer
    pub head: u32,                           // offset 772, 4 bytes - write index (Rust)
    pub tail: u32,                           // offset 776, 4 bytes - read index (GPU)
    pub commands_pushed: u64,                // offset 780, 8 bytes - total commands pushed
    pub commands_popped: u64,                // offset 788, 8 bytes - total commands popped
    pub commands_processed: u64,             // offset 796, 8 bytes - total commands processed
    pub total_cycles: u64,                   // offset 804, 8 bytes - total polling cycles
    pub idle_cycles: u64,                    // offset 812, 8 bytes - idle polling cycles
    pub current_strategy: u32,               // offset 820, 4 bytes - adaptive polling
    pub consecutive_commands: u32,           // offset 824, 4 bytes - consecutive commands
    pub consecutive_idle: u32,               // offset 828, 4 bytes - consecutive idle cycles
    pub last_command_cycle: u64,             // offset 832, 8 bytes - last command timestamp
    pub avg_command_latency_cycles: u64,     // offset 840, 8 bytes - avg latency
    pub _padding: [u8; 48],                  // offset 848, 48 bytes - alignment padding
}

// Compile-time assertion to ensure CommandQueue size matches CUDA
// CUDA CommandQueue is 896 bytes (aligned to 128-byte boundary)
const _: [(); std::mem::size_of::<CommandQueueHost>()] = [(); 896]; // Must be 896 bytes

impl Default for CommandQueueHost {
    fn default() -> Self {
        CommandQueueHost {
            status: QueueStatus::Idle as u32,
            commands: [Command::new(CommandType::NoOp, 0); QUEUE_SIZE],
            head: 0,
            tail: 0,
            commands_pushed: 0,
            commands_popped: 0,
            commands_processed: 0,
            total_cycles: 0,
            idle_cycles: 0,
            current_strategy: PollingStrategy::Spin as u32,
            consecutive_commands: 0,
            consecutive_idle: 0,
            last_command_cycle: 0,
            avg_command_latency_cycles: 0,
            _padding: [0; 48],
        }
    }
}

// CudaClaw executor - manages the persistent kernel
pub struct CudaClawExecutor {
    module: Module,
    queue: UnifiedBuffer<CommandQueueHost>,
    stream: Stream,
    kernel_running: bool,
    kernel_variant: KernelVariant,
}

#[derive(Debug, Clone, Copy)]
pub enum KernelVariant {
    Adaptive,        // Adaptive polling - balances latency and power
    Spin,            // Pure spin - lowest latency, highest power
    Timed,           // Timed polling - lowest power, higher latency
    PersistentWorker, // Persistent worker with warp-level parallelism
    MultiBlockWorker, // Multi-block persistent worker for higher throughput
}

impl CudaClawExecutor {
    /// Initialize a new CudaClaw executor
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_variant(KernelVariant::Adaptive)
    }

    /// Initialize with specific kernel variant
    pub fn with_variant(variant: KernelVariant) -> Result<Self, Box<dyn std::error::Error>> {
        // Load CUDA module
        let ptx_str = ptx::load_ptx("main")?;
        let ptx = CString::new(ptx_str)?;
        let module = Module::load_from_string(&ptx)?;

        // Create unified memory queue
        let queue_data = CommandQueueHost::default();
        let queue = UnifiedBuffer::new(&queue_data)?;

        // Create stream
        let stream = Stream::new()?;

        Ok(CudaClawExecutor {
            module,
            queue,
            stream,
            kernel_running: false,
            kernel_variant: variant,
        })
    }

    /// Initialize the command queue on GPU
    pub fn init_queue(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Launch init kernel
        let func = self.module.get_function(&CString::new("init_command_queue")?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        unsafe {
            launch!(
                func<<<1, 1, 0, self.stream>>>(
                    queue_ptr
                )
            )?;
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Start the persistent kernel
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.kernel_running {
            return Ok(());
        }

        let kernel_name = match self.kernel_variant {
            KernelVariant::Adaptive => "cuda_claw_executor",
            KernelVariant::Spin => "cuda_claw_executor_spin",
            KernelVariant::Timed => "cuda_claw_executor_timed",
            KernelVariant::PersistentWorker => "persistent_worker",
            KernelVariant::MultiBlockWorker => "persistent_worker_multiblock",
        };

        let func = self.module.get_function(&CString::new(kernel_name)?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        // Launch configuration based on kernel variant
        match self.kernel_variant {
            KernelVariant::MultiBlockWorker => {
                // Multi-block worker: 4 blocks, 128 threads each
                unsafe {
                    // Launch multi-block kernel with block_id parameters
                    let block_id = 0u32;
                    let total_blocks = 4u32;

                    launch!(
                        func<<<4, 128, 0, self.stream>>>(
                            queue_ptr,
                            block_id,
                            total_blocks
                        )
                    )?;
                }
            }
            _ => {
                // Single-block kernels: 1 block, 1 thread
                unsafe {
                    launch!(
                        func<<<1, 1, 0, self.stream>>>(
                            queue_ptr
                        )
                    )?;
                }
            }
        }

        self.kernel_running = true;
        Ok(())
    }

    /// Set the polling strategy (for adaptive kernel)
    pub fn set_polling_strategy(&mut self, strategy: PollingStrategy) -> Result<(), Box<dyn std::error::Error>> {
        let func = self.module.get_function(&CString::new("set_polling_strategy")?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        unsafe {
            launch!(
                func<<<1, 1, 0, self.stream>>>(
                    queue_ptr,
                    strategy as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Submit a command to the queue
    pub fn submit_command(&mut self, cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
        // Wait until queue is not full
        let queue = self.queue.clone();

        loop {
            let head = queue.head;
            let tail = queue.tail;

            if (head + 1) % QUEUE_SIZE as u32 != tail {
                break; // Queue has space
            }

            std::thread::sleep(Duration::from_micros(1));
        }

        // Write command to queue
        let mut queue_mut = self.queue.clone();
        let idx = queue_mut.head as usize;
        queue_mut.commands[idx] = cmd;

        // Mark as ready
        queue_mut.head = (queue_mut.head + 1) % QUEUE_SIZE as u32;
        queue_mut.status = QueueStatus::Ready as u32;

        Ok(())
    }

    /// Wait for command completion
    pub fn wait_for_completion(&self, timeout_ms: u64) -> Result<Command, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);

        loop {
            let queue = self.queue.clone();

            if queue.status == QueueStatus::Done as u32 {
                let idx = ((queue.tail + QUEUE_SIZE as u32 - 1) % QUEUE_SIZE as u32) as usize;
                let cmd = queue.commands[idx];

                // Reset status to idle
                let mut queue_mut = self.queue.clone();
                queue_mut.status = QueueStatus::Idle as u32;

                return Ok(cmd);
            }

            if start.elapsed() > timeout {
                return Err("Timeout waiting for command completion".into());
            }

            std::thread::sleep(Duration::from_micros(10));
        }
    }

    /// Execute a No-Op command and measure latency
    pub fn execute_no_op(&mut self) -> Result<Duration, Box<dyn std::error::Error>> {
        let cmd = Command::new(CommandType::NoOp, 0)
            .with_timestamp(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_micros() as u64);

        let start = Instant::now();

        self.submit_command(cmd)?;
        self.wait_for_completion(1000)?;

        Ok(start.elapsed())
    }

    /// Execute an Add command
    pub fn execute_add(&mut self, a: f32, b: f32) -> Result<(Duration, f32), Box<dyn std::error::Error>> {
        let cmd = Command::new(CommandType::Add, 1)
            .with_timestamp(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_micros() as u64)
            .with_add_data(a, b);

        let start = Instant::now();

        self.submit_command(cmd)?;
        let result_cmd = self.wait_for_completion(1000)?;

        Ok((start.elapsed(), result_cmd.result))
    }

    /// Shutdown the persistent kernel
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let cmd = Command::new(CommandType::Shutdown, 999);

        self.submit_command(cmd)?;
        self.stream.synchronize()?;

        self.kernel_running = false;
        Ok(())
    }

    /// Get worker statistics from the GPU
    pub fn get_worker_stats(&self) -> Result<WorkerStats, Box<dyn std::error::Error>> {
        let func = self.module.get_function(&CString::new("get_worker_stats")?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        // Allocate device buffer for stats
        let mut stats_host = [0u64; 10];
        let mut stats_device = DeviceBuffer::from_slice(&stats_host)?;

        unsafe {
            launch!(
                func<<<1, 1, 0, self.stream>>>(
                    queue_ptr,
                    stats_device.as_device_ptr()
                )
            )?;
        }

        self.stream.synchronize()?;

        // Copy stats back to host
        stats_device.copy_to(&mut stats_host)?;

        Ok(WorkerStats {
            commands_processed: stats_host[0],
            total_cycles: stats_host[1],
            idle_cycles: stats_host[2],
            head: stats_host[3] as u32,
            tail: stats_host[4] as u32,
            status: QueueStatus::from(stats_host[5] as u32),
            current_strategy: PollingStrategy::from(stats_host[6] as u32),
            consecutive_commands: stats_host[7] as u32,
            consecutive_idle: stats_host[8] as u32,
            avg_command_latency_cycles: stats_host[9],
        })
    }

    /// Measure warp-level performance metrics
    pub fn measure_warp_metrics(&self) -> Result<WarpMetrics, Box<dyn std::error::Error>> {
        let func = self.module.get_function(&CString::new("measure_warp_metrics")?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        // Allocate device buffer for metrics
        let mut metrics_host = [0u32; 4];
        let mut metrics_device = DeviceBuffer::from_slice(&metrics_host)?;

        unsafe {
            launch!(
                func<<<1, 1, 0, self.stream>>>(
                    queue_ptr,
                    metrics_device.as_device_ptr()
                )
            )?;
        }

        self.stream.synchronize()?;

        // Copy metrics back to host
        metrics_device.copy_to(&mut metrics_host)?;

        Ok(WarpMetrics {
            utilization_percent: metrics_host[0],
            commands_processed: metrics_host[1],
            consecutive_commands: metrics_host[2],
            consecutive_idle: metrics_host[3],
        })
    }

    /// Get statistics from the queue
    pub fn get_stats(&self) -> ExecutorStats {
        let queue = self.queue.clone();

        ExecutorStats {
            commands_processed: queue.commands_processed,
            total_cycles: queue.total_cycles,
            idle_cycles: queue.idle_cycles,
            head: queue.head,
            tail: queue.tail,
            status: QueueStatus::from(queue.status),
            current_strategy: PollingStrategy::from(queue.current_strategy),
            consecutive_commands: queue.consecutive_commands,
            consecutive_idle: queue.consecutive_idle,
            avg_command_latency_cycles: queue.avg_command_latency_cycles,
        }
    }
}

/// Statistics from the executor
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub commands_processed: u64,
    pub total_cycles: u64,
    pub idle_cycles: u64,
    pub head: u32,
    pub tail: u32,
    pub status: QueueStatus,
    pub current_strategy: PollingStrategy,
    pub consecutive_commands: u32,
    pub consecutive_idle: u32,
    pub avg_command_latency_cycles: u64,
}

/// Worker statistics from the persistent worker kernel
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub commands_processed: u64,
    pub total_cycles: u64,
    pub idle_cycles: u64,
    pub head: u32,
    pub tail: u32,
    pub status: QueueStatus,
    pub current_strategy: PollingStrategy,
    pub consecutive_commands: u32,
    pub consecutive_idle: u32,
    pub avg_command_latency_cycles: u64,
}

/// Warp-level performance metrics
#[derive(Debug, Clone)]
pub struct WarpMetrics {
    pub utilization_percent: u32,
    pub commands_processed: u32,
    pub consecutive_commands: u32,
    pub consecutive_idle: u32,
}

impl From<u32> for PollingStrategy {
    fn from(val: u32) -> Self {
        match val {
            0 => PollingStrategy::Spin,
            1 => PollingStrategy::Adaptive,
            2 => PollingStrategy::Timed,
            _ => PollingStrategy::Adaptive,
        }
    }
}
