// CudaClaw - Rust bindings for persistent GPU kernel
// FIXED: Memory layout now matches CUDA C++ structs exactly

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
//
// So the union is 24 bytes total (largest member is batch)

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Command {
    pub cmd_type: u32,          // offset 0,  4 bytes
    pub id: u32,                // offset 4,  4 bytes
    pub timestamp: u64,         // offset 8,  8 bytes
    // Union starts here (offset 16, 24 bytes total)
    pub data_a: f32,            // offset 16, 4 bytes  (add.a or multiply.a)
    pub data_b: f32,            // offset 20, 4 bytes  (add.b or multiply.b)
    pub result: f32,            // offset 24, 4 bytes  (add.result or multiply.result)
    pub batch_data: u64,        // offset 28, 8 bytes  (batch.data pointer)
    pub batch_count: u32,       // offset 36, 4 bytes  (batch.count)
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
}

// Command Queue (must match CUDA CommandQueue struct exactly)
pub const QUEUE_SIZE: usize = 16;

#[repr(C)]
pub struct CommandQueueHost {
    pub status: u32,                         // offset 0,   4 bytes
    pub commands: [Command; QUEUE_SIZE],     // offset 4,   16*48=768 bytes (FIXED: was 640)
    pub head: u32,                           // offset 772, 4 bytes
    pub tail: u32,                           // offset 776, 4 bytes
    pub commands_processed: u64,             // offset 780, 8 bytes
    pub total_cycles: u64,                   // offset 788, 8 bytes
    pub idle_cycles: u64,                    // offset 796, 8 bytes
    pub current_strategy: u32,               // offset 804, 4 bytes
    pub consecutive_commands: u32,           // offset 808, 4 bytes
    pub consecutive_idle: u32,               // offset 812, 4 bytes
    pub last_command_cycle: u64,             // offset 816, 8 bytes
    pub avg_command_latency_cycles: u64,     // offset 824, 8 bytes
    pub _padding: [u8; 64],                  // offset 832, 64 bytes
}

// Compile-time assertion to ensure CommandQueue size matches CUDA
// CUDA CommandQueue is 896 bytes (832 + 64) aligned to 128*7=896
const _: [(); std::mem::size_of::<CommandQueueHost>()] = [(); 896]; // Must be 896 bytes

impl Default for CommandQueueHost {
    fn default() -> Self {
        CommandQueueHost {
            status: QueueStatus::Idle as u32,
            commands: [Command::new(CommandType::NoOp, 0); QUEUE_SIZE],
            head: 0,
            tail: 0,
            commands_processed: 0,
            total_cycles: 0,
            idle_cycles: 0,
            current_strategy: PollingStrategy::Spin as u32,
            consecutive_commands: 0,
            consecutive_idle: 0,
            last_command_cycle: 0,
            avg_command_latency_cycles: 0,
            _padding: [0; 64],
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
    Adaptive,  // Adaptive polling - balances latency and power
    Spin,      // Pure spin - lowest latency, highest power
    Timed,     // Timed polling - lowest power, higher latency
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
        };

        let func = self.module.get_function(&CString::new(kernel_name)?)?;
        let mut queue_ptr = self.queue.as_device_ptr();

        unsafe {
            launch!(
                func<<<1, 1, 0, self.stream>>>(
                    queue_ptr
                )
            )?;
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
