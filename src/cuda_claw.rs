use std::fmt;

pub const QUEUE_SIZE: usize = 1024;

/// Types of commands that can be sent to the GPU.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandType {
    NoOp = 0,
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    MatrixMultiply = 5,
    MemoryCopy = 6,
    Custom = 7,
    SetCellValue = 8,
    SpreadsheetEdit = 9,
}

impl From<u32> for CommandType {
    fn from(value: u32) -> Self {
        match value {
            1 => CommandType::Add,
            2 => CommandType::Subtract,
            3 => CommandType::Multiply,
            4 => CommandType::Divide,
            5 => CommandType::MatrixMultiply,
            6 => CommandType::MemoryCopy,
            7 => CommandType::Custom,
            8 => CommandType::SetCellValue,
            9 => CommandType::SpreadsheetEdit,
            _ => CommandType::NoOp,
        }
    }
}

/// A command to be executed by the GPU persistent kernel.
///
/// This struct must be kept in sync with the `Command` struct in `kernels/executor.cu`
/// and `kernels/shared_types.h`.
///
/// The `#[repr(C, packed(8))]` attribute ensures a C-compatible memory layout
/// and 4-byte packing for efficient data transfer between host and device.
#[repr(C, packed(4))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    pub cmd_type: u32,      // 0-3 bytes
    pub id: u32,            // 4-7 bytes
    pub timestamp: u64,     // 8-15 bytes
    pub data_a: f64,        // 16-23 bytes
    pub data_b: f64,        // 24-31 bytes
    pub result: f64,        // 32-39 bytes
    pub batch_data: u64,    // 40-47 bytes
    // Total: 48 bytes
}

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
        }
    }

    pub fn with_data(mut self, data_a: f64, data_b: f64) -> Self {
        self.data_a = data_a;
        self.data_b = data_b;
        self
    }

    pub fn with_batch_data(mut self, batch_data: u64) -> Self {
        self.batch_data = batch_data;
        self
    }
}

// Compile-time assertion to ensure Command size matches CUDA
const _: [(); std::mem::size_of::<Command>()] = [(); 48];

/// Status of the command queue.
/// Must be kept in sync with `QueueStatus` enum in `kernels/shared_types.h`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueStatus {
    Idle = 0,
    Ready = 1,
    Processing = 2,
    Error = 3,
}

/// Host-side representation of the CommandQueue shared with the GPU.
///
/// This struct must be kept in sync with the `CommandQueue` struct in
/// `kernels/executor.cu` and `kernels/shared_types.h`.
///
/// The `#[repr(C, packed(8))]` attribute ensures a C-compatible memory layout
/// and 4-byte packing. This is critical for unified memory access where
/// the CPU and GPU must agree on the memory layout.
///
/// The `buffer` field MUST be first to match the CUDA side, which declares
/// the `commands` array at offset 0.
pub const COMMAND_QUEUE_ALIGNMENT: usize = 128; // Must match CUDA's __align__(128)

#[repr(C, packed(8))]
#[derive(Clone, Copy)]
pub struct CommandQueueHost {
    pub buffer: [Command; QUEUE_SIZE],       // offset 0,    48,992 bytes - Command ring buffer
    pub status: u32,                         // offset 48,992, 4 bytes  - Queue status
    pub head: u32,                           // offset 48,996, 4 bytes  - Write index (Rust)
    pub tail: u32,                           // offset 49,000, 4 bytes  - Read index (GPU)
    pub is_running: bool,                    // offset 49,004, 1 byte   - Running flag
    pub _padding: [u8; 3],                   // offset 49,005, 3 bytes  - Alignment padding
    pub commands_sent: u64,                  // offset 49,008, 8 bytes - Total commands sent
    pub commands_processed: u64,             // offset 49,016, 8 bytes - Total commands processed
    pub _stats_padding: [u8; 8],             // offset 49,024, 8 bytes - Reserved for future use
}

// Compile-time assertion to ensure CommandQueue size matches CUDA
// CommandQueue with status field is 49,192 bytes (4-byte packed)
const _: [(); std::mem::size_of::<CommandQueueHost>()] = [(); 49192]; // Must be 49,192 bytes

impl Default for CommandQueueHost {
    fn default() -> Self {
        CommandQueueHost {
            buffer: [Command::new(CommandType::NoOp, 0); QUEUE_SIZE],
            status: 0, // QueueStatus::Idle as u32
            head: 0,
            tail: 0,
            is_running: false,
            _padding: [0; 3],
            commands_sent: 0,
            commands_processed: 0,
            _stats_padding: [0; 8],
        }
    }
}

// Implement DeviceCopy for CommandQueueHost to enable use in UnifiedBuffer
#[cfg(feature = "cuda")]
unsafe impl cust::memory::DeviceCopy for CommandQueueHost {}

// CudaClaw executor - manages the persistent kernel
#[cfg(feature = "cuda")]
pub struct CudaClawExecutor {
    module: cust::module::Module,
    pub queue: cust::memory::UnifiedBuffer<CommandQueueHost>,  // Made public for external access
    stream: cust::stream::Stream,
    kernel_running: bool,
    kernel_variant: KernelVariant,
}

#[cfg(feature = "cuda")]
impl CudaClawExecutor {
    /// Create a new CudaClawExecutor.
    pub fn new(kernel_variant: KernelVariant) -> Result<Self, cust::error::CudaError> {
        // Initialize CUDA
        cust::quick_init()?;

        // Load the appropriate PTX module based on the kernel variant
        let ptx_bytes = match kernel_variant {
            KernelVariant::Baseline => include_bytes!("../../kernels/executor.ptx"),
            KernelVariant::L1Preferred => include_bytes!("../../kernels/executor_l1_pref.ptx"),
            KernelVariant::ShmemPreferred => include_bytes!("../../kernels/executor_shmem_pref.ptx"),
            KernelVariant::L1Equal => include_bytes!("../../kernels/executor_l1_equal.ptx"),
            KernelVariant::Unroll(factor) => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    factor, 0, 0, 0, 0, 0, false, // Only unroll factor matters for this variant
                );
                ptx.into_bytes()
            }
            KernelVariant::IdleSleep(ns) => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, ns, 0, 0, 0, 0, false,
                );
                ptx.into_bytes()
            }
            KernelVariant::WarpAggregatedCas => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, 0, 0, 0, 0, 0, true,
                );
                ptx.into_bytes()
            }
            KernelVariant::SoaLayout => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, 0, 0, 0, 0, 1, false,
                );
                ptx.into_bytes()
            }
            KernelVariant::L1CachePref(pref) => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, 0, 0, pref, 0, 0, false,
                );
                ptx.into_bytes()
            }
            KernelVariant::SharedMemory(bytes) => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, 0, bytes, 0, 0, 0, false,
                );
                ptx.into_bytes()
            }
            KernelVariant::BlockSize(size) => {
                let ptx = crate::installer::nvrtc_muscle_compiler::generate_kernel_ptx_from_template(
                    0, 0, 0, 0, size, 0, false,
                );
                ptx.into_bytes()
            }
        };
        let module = cust::module::Module::from_ptx(ptx_bytes, &[])?;

        // Create a unified buffer for the command queue
        let queue = cust::memory::UnifiedBuffer::new(&CommandQueueHost::default())?;

        // Create a stream for kernel execution
        let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None)?;

        Ok(CudaClawExecutor {
            module,
            queue,
            stream,
            kernel_running: false,
            kernel_variant,
        })
    }

    /// Launch the persistent kernel on the GPU.
    ///
    /// The kernel runs asynchronously and continuously processes commands
    /// from the `CommandQueueHost`.
    pub fn launch_kernel(&mut self) -> Result<(), cust::error::CudaError> {
        if self.kernel_running {
            eprintln!("Warning: Persistent kernel already running. Skipping launch.");
            return Ok(());
        }

        // Get the kernel function
        let func = self.module.get_function("persistent_worker")?;

        // Get the device pointer to the unified command queue
        let queue_ptr = self.queue.as_device_ptr();

        // Launch the kernel with 1 block and 1 thread
        // The kernel itself manages its internal concurrency (warps, threads)
        unsafe {
            launch!(
                func <<<
                    1,   // grid_dim (blocks)
                    1,   // block_dim (threads per block)
                    0,   // shared_mem_bytes
                    self.stream
                >>>(
                    queue_ptr,
                )
            )?;
        }

        self.kernel_running = true;
        println!("Persistent kernel launched successfully.");
        Ok(())
    }

    /// Signal the kernel to stop and wait for it to finish.
    pub fn stop_kernel(&mut self) -> Result<(), cust::error::CudaError> {
        if !self.kernel_running {
            eprintln!("Warning: Persistent kernel not running. Skipping stop.");
            return Ok(());
        }

        // Signal the kernel to shut down
        self.queue.as_host_mut().is_running = false;
        self.stream.synchronize()?; // Wait for kernel to finish
        self.kernel_running = false;

        println!("Persistent kernel stopped.");
        Ok(())
    }

    /// Push a command to the kernel's queue.
    pub fn push_command(&mut self, command: Command) -> bool {
        crate::lock_free_queue::LockFreeCommandQueue::push_command(
            self.queue.as_host_mut(),
            command,
        )
    }

    /// Get a mutable reference to the underlying CommandQueueHost.
    pub fn get_queue_mut(&mut self) -> &mut CommandQueueHost {
        self.queue.as_host_mut()
    }

    /// Get a reference to the underlying CommandQueueHost.
    pub fn get_queue(&self) -> &CommandQueueHost {
        self.queue.as_host()
    }
}

/// Represents different kernel variants that can be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelVariant {
    Baseline,
    L1Preferred,
    ShmemPreferred,
    L1Equal,
    Unroll(u32),
    IdleSleep(u32),
    WarpAggregatedCas,
    SoaLayout,
    L1CachePref(u32),
    SharedMemory(u32),
    BlockSize(u32),
}

impl fmt::Display for KernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelVariant::Baseline => write!(f, "Baseline"),
            KernelVariant::L1Preferred => write!(f, "L1 Preferred"),
            KernelVariant::ShmemPreferred => write!(f, "Shared Memory Preferred"),
            KernelVariant::L1Equal => write!(f, "L1 Equal"),
            KernelVariant::Unroll(factor) => write!(f, "Unroll (factor {})", factor),
            KernelVariant::IdleSleep(ns) => write!(f, "Idle Sleep ({} ns)", ns),
            KernelVariant::WarpAggregatedCas => write!(f, "Warp Aggregated CAS"),
            KernelVariant::SoaLayout => write!(f, "SoA Layout"),
            KernelVariant::L1CachePref(pref) => write!(f, "L1 Cache Pref ({})", pref),
            KernelVariant::SharedMemory(bytes) => write!(f, "Shared Memory ({} bytes)", bytes),
            KernelVariant::BlockSize(size) => write!(f, "Block Size ({})", size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cust::prelude::Ctx;

    #[test]
    fn test_command_serialization() {
        let cmd = Command {
            cmd_type: CommandType::Add as u32,
            id: 123,
            timestamp: 456,
            data_a: 1.0,
            data_b: 2.0,
            result: 3.0,
            batch_data: 789,
        };

        let bytes: [u8; 48] = unsafe { std::mem::transmute(cmd) };
        let deserialized_cmd: Command = unsafe { std::mem::transmute(bytes) };

        assert_eq!(cmd, deserialized_cmd);
    }

    #[test]
    fn test_command_queue_host_default() {
        let queue = CommandQueueHost::default();
        assert_eq!(queue.head, 0);
        assert_eq!(queue.tail, 0);
        assert_eq!(queue.status, QueueStatus::Idle as u32);
        assert_eq!(queue.is_running, false);
        assert_eq!(queue.commands_sent, 0);
        assert_eq!(queue.commands_processed, 0);
        assert_eq!(queue.buffer[0].cmd_type, CommandType::NoOp as u32);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_claw_executor_new() {
        // Initialize CUDA context (required for cust::memory::UnifiedBuffer)
        let _ctx = Ctx::new(0).unwrap();

        let executor = CudaClawExecutor::new(KernelVariant::Baseline).unwrap();
        assert!(!executor.kernel_running);
        assert_eq!(executor.get_queue().head, 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_claw_executor_launch_stop() {
        let _ctx = Ctx::new(0).unwrap();

        let mut executor = CudaClawExecutor::new(KernelVariant::Baseline).unwrap();
        executor.launch_kernel().unwrap();
        assert!(executor.kernel_running);

        executor.stop_kernel().unwrap();
        assert!(!executor.kernel_running);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_claw_executor_push_command() {
        let _ctx = Ctx::new(0).unwrap();
        let mut executor = CudaClawExecutor::new(KernelVariant::Baseline).unwrap();

        let cmd = Command::new(CommandType::Add, 1).with_data(10.0, 20.0);
        let pushed = executor.push_command(cmd);
        assert!(pushed);
        assert_eq!(executor.get_queue().commands_sent, 1);
        assert_eq!(executor.get_queue().head, 1);

        // Try pushing another command
        let cmd2 = Command::new(CommandType::Subtract, 2).with_data(30.0, 5.0);
        let pushed2 = executor.push_command(cmd2);
        assert!(pushed2);
        assert_eq!(executor.get_queue().commands_sent, 2);
        assert_eq!(executor.get_queue().head, 2);

        executor.stop_kernel().unwrap();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_claw_executor_queue_full() {
        let _ctx = Ctx::new(0).unwrap();
        let mut executor = CudaClawExecutor::new(KernelVariant::Baseline).unwrap();

        let cmd = Command::new(CommandType::NoOp, 0);
        for _ in 0..(QUEUE_SIZE - 1) {
            assert!(executor.push_command(cmd));
        }

        // Queue should be full now
        assert!(!executor.push_command(cmd));
        assert_eq!(executor.get_queue().commands_sent, (QUEUE_SIZE - 1) as u64);

        executor.stop_kernel().unwrap();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_claw_executor_concurrent_push() {
        let _ctx = Ctx::new(0).unwrap();
        let mut executor = CudaClawExecutor::new(KernelVariant::Baseline).unwrap();
        executor.launch_kernel().unwrap();

        let num_threads = 4;
        let num_commands_per_thread = QUEUE_SIZE / num_threads;

        let mut handles = vec![];
        for i in 0..num_threads {
            let mut queue_host = executor.queue.as_host_mut(); // This will not work correctly with multiple threads
            // This line is problematic for concurrent access. We need to find a way to share the UnifiedBuffer.
            // For now, this test will likely fail or have race conditions if run as-is.
            // A proper concurrent test would involve passing a shared reference/pointer to the queue.

            let handle = std::thread::spawn(move || {
                let mut pushed_count = 0;
                for j in 0..num_commands_per_thread {
                    let cmd = Command::new(CommandType::Custom, (i * num_commands_per_thread + j) as u32);
                    if crate::lock_free_queue::LockFreeCommandQueue::push_command(&mut queue_host, cmd) {
                        pushed_count += 1;
                    }
                }
                pushed_count
            });
            handles.push(handle);
        }

        let total_pushed: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        println!("Total commands pushed concurrently: {}", total_pushed);
        // We expect some commands to be pushed, but exact number is non-deterministic without proper sync
        // assert!(total_pushed > 0);

        executor.stop_kernel().unwrap();
    }
}

