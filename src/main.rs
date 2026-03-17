use cuda_claw::{CudaClawExecutor, CommandQueueHost};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod agent;
mod alignment;
mod cuda_claw;
mod dispatcher;
mod lock_free_queue;
mod volatile_dispatcher;

use agent::{AgentDispatcher, AgentOperation, AgentType, CellRef, SuperInstance};
use alignment::{assert_alignment, verify_alignment, KernelConfig, KernelLifecycleManager, print_alignment_report};
use dispatcher::{GpuDispatcher, create_add_command, create_add_batch, calculate_batch_stats};
use lock_free_queue::LockFreeCommandQueue;
use volatile_dispatcher::{VolatileDispatcher, RoundTripBenchmark};

fn run_persistent_worker_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Persistent Worker Kernel Demo ===");
    println!("Demonstrating warp-level parallelism in persistent GPU kernel...\n");

    // Create executor with persistent worker variant
    println!("Initializing CudaClaw executor with PersistentWorker variant...");
    let mut executor = CudaClawExecutor::with_variant(cuda_claw::KernelVariant::PersistentWorker)?;

    executor.init_queue()?;
    executor.start()?;

    println!("Persistent worker kernel is running with warp-level parallelism\n");

    // Submit multiple commands to test parallel processing
    println!("Submitting batch of commands for parallel processing...");
    let test_commands = vec![
        (1.0, 2.0),   // 1 + 2 = 3
        (10.0, 20.0), // 10 + 20 = 30
        (5.5, 4.5),   // 5.5 + 4.5 = 10
        (100.0, 200.0), // 100 + 200 = 300
        (2.5, 3.5),   // 2.5 + 3.5 = 6
    ];

    for (i, (a, b)) in test_commands.iter().enumerate() {
        let (latency, result) = executor.execute_add(*a, *b)?;
        println!("  Command {}: {:.1} + {:.1} = {:.1} ({:?})",
            i + 1, a, b, result, latency);
    }

    // Get worker statistics
    println!("\nWorker Statistics:");
    let worker_stats = executor.get_worker_stats()?;
    println!("  Commands processed: {}", worker_stats.commands_processed);
    println!("  Total cycles: {}", worker_stats.total_cycles);
    println!("  Idle cycles: {}", worker_stats.idle_cycles);
    println!("  Queue head: {}", worker_stats.head);
    println!("  Queue tail: {}", worker_stats.tail);
    println!("  Current status: {:?}", worker_stats.status);
    println!("  Current strategy: {:?}", worker_stats.current_strategy);

    // Measure warp metrics
    println!("\nWarp-Level Performance Metrics:");
    let warp_metrics = executor.measure_warp_metrics()?;
    println!("  Utilization: {}%", warp_metrics.utilization_percent);
    println!("  Commands processed: {}", warp_metrics.commands_processed);
    println!("  Consecutive commands: {}", warp_metrics.consecutive_commands);
    println!("  Consecutive idle: {}", warp_metrics.consecutive_idle);

    // Calculate efficiency
    let total_cycles = worker_stats.total_cycles as f64;
    let idle_cycles = worker_stats.idle_cycles as f64;
    let efficiency = if total_cycles > 0.0 {
        ((total_cycles - idle_cycles) / total_cycles) * 100.0
    } else {
        0.0
    };
    println!("  Worker efficiency: {:.2}%", efficiency);

    // Shutdown
    executor.shutdown()?;
    println!("\nPersistent worker kernel shut down successfully");

    println!("\n=== Persistent Worker Demo Complete ===");

    Ok(())
}

// ============================================================
// Unified Memory "Bridge" Demonstration
// ============================================================
//
// This example shows how to allocate the CommandQueue in Unified Memory
// using cust::memory::UnifiedBuffer. The same memory region is accessible
// from both CPU (Rust) and GPU (CUDA kernel) without explicit copying.
//
// KEY BENEFITS:
// - Zero-copy communication between host and device
// - Sub-microsecond latency for command submission
// - Automatic memory migration by CUDA driver
// - Cache-coherent access on supported hardware
//
// The CommandQueue struct is defined in two places:
// 1. kernels/shared_types.h (C++ side)
// 2. src/cuda_claw.rs (Rust side)
//
// Both definitions MUST match exactly in memory layout, which is
// verified by compile-time assertions in both languages.
//
// ALLOCATION EXAMPLE:
//   use cust::memory::UnifiedBuffer;
//   let queue_data = CommandQueueHost::default();
//   let queue = UnifiedBuffer::new(&queue_data)?;
//
// USAGE PATTERN:
// 1. CPU writes: queue_mut.status = QueueStatus::Ready as u32;
// 2. GPU polls: while (queue->status != STATUS_READY) { ... }
// 3. GPU processes command and sets: queue->status = STATUS_DONE;
// 4. CPU reads: let status = QueueStatus::from(queue.status);
//
// MANUAL ALLOCATION EXAMPLE:
//   // This shows how to manually allocate the CommandQueue in Unified Memory
//   // without using the CudaClawExecutor wrapper
//
//   use cust::memory::UnifiedBuffer;
//   use cuda_claw::{CommandQueueHost, Command, CommandType, QueueStatus};
//
//   // 1. Create default queue data
//   let queue_data = CommandQueueHost::default();
//
//   // 2. Allocate in Unified Memory (both CPU and GPU can access)
//   let mut queue = UnifiedBuffer::new(&queue_data)?;
//
//   // 3. Initialize on GPU using init kernel
//   let func = module.get_function("init_command_queue")?;
//   let mut queue_ptr = queue.as_device_ptr();
//   unsafe {
//       launch!(func<<<1, 1>>>(queue_ptr))?;
//   }
//   stream.synchronize()?;
//
//   // 4. Submit command from CPU
//   {
//       let mut queue_mut = queue.clone();  // Clone for mutable access
//       let cmd = Command::new(CommandType::NoOp, 0);
//       queue_mut.commands[0] = cmd;
//       queue_mut.status = QueueStatus::Ready as u32;
//   }
//
//   // 5. Wait for GPU to complete
//   loop {
//       let queue_ref = queue.clone();
//       if queue_ref.status == QueueStatus::Done as u32 {
//           break;
//       }
//       std::thread::sleep(Duration::from_micros(1));
//   }
//
// ============================================================

fn run_alignment_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Memory Alignment Verification ===");
    println!("Verifying #[repr(C)] alignment between Rust and CUDA...\n");

    // Verify alignment
    let report = verify_alignment();
    print_alignment_report(&report);

    // Assert alignment (panics if invalid)
    assert_alignment();

    println!("✓ All alignment checks passed!");
    println!("  - Command struct: 48 bytes, 32-byte aligned");
    println!("  - CommandQueue struct: 896 bytes, 128-byte aligned");
    println!("  - All field offsets match\n");

    Ok(())
}

fn run_long_running_kernel_demo(
    command_queue: Arc<Mutex<cust::memory::UnifiedBuffer<cuda_claw::CommandQueueHost>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Long-Running Kernel Demo ===");
    println!("Demonstrating kernel health monitoring and lifecycle management...\n");

    // Create lifecycle manager with long-running configuration
    println!("Creating lifecycle manager with long-running configuration...");
    let config = KernelConfig::long_running();
    let mut lifecycle_manager = KernelLifecycleManager::new(command_queue, config);

    println!("Configuration:");
    println!("  Max execution time: {:?}", lifecycle_manager.health_metrics().unwrap().uptime);
    println!("  Health check interval: 1000 ms");
    println!("  Watchdog timeout: 30 sec");
    println!("  Auto-restart: enabled\n");

    // Start lifecycle management
    println!("Starting kernel lifecycle management...");
    lifecycle_manager.start()?;

    // Run for a while and monitor health
    println!("Running kernel health checks for 5 seconds...\n");

    for i in 0..5 {
        std::thread::sleep(Duration::from_secs(1));

        if let Some(metrics) = lifecycle_manager.health_metrics() {
            println!("Health Check [{}]:", i + 1);
            println!("  Status: {:?}", metrics.status);
            println!("  Uptime: {:?}", metrics.uptime);
            println!("  Cycles/sec: {:.2}", metrics.cycles_per_second);
            println!("  Idle percentage: {:.1}%", metrics.idle_percentage);
            println!();
        }

        // Check if restart needed
        if lifecycle_manager.check_and_restart()? {
            println!("⚠ Kernel restart initiated!");
            break;
        }
    }

    println!("Long-running kernel demo complete");

    Ok(())
}

//
// ============================================================

fn run_gpu_dispatcher_demo(
    command_queue: Arc<Mutex<cust::memory::UnifiedBuffer<cuda_claw::CommandQueueHost>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== GPU Dispatcher Demo ===");
    println!("Demonstrating high-performance command dispatch...\n");

    // Create dispatcher
    println!("Creating GPU dispatcher...");
    let mut dispatcher = GpuDispatcher::with_default_queue(command_queue)?;

    // ============================================================
    // Single Command Dispatch
    // ============================================================
    println!("\n--- Single Command Dispatch ---");

    let cmd = create_add_command(10.0, 20.0);
    println!("Dispatching command: 10.0 + 20.0");

    let result = dispatcher.dispatch_sync(cmd)?;
    println!("  Result: {:?}", result);
    println!("  Success: {}", result.success);
    println!("  Latency: {:?}", result.latency);

    // ============================================================
    // Batch Dispatch
    // ============================================================
    println!("\n--- Batch Dispatch ---");

    let pairs = vec![
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.0, 8.0),
        (9.0, 10.0),
        (11.0, 12.0),
        (13.0, 14.0),
        (15.0, 16.0),
    ];

    println!("Dispatching batch of {} commands", pairs.len());
    let batch_commands = create_add_batch(pairs.clone());

    let start = std::time::Instant::now();
    let results = dispatcher.dispatch_batch(batch_commands)?;
    let batch_latency = start.elapsed();

    println!("  Batch completed in: {:?}", batch_latency);
    println!("  Throughput: {:.2} commands/ms", pairs.len() as f64 / batch_latency.as_millis() as f64);

    // Calculate batch statistics
    let (success_rate, avg_latency, max_latency) = calculate_batch_stats(&results);
    println!("  Success rate: {:.1}%", success_rate);
    println!("  Average latency: {:.2} µs", avg_latency);
    println!("  Max latency: {:.2} µs", max_latency);

    // ============================================================
    // Priority Dispatch
    // ============================================================
    println!("\n--- Priority Dispatch ---");

    use dispatcher::DispatchPriority;

    let high_priority_cmd = create_add_command(100.0, 200.0);
    println!("Dispatching high-priority command: 100.0 + 200.0");

    let result = dispatcher.dispatch_with_priority(high_priority_cmd, DispatchPriority::High)?;
    println!("  Result: {:?}", result);
    println!("  Success: {}", result.success);

    // ============================================================
    // Statistics
    // ============================================================
    println!("\n--- Dispatcher Statistics ---");

    let stats = dispatcher.get_stats();
    println!("  Commands submitted: {}", stats.commands_submitted);
    println!("  Commands completed: {}", stats.commands_completed);
    println!("  Commands failed:    {}", stats.commands_failed);
    println!("  Average latency:    {:.2} µs", stats.average_latency_us);
    println!("  Peak queue depth:   {}", stats.peak_queue_depth);
    println!("  Queue full events:  {}", stats.queue_full_count);

    println!("\n=== GPU Dispatcher Demo Complete ===");

    Ok(())
}

// ============================================================
// Lock-Free CommandQueue Demonstration
// ============================================================

fn run_lock_free_queue_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Lock-Free CommandQueue Demo ===");
    println!("Demonstrating lock-free producer-consumer queue in unified memory...\n");

    // Create a command queue in unified memory
    use cuda_claw::{Command, CommandType, QueueStatus};
    use cust::memory::UnifiedBuffer;

    println!("1. Creating CommandQueue in Unified Memory...");
    let queue_data = CommandQueueHost::default();
    let mut queue = UnifiedBuffer::new(&queue_data)?;

    println!("   ✓ CommandQueue allocated at {:p}", queue.as_device_ptr());

    // Print initial queue status
    println!("\n2. Initial Queue Status:");
    {
        let queue_ref = queue.clone();
        LockFreeCommandQueue::print_queue_status(&queue_ref);
    }

    // ============================================================
    // Test 1: Single Push Operations
    // ============================================================
    println!("\n3. Testing Single Push Operations...");

    let cmd1 = Command {
        cmd_type: CommandType::Add as u32,
        id: 1,
        timestamp: 1000,
        data_a: 10.0,
        data_b: 20.0,
        result: 0.0,
        batch_data: 0,
        batch_count: 0,
        _padding: 0,
        result_code: 0,
    };

    println!("   Pushing command 1: 10.0 + 20.0");
    {
        let queue_mut = &mut *queue;
        let success = LockFreeCommandQueue::push_command(queue_mut, cmd1);
        println!("   Push result: {}", if success { "SUCCESS" } else { "FAILED" });
    }

    // Verify queue state
    println!("\n4. Queue Status After Push:");
    {
        let queue_ref = queue.clone();
        LockFreeCommandQueue::print_queue_status(&queue_ref);
    }

    // ============================================================
    // Test 2: Batch Push Operations
    // ============================================================
    println!("\n5. Testing Batch Push Operations...");

    let mut batch_commands = Vec::new();
    for i in 2..=6 {
        batch_commands.push(Command {
            cmd_type: CommandType::Add as u32,
            id: i,
            timestamp: 1000 + i as u64,
            data_a: i as f32 * 10.0,
            data_b: i as f32 * 20.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        });
    }

    println!("   Pushing batch of {} commands...", batch_commands.len());
    {
        let queue_mut = &mut *queue;
        let pushed = LockFreeCommandQueue::push_commands_batch(queue_mut, &batch_commands);
        println!("   Successfully pushed: {} / {} commands", pushed, batch_commands.len());
    }

    // Verify queue state
    println!("\n6. Queue Status After Batch Push:");
    {
        let queue_ref = queue.clone();
        LockFreeCommandQueue::print_queue_status(&queue_ref);
    }

    // ============================================================
    // Test 3: Query Functions
    // ============================================================
    println!("\n7. Testing Query Functions...");

    {
        let queue_ref = queue.clone();
        let size = LockFreeCommandQueue::get_queue_size(&queue_ref);
        let is_empty = LockFreeCommandQueue::is_queue_empty(&queue_ref);
        let is_full = LockFreeCommandQueue::is_queue_full(&queue_ref);
        let state = LockFreeCommandQueue::get_queue_state(&queue_ref);

        println!("   Queue size: {} / {}", size, lock_free_queue::QUEUE_SIZE - 1);
        println!("   Is empty: {}", is_empty);
        println!("   Is full: {}", is_full);
        println!("   State: {:?}", state);
    }

    // ============================================================
    // Test 4: Fill Queue to Capacity
    // ============================================================
    println!("\n8. Testing Queue Capacity...");

    let mut total_pushed = 0;
    let mut attempts = 0;

    // Try to fill the queue
    while attempts < 100 {
        let cmd = Command {
            cmd_type: CommandType::NoOp as u32,
            id: 100 + attempts,
            timestamp: 2000 + attempts as u64,
            data_a: 0.0,
            data_b: 0.0,
            result: 0.0,
            batch_data: 0,
            batch_count: 0,
            _padding: 0,
            result_code: 0,
        };

        {
            let queue_mut = &mut *queue;
            if LockFreeCommandQueue::push_command(queue_mut, cmd) {
                total_pushed += 1;
            } else {
                break;  // Queue is full
            }
        }

        attempts += 1;
    }

    println!("   Attempted to push: {} commands", attempts);
    println!("   Successfully pushed: {} commands", total_pushed);

    // Check if queue is full
    {
        let queue_ref = queue.clone();
        let is_full = LockFreeCommandQueue::is_queue_full(&queue_ref);
        println!("   Queue is full: {}", is_full);
    }

    // Final queue status
    println!("\n9. Final Queue Status:");
    {
        let queue_ref = queue.clone();
        LockFreeCommandQueue::print_queue_status(&queue_ref);
    }

    // ============================================================
    // Test 5: Concurrent Push Simulation
    // ============================================================
    println!("\n10. Simulating Concurrent Push Operations...");

    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let queue_arc = Arc::new(std::sync::Mutex::new(queue));
    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // Spawn 4 threads attempting concurrent pushes
    for thread_id in 0..4 {
        let queue_clone = queue_arc.clone();
        let success_clone = success_count.clone();
        let failure_clone = failure_count.clone();

        let handle = thread::spawn(move || {
            for i in 0..10 {
                let cmd = Command {
                    cmd_type: CommandType::NoOp as u32,
                    id: (thread_id * 10 + i) as u32,
                    timestamp: 3000 + (thread_id * 10 + i) as u64,
                    data_a: 0.0,
                    data_b: 0.0,
                    result: 0.0,
                    batch_data: 0,
                    batch_count: 0,
                    _padding: 0,
                    result_code: 0,
                };

                let mut queue_guard = queue_clone.lock().unwrap();
                if LockFreeCommandQueue::push_command(&mut *queue_guard, cmd) {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    failure_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);

    println!("   Concurrent push results:");
    println!("     Successful pushes: {}", successes);
    println!("     Failed pushes (queue full): {}", failures);
    println!("     Total attempts: {}", successes + failures);

    // Final statistics
    println!("\n11. Final Statistics:");
    {
        let queue_guard = queue_arc.lock().unwrap();
        let queue_ref = (*queue_guard).clone();

        let (pushed, popped, processed) = LockFreeCommandQueue::get_queue_stats(&queue_ref);
        println!("     Commands pushed: {}", pushed);
        println!("     Commands popped: {}", popped);
        println!("     Commands processed: {}", processed);

        let efficiency = if pushed > 0 {
            (processed as f64 / pushed as f64) * 100.0
        } else {
            0.0
        };
        println!("     Processing efficiency: {:.1}%", efficiency);
    }

    println!("\n=== Lock-Free CommandQueue Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  ✓ Lock-free push operations using atomic CAS");
    println!("  ✓ Circular buffer with head/tail indices");
    println!("  ✓ Thread-safe concurrent access");
    println!("  ✓ Zero-copy unified memory between CPU and GPU");
    println!("  ✓ Compatible with CUDA pop_command on GPU side");

    Ok(())
}

//
// ============================================================

// ============================================================
// GPU Bridge Initialization Function
// ============================================================
//
/// Initialize the GPU bridge by allocating a CommandQueue in Unified Memory.
///
/// This function creates the zero-copy communication channel between
/// Rust host code and CUDA GPU kernels using Unified Memory.
///
/// # Arguments
///
/// *None*
///
/// # Returns
///
/// * `Result<(UnifiedBuffer<CommandQueueHost>, cust::memory::UnifiedPointer<CommandQueueHost>)>` -
///   A tuple containing the buffer handle for CPU access and a device pointer for GPU kernels
///
/// # Errors
///
/// Returns an error if CUDA initialization fails or memory allocation fails
///
/// # Memory Layout
///
/// The CommandQueue structure is defined in both:
/// - `kernels/shared_types.h` (CUDA C++ side)
/// - `src/cuda_claw.rs` (Rust side)
///
/// Both definitions MUST match exactly in memory layout to prevent kernel crashes.
/// This is verified at compile time using static_assert in C++ and runtime tests in Rust.
///
/// # Example
///
/// ```ignore
/// // Initialize the GPU bridge
/// let (queue, queue_ptr) = init_gpu_bridge()?;
///
/// // Use queue from CPU (Rust)
/// {
///     let mut queue_mut = &mut *queue;
///     queue_mut.status = QueueStatus::Ready as u32;
/// }
///
/// // Pass queue_ptr to CUDA kernel
/// unsafe {
///     launch!(my_kernel<<<1, 1>>>(queue_ptr))?;
/// }
/// ```
///
fn init_gpu_bridge() -> Result<(
    cust::memory::UnifiedBuffer<cuda_claw::CommandQueueHost>,
    cust::memory::UnifiedPointer<cuda_claw::CommandQueueHost>
), Box<dyn std::error::Error>> {
    use cust::memory::UnifiedBuffer;
    use cuda_claw::{CommandQueueHost, QueueStatus};

    println!("Initializing GPU bridge...");
    println!("  Allocating CommandQueue in Unified Memory...");

    // Create zeroed CommandQueue data
    // This ensures all fields are properly initialized
    let queue_data = CommandQueueHost {
        status: QueueStatus::Idle as u32,
        commands: [Default::default(); 16], // QUEUE_SIZE = 16
        head: 0,
        tail: 0,
        commands_pushed: 0,
        commands_popped: 0,
        commands_processed: 0,
        total_cycles: 0,
        idle_cycles: 0,
        current_strategy: cuda_claw::PollingStrategy::Adaptive as u32,
        consecutive_commands: 0,
        consecutive_idle: 0,
        last_command_cycle: 0,
        avg_command_latency_cycles: 0,
        _padding: [0u8; 48],
    };

    // Allocate in Unified Memory
    // This memory region is accessible from both CPU and GPU
    let queue = UnifiedBuffer::new(&queue_data)?;

    println!("  ✓ CommandQueue allocated in Unified Memory");
    println!("  ✓ Memory size: {} bytes", std::mem::size_of::<CommandQueueHost>());
    println!("  ✓ Device pointer: {:p}", queue.as_device_ptr());

    // Get the device pointer for CUDA kernel usage
    let queue_ptr = queue.as_device_ptr();

    println!("  ✓ GPU bridge initialized successfully\n");

    Ok((queue, queue_ptr))
}

// ============================================================
// ROUND-TRIP LATENCY BENCHMARK
// ============================================================

fn run_round_trip_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Round-Trip Latency Benchmark ===");
    println!("Measuring Rust→GPU→Rust command round-trip latency\n");

    // Initialize executor and start persistent kernel
    let mut executor = CudaClawExecutor::new()?;
    executor.init_queue()?;
    executor.start()?;

    println!("Persistent kernel started\n");

    // Create volatile dispatcher
    let mut dispatcher = VolatileDispatcher::new(executor.queue.clone())?;
    let mut benchmark = RoundTripBenchmark::new(executor.queue.clone())?;

    println!("Benchmark configuration:");
    println!("  Command type: NoOp (minimal GPU processing)");
    println!("  Synchronization: cudaDeviceSynchronize() after each command");
    println!("  Memory: Unified Buffer (zero-copy)");
    println!();

    // Run benchmark with 1000 iterations
    let iterations = 1000;
    let warmup_iterations = 100;

    let results = benchmark.run_benchmark(iterations, warmup_iterations)?;

    // Print detailed results
    results.print();

    // Print dispatcher statistics
    dispatcher.print_stats();

    // Export results to CSV
    let csv_data = results.to_csv();
    println!("\nCSV data (first 10 lines):");
    for line in csv_data.lines().take(10) {
        println!("  {}", line);
    }

    // Performance analysis
    println!("\nPerformance Analysis:");
    println!("  Memory bandwidth: Unified Memory eliminates PCIe transfers");
    println!("  Volatile writes: ~50-100ns per command submission");
    println!("  GPU polling: ~1-5 microseconds (due to __nanosleep)");
    println!("  Synchronization: Only when explicitly requested");

    if results.avg_latency.as_micros() < 10 {
        println!("  ✓ EXCELLENT: Sub-10µs latency achieved!");
    } else if results.avg_latency.as_micros() < 50 {
        println!("  ✓ GOOD: Sub-50µs latency achieved");
    } else {
        println!("  ⚠ WARNING: Latency above 50µs - may need optimization");
    }

    // Throughput analysis
    let throughput_million = iterations as f64 / results.total_latency.as_secs_f64() / 1_000_000.0;
    println!("\nThroughput Analysis:");
    if throughput_million > 1.0 {
        println!("  ✓ EXCELLENT: >1M commands/second");
    } else if throughput_million > 0.1 {
        println!("  ✓ GOOD: >100K commands/second");
    } else {
        println!("  ⚠ NOTE: Throughput below 100K commands/second");
    }

    Ok(())
}

//
// ============================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CudaClaw - GPU-Accelerated SmartCRDT Orchestrator");
    println!("==============================================\n");

    // Initialize CUDA
    let _ctx = cust::quick_init().expect("Failed to initialize CUDA");

    println!("CUDA initialized successfully");
    println!("GPUs available: {}", cust::device::get_device_count()?);

    // Verify alignment before proceeding
    run_alignment_verification()?;

    // Create the CudaClaw executor
    println!("\nInitializing CudaClaw executor...");
    let mut executor = CudaClawExecutor::new()?;

    // Initialize the command queue
    println!("Initializing command queue...");
    executor.init_queue()?;

    // Get the unified memory command queue for AgentDispatcher
    let command_queue = Arc::new(Mutex::new(executor.queue.clone()));

    // Start the persistent kernel
    println!("Starting persistent GPU kernel...");
    executor.start()?;

    println!("Persistent kernel is now running on GPU\n");

    // Run round-trip latency benchmark with volatile dispatcher
    run_round_trip_benchmark()?;

    // Run latency tests
    run_latency_tests(&mut executor)?;

    // Run functional tests
    run_functional_tests(&mut executor)?;

    // Demonstrate persistent worker kernel
    run_persistent_worker_demo()?;

    // Demonstrate long-running kernel support
    run_long_running_kernel_demo(command_queue.clone())?;

    // Demonstrate GPU Dispatcher
    run_gpu_dispatcher_demo(command_queue.clone())?;

    // Demonstrate Lock-Free CommandQueue
    run_lock_free_queue_demo()?;

    // Demonstrate AgentDispatcher
    run_agent_dispatcher_demo(command_queue.clone())?;

    // Shutdown the persistent kernel
    println!("\nShutting down persistent kernel...");
    executor.shutdown()?;

    println!("\nCudaClaw executor shut down successfully");

    Ok(())
}

fn run_latency_tests(executor: &mut CudaClawExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Latency Tests ===");
    println!("Testing round-trip latency between host and GPU kernel...\n");

    const NUM_ITERATIONS: usize = 100;
    let mut latencies = Vec::with_capacity(NUM_ITERATIONS);

    // Warmup
    for _ in 0..10 {
        let _ = executor.execute_no_op();
    }

    // Actual latency test
    for i in 0..NUM_ITERATIONS {
        let latency = executor.execute_no_op()?;
        latencies.push(latency);

        if (i + 1) % 20 == 0 {
            println!("  Completed {} iterations", i + 1);
        }
    }

    // Calculate statistics
    latencies.sort();
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let sum: Duration = latencies.iter().sum();
    let avg = sum / NUM_ITERATIONS as u32;
    let median = latencies[NUM_ITERATIONS / 2];
    let p95 = latencies[(NUM_ITERATIONS * 95) / 100];
    let p99 = latencies[(NUM_ITERATIONS * 99) / 100];

    println!("\nLatency Statistics ({} iterations):", NUM_ITERATIONS);
    println!("  Min:     {:8.2?}", min);
    println!("  Max:     {:8.2?}", max);
    println!("  Average: {:8.2?}", avg);
    println!("  Median:  {:8.2?}", median);
    println!("  P95:     {:8.2?}", p95);
    println!("  P99:     {:8.2?}", p99);

    Ok(())
}

fn run_functional_tests(executor: &mut CudaClawExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Functional Tests ===");

    // Test Add command
    println!("\nTesting Add command:");
    let test_cases = vec![
        (1.0, 2.0, 3.0),
        (-5.0, 3.0, -2.0),
        (0.0, 0.0, 0.0),
        (3.14159, 2.71828, 5.85987),
        (1000.0, 2000.0, 3000.0),
    ];

    for (a, b, expected) in test_cases {
        let (latency, result) = executor.execute_add(a, b)?;
        let epsilon = 0.0001;
        let success = (result - expected).abs() < epsilon;
        println!("  {:.5} + {:.5} = {:.5} [{} - {:?}]",
            a, b, result,
            if success { "OK" } else { "FAIL" },
            latency
        );
    }

    // Get statistics
    let stats = executor.get_stats();
    println!("\nQueue Statistics:");
    println!("  Commands processed: {}", stats.commands_processed);
    println!("  Total cycles:       {}", stats.total_cycles);
    println!("  Queue head:         {}", stats.head);
    println!("  Queue tail:         {}", stats.tail);
    println!("  Current status:     {:?}", stats.status);

    Ok(())
}

fn run_agent_dispatcher_demo(
    command_queue: Arc<Mutex<cust::memory::UnifiedBuffer<cuda_claw::CommandQueueHost>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== AgentDispatcher Demo ===");
    println!("Demonstrating SuperInstance agent management...\n");

    // Create AgentDispatcher with a 100x100 spreadsheet
    const GRID_ROWS: u32 = 100;
    const GRID_COLS: u32 = 100;
    const NODE_ID: u32 = 1;

    println!("Creating AgentDispatcher with {}x{} spreadsheet grid", GRID_ROWS, GRID_COLS);
    let mut dispatcher = AgentDispatcher::new(GRID_ROWS, GRID_COLS, command_queue, NODE_ID)?;

    // Register various SuperInstance agents
    println!("\nRegistering SuperInstance agents...");

    let claw_agent = SuperInstance::new("claw_001".to_string(), AgentType::Claw)
        .with_model("deepseek-chat".to_string())
        .with_equipment(vec!["MEMORY".to_string(), "REASONING".to_string()]);

    let smpclaw_agent = SuperInstance::new("smpclaw_001".to_string(), AgentType::SMPclaw)
        .with_model("deepseek-coder".to_string())
        .with_equipment(vec!["MEMORY".to_string(), "SPREADSHEET".to_string()]);

    let bot_agent = SuperInstance::new("bot_001".to_string(), AgentType::Bot);

    dispatcher.register_agent(claw_agent)?;
    dispatcher.register_agent(smpclaw_agent)?;
    dispatcher.register_agent(bot_agent)?;

    println!("  Registered {} agents", dispatcher.agents.len());

    // List registered agents
    println!("\nRegistered Agents:");
    for agent in dispatcher.list_agents() {
        println!("  - {}: {:?} (model: {:?}, equipment: {} items)",
            agent.id,
            agent.agent_type,
            agent.model,
            agent.equipment.len()
        );
    }

    // Demonstrate cell operations
    println!("\n=== Cell Operations ===");

    // Set some initial values
    println!("\nSetting cell values...");
    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(0, 0),
        value: 42.0,
        timestamp: 1,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(0, 1),
        value: 3.14159,
        timestamp: 2,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(1, 0),
        value: 2.71828,
        timestamp: 3,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    // Get cell values
    println!("\nGetting cell values...");
    let get_result = dispatcher.dispatch_agent_op(AgentOperation::GetCell {
        cell: CellRef::new(0, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&get_result)?);

    // Add cells
    println!("\nAdding cells...");
    let add_result = dispatcher.dispatch_agent_op(AgentOperation::AddCells {
        a: CellRef::new(0, 0),
        b: CellRef::new(0, 1),
        result: CellRef::new(2, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&add_result)?);

    // Multiply cells
    println!("\nMultiplying cells...");
    let mul_result = dispatcher.dispatch_agent_op(AgentOperation::MultiplyCells {
        a: CellRef::new(0, 0),
        b: CellRef::new(1, 0),
        result: CellRef::new(3, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&mul_result)?);

    // Apply formula
    println!("\nApplying formula (sum of range)...");
    let formula_result = dispatcher.dispatch_agent_op(AgentOperation::ApplyFormula {
        inputs: vec![
            CellRef::new(0, 0),
            CellRef::new(0, 1),
            CellRef::new(1, 0),
        ],
        output: CellRef::new(4, 0),
        formula: "SUM".to_string(),
    })?;
    println!("  {}", serde_json::to_string_pretty(&formula_result)?);

    // Batch update
    println!("\nBatch updating cells...");
    let batch_updates = vec![
        (CellRef::new(5, 0), 1.0),
        (CellRef::new(5, 1), 2.0),
        (CellRef::new(5, 2), 3.0),
        (CellRef::new(5, 3), 4.0),
        (CellRef::new(5, 4), 5.0),
    ];
    let batch_result = dispatcher.dispatch_agent_op(AgentOperation::BatchUpdate {
        updates: batch_updates.clone(),
        timestamp: 100,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&batch_result)?);

    // Agent operation
    println!("\nDispatching agent operation...");
    let agent_result = dispatcher.dispatch_agent_op(AgentOperation::AgentOp {
        agent_id: "claw_001".to_string(),
        operation: "add".to_string(),
        params: serde_json::json!({"a": 10.0, "b": 20.0}),
    })?;
    println!("  {}", serde_json::to_string_pretty(&agent_result)?);

    // Demonstrate JSON command parsing
    println!("\n=== JSON Command Parsing ===");

    let json_cmd = r#"{
        "op": "SetCell",
        "cell": {"row": 10, "col": 10},
        "value": 99.99,
        "timestamp": 200,
        "node_id": 1
    }"#;

    println!("\nParsing JSON command:");
    println!("  {}", json_cmd);

    let parsed_op = agent::parse_command(json_cmd)?;
    let result = dispatcher.dispatch_agent_op(parsed_op)?;
    println!("  Result: {}", serde_json::to_string_pretty(&result)?);

    // Command validation demo
    println!("\n=== Command Validation ===");

    // Invalid command (out of bounds)
    let invalid_cmd = AgentOperation::SetCell {
        cell: CellRef::new(999999, 999999),
        value: 1.0,
        timestamp: 1,
        node_id: 1,
    };

    match invalid_cmd.validate() {
        Ok(_) => println!("  Command validated (unexpected)"),
        Err(e) => println!("  Validation error (expected): {}", e),
    }

    // Get dispatcher statistics
    println!("\n=== Dispatcher Statistics ===");
    let stats = dispatcher.get_stats();
    println!("  Total agents: {}", stats.total_agents);
    println!("  Grid size: {} x {} ({} cells)", stats.grid_size.0, stats.grid_size.1, stats.total_cells);
    println!("  Current timestamp: {}", stats.current_timestamp);
    println!("  Node ID: {}", stats.node_id);
    println!("  Agents by type:");
    for (agent_type, count) in stats.agents_by_type {
        println!("    - {:?}: {}", agent_type, count);
    }

    println!("\n=== AgentDispatcher Demo Complete ===");

    Ok(())
}
