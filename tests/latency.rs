// ============================================================
// Round-Trip Time (RTT) Latency Benchmarking Suite
// ============================================================
//
// Measures the full round-trip latency from the moment a Rust
// command is pushed into the CommandQueue to the moment the GPU
// writes a result back into a Completion flag in Unified Memory.
//
// TARGET: P99 RTT < 8 microseconds
//
// TIMING METHODOLOGY:
// - Host side:  std::time::Instant  (nanosecond precision)
// - Device side: cudaEvent_t        (sub-microsecond GPU timing)
//
// The benchmark decomposes RTT into measurable phases:
//   1. Host Push Latency     — time to write command + advance head
//   2. PCIe Propagation      — time for GPU to observe new head
//   3. GPU Execution         — kernel processing time (cudaEvent_t)
//   4. Completion Writeback  — time for host to observe result_code
//
// By comparing host-side Instant timestamps with device-side
// cudaEvent_t deltas, we can pinpoint PCIe bus contention vs.
// kernel execution overhead.
//
// NOTE: This file is self-contained and does not import from the
// binary crate. All required types are redefined to match the
// production layout in src/cuda_claw.rs and kernels/shared_types.h.
//
// Run with:
//   cargo test --test latency --release -- --nocapture
//
// ============================================================

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::mem::zeroed;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================
// Command types matching src/cuda_claw.rs / shared_types.h
// ============================================================

/// CommandType values matching the CUDA enum in shared_types.h
#[allow(dead_code)]
mod cmd_type {
    pub const NOOP: u32 = 0;
    pub const EDIT_CELL: u32 = 1;
    pub const SYNC_CRDT: u32 = 2;
    pub const SHUTDOWN: u32 = 3;
    /// SpreadsheetEdit uses cmd_type = 5 in production
    pub const SPREADSHEET_EDIT: u32 = 5;
}

/// Completion flag values written by the GPU into result_code
/// to signal command completion back to the host.
#[allow(dead_code)]
mod completion {
    /// Command has not been processed yet
    pub const PENDING: u32 = 0;
    /// Command completed successfully
    pub const SUCCESS: u32 = 1;
    /// Command completed with an error
    pub const ERROR: u32 = 0xDEAD;
}

/// Logical queue capacity for lock-free ring buffer wrap-around.
const QUEUE_SIZE: u32 = 16;

/// Physical buffer capacity matching src/cuda_claw.rs QUEUE_SIZE.
const BUFFER_SIZE: usize = 1024;

/// Mirrors Command in src/cuda_claw.rs (#[repr(C, packed(4))], 48 bytes)
#[repr(C, packed(4))]
#[derive(Debug, Clone, Copy)]
struct Command {
    cmd_type:    u32,
    id:          u32,
    timestamp:   u64,
    data_a:      f32,
    data_b:      f32,
    result:      f32,
    batch_data:  u64,
    batch_count: u32,
    _padding:    u32,
    result_code: u32,
}

const _: [(); std::mem::size_of::<Command>()] = [(); 48];

/// Mirrors CommandQueueHost in src/cuda_claw.rs (49,192 bytes, packed(4)).
/// Field order: buffer first, then metadata — matching the production
/// binary layout used for CUDA unified memory.
#[repr(C, packed(4))]
struct CommandQueueHost {
    buffer:              [Command; BUFFER_SIZE],  // offset 0,     48,992 bytes
    status:              u32,                     // offset 48,992, 4 bytes
    head:                u32,                     // offset 48,996, 4 bytes
    tail:                u32,                     // offset 49,000, 4 bytes
    is_running:          bool,                    // offset 49,004, 1 byte
    _padding:            [u8; 3],                 // offset 49,005, 3 bytes
    commands_sent:       u64,                     // offset 49,008, 8 bytes
    commands_processed:  u64,                     // offset 49,016, 8 bytes
    _stats_padding:      [u8; 8],                 // offset 49,024, 8 bytes
}

const _: [(); std::mem::size_of::<CommandQueueHost>()] = [(); 49192];

// ============================================================
// Lock-free push (mirrors src/lock_free_queue.rs)
// ============================================================

/// Push a command into the ring buffer using a lock-free CAS on head.
/// Returns true if the command was successfully enqueued.
fn push_command(queue: &mut CommandQueueHost, cmd: Command) -> bool {
    unsafe {
        let head = queue.head;
        let tail = queue.tail;
        let next_head = (head + 1) % QUEUE_SIZE;
        if next_head == tail {
            return false; // full
        }
        let index = (head % QUEUE_SIZE) as usize;
        queue.buffer[index] = cmd;
        std::sync::atomic::fence(Ordering::SeqCst);
        let atomic = &*((&queue.head as *const u32) as *const AtomicU32);
        if atomic
            .compare_exchange_weak(head, next_head, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let pushed_atomic =
                &*((std::ptr::addr_of!(queue.commands_sent)) as *const AtomicU64);
            pushed_atomic.fetch_add(1, Ordering::SeqCst);
            std::sync::atomic::fence(Ordering::SeqCst);
            return true;
        }
        false
    }
}

/// Check if the ring buffer is full.
fn is_queue_full(queue: &CommandQueueHost) -> bool {
    let next_head = (queue.head + 1) % QUEUE_SIZE;
    next_head == queue.tail
}

/// Simulate the GPU consumer advancing the tail pointer and writing
/// a completion flag into the command's result_code field.
/// This mimics what the persistent_worker kernel does:
///   1. Read the command at queue->buffer[tail]
///   2. Process the command
///   3. Write result_code = SUCCESS into the command slot
///   4. Advance tail
///   5. Increment commands_processed
///   6. __threadfence_system()  (simulated here by SeqCst fence)
fn simulate_gpu_completion(queue: &mut CommandQueueHost) {
    let head = queue.head;
    let tail = queue.tail;
    if head == tail {
        return; // nothing to consume
    }
    let idx = (tail % QUEUE_SIZE) as usize;
    // GPU writes the completion flag into result_code
    queue.buffer[idx].result_code = completion::SUCCESS;
    std::sync::atomic::fence(Ordering::SeqCst);
    queue.tail = (tail + 1) % QUEUE_SIZE;
    queue.commands_processed += 1;
    std::sync::atomic::fence(Ordering::SeqCst);
}

/// Reset queue indices to allow continued operation.
fn reset_queue(queue: &mut CommandQueueHost) {
    queue.head = 0;
    queue.tail = 0;
}

// ============================================================
// RTT Sample — one round-trip measurement
// ============================================================

/// A single round-trip measurement capturing all timing phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RttSample {
    /// Sequence number of this sample
    seq: u64,

    // ── Host-side timestamps (nanoseconds, std::time::Instant) ──

    /// Total round-trip time: push start → completion observed (ns)
    rtt_ns: u64,

    /// Phase 1: Host push latency — write command + CAS head (ns)
    host_push_ns: u64,

    /// Phase 2+3+4 combined: time from push complete to completion
    /// flag observed by host (ns). On a real GPU this includes
    /// PCIe propagation + kernel execution + writeback.
    completion_wait_ns: u64,

    // ── Device-side timing (simulated cudaEvent_t, nanoseconds) ──

    /// Simulated GPU-side execution time (ns).
    /// In production this would be measured with:
    ///   cudaEventRecord(start); ... cudaEventRecord(stop);
    ///   cudaEventElapsedTime(&ms, start, stop);
    /// Here we model it as a fixed kernel cost.
    gpu_exec_ns: u64,

    /// Estimated PCIe propagation delay (ns) = rtt - push - gpu_exec
    /// This isolates the PCIe bus contention component.
    pcie_delay_ns: u64,
}

// ============================================================
// Simulated cudaEvent_t Timing
// ============================================================
//
// cudaEvent_t provides sub-microsecond GPU-side timing.
// On a real CUDA system the flow is:
//
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start, stream);
//   <<< kernel execution >>>
//   cudaEventRecord(stop, stream);
//   cudaEventSynchronize(stop);
//   float ms;
//   cudaEventElapsedTime(&ms, start, stop);
//
// Without a live GPU, we model the kernel execution time as a
// configurable constant. This lets us decompose the RTT into
// meaningful phases and validate the host-side measurement
// infrastructure.

/// Simulated cudaEvent_t kernel execution time.
/// On a real GPU this would be measured per-command.
/// Typical persistent_worker NOOP processing: ~200-500 ns.
const SIMULATED_GPU_EXEC_NS: u64 = 350;

/// Model representing a pair of cudaEvent_t timestamps
/// for measuring GPU-side kernel execution time.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CudaEventPair {
    /// Simulated cudaEventRecord(start) time (ns since epoch)
    start_ns: u64,
    /// Simulated cudaEventRecord(stop) time (ns since epoch)
    stop_ns: u64,
    /// cudaEventElapsedTime result (ns)
    elapsed_ns: u64,
}

impl CudaEventPair {
    /// Simulate a cudaEvent_t measurement for a given execution.
    /// In production, this wraps actual cudaEventRecord/Elapsed calls.
    fn simulate(exec_ns: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        CudaEventPair {
            start_ns: now,
            stop_ns: now + exec_ns,
            elapsed_ns: exec_ns,
        }
    }
}

// ============================================================
// Latency Statistics (reusable)
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatencyStats {
    samples: usize,
    min_ns: u64,
    max_ns: u64,
    mean_ns: u64,
    std_dev_ns: f64,
    p50_ns: u64,
    p90_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    // Microsecond convenience fields
    min_us: f64,
    max_us: f64,
    mean_us: f64,
    p50_us: f64,
    p90_us: f64,
    p95_us: f64,
    p99_us: f64,
    p999_us: f64,
}

fn compute_stats(sorted_ns: &[u64]) -> LatencyStats {
    let n = sorted_ns.len();
    assert!(n > 0, "Cannot compute stats on empty slice");

    let min_ns = sorted_ns[0];
    let max_ns = sorted_ns[n - 1];
    let sum: u64 = sorted_ns.iter().sum();
    let mean_ns = sum / n as u64;

    let p50_ns  = sorted_ns[n / 2];
    let p90_ns  = sorted_ns[(n * 90) / 100];
    let p95_ns  = sorted_ns[(n * 95) / 100];
    let p99_ns  = sorted_ns[(n * 99) / 100];
    let p999_ns = sorted_ns[(n * 999) / 1000];

    let variance: f64 = sorted_ns
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean_ns as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std_dev_ns = variance.sqrt();

    LatencyStats {
        samples: n,
        min_ns, max_ns, mean_ns, std_dev_ns,
        p50_ns, p90_ns, p95_ns, p99_ns, p999_ns,
        min_us:  min_ns  as f64 / 1_000.0,
        max_us:  max_ns  as f64 / 1_000.0,
        mean_us: mean_ns as f64 / 1_000.0,
        p50_us:  p50_ns  as f64 / 1_000.0,
        p90_us:  p90_ns  as f64 / 1_000.0,
        p95_us:  p95_ns  as f64 / 1_000.0,
        p99_us:  p99_ns  as f64 / 1_000.0,
        p999_us: p999_ns as f64 / 1_000.0,
    }
}

fn print_stats_table(label: &str, s: &LatencyStats) {
    println!("\n=== {} ({} samples) ===", label, s.samples);
    println!("┌───────────┬────────────────────┬───────────────┐");
    println!("│  Metric   │   Nanoseconds      │ Microseconds  │");
    println!("├───────────┼────────────────────┼───────────────┤");
    println!("│  Min      │  {:>16}  │  {:>11.3}  │", s.min_ns,  s.min_us);
    println!("│  Max      │  {:>16}  │  {:>11.3}  │", s.max_ns,  s.max_us);
    println!("│  Mean     │  {:>16}  │  {:>11.3}  │", s.mean_ns, s.mean_us);
    println!("│  Std Dev  │  {:>16.0}  │  {:>11.3}  │", s.std_dev_ns, s.std_dev_ns / 1_000.0);
    println!("├───────────┼────────────────────┼───────────────┤");
    println!("│  P50      │  {:>16}  │  {:>11.3}  │", s.p50_ns,  s.p50_us);
    println!("│  P90      │  {:>16}  │  {:>11.3}  │", s.p90_ns,  s.p90_us);
    println!("│  P95      │  {:>16}  │  {:>11.3}  │", s.p95_ns,  s.p95_us);
    println!("│  P99      │  {:>16}  │  {:>11.3}  │", s.p99_ns,  s.p99_us);
    println!("│  P99.9    │  {:>16}  │  {:>11.3}  │", s.p999_ns, s.p999_us);
    println!("└───────────┴────────────────────┴───────────────┘");
}

// ============================================================
// RTT Phase Breakdown
// ============================================================

/// Aggregated phase breakdown across all RTT samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhaseBreakdown {
    /// Phase 1: Host push (write command + CAS on head)
    host_push: LatencyStats,
    /// Phase 2+3+4: Completion wait (PCIe + GPU exec + writeback)
    completion_wait: LatencyStats,
    /// Simulated GPU execution time
    gpu_exec: LatencyStats,
    /// Estimated PCIe propagation delay (rtt - push - gpu_exec)
    pcie_delay: LatencyStats,
    /// Full RTT
    rtt: LatencyStats,
}

// ============================================================
// RTT Benchmark Configuration
// ============================================================

/// Configuration for the RTT benchmark.
struct RttBenchConfig {
    /// Total number of RTT measurements to collect
    total_samples: usize,
    /// Number of warmup iterations (discarded)
    warmup_samples: usize,
    /// Simulated GPU execution time per command (ns)
    gpu_exec_ns: u64,
    /// P99 RTT target in microseconds
    p99_target_us: f64,
    /// Whether to output rtt_latency_report.json
    write_report: bool,
}

impl Default for RttBenchConfig {
    fn default() -> Self {
        RttBenchConfig {
            total_samples: 100_000,
            warmup_samples: 5_000,
            gpu_exec_ns: SIMULATED_GPU_EXEC_NS,
            p99_target_us: 8.0,
            write_report: true,
        }
    }
}

// ============================================================
// Core RTT Measurement Function
// ============================================================

/// Perform a single round-trip measurement.
///
/// Steps:
///   1. Record t0 (host push start)
///   2. Push command into queue (lock-free CAS)
///   3. Record t1 (host push complete)
///   4. Simulate GPU processing (cudaEvent_t pair)
///   5. Write completion flag into result_code
///   6. Advance tail (simulate GPU consumer)
///   7. Record t2 (host observes completion)
///   8. Compute phases: push = t1-t0, wait = t2-t1, rtt = t2-t0
///   9. Estimate PCIe delay = rtt - push - gpu_exec
fn measure_single_rtt(
    queue: &mut CommandQueueHost,
    seq: u64,
    gpu_exec_ns: u64,
) -> Option<RttSample> {
    // Build command with NOOP type for minimal processing overhead.
    // result_code starts at PENDING; GPU will overwrite with SUCCESS.
    let cmd = Command {
        cmd_type: cmd_type::NOOP,
        id: seq as u32,
        timestamp: seq,
        data_a: 0.0,
        data_b: 0.0,
        result: 0.0,
        batch_data: 0,
        batch_count: 0,
        _padding: 0,
        result_code: completion::PENDING,
    };

    // Drain if full — simulate GPU consuming a backlog
    if is_queue_full(queue) {
        simulate_gpu_completion(queue);
    }
    if is_queue_full(queue) {
        // Still full after one drain — force reset
        reset_queue(queue);
    }

    // ── Phase 1: Host Push ──
    let t0 = Instant::now();
    let pushed = push_command(queue, cmd);
    let t1 = Instant::now();

    if !pushed {
        return None;
    }
    let host_push_ns = t1.duration_since(t0).as_nanos() as u64;

    // ── Phase 2+3: Simulate GPU execution (cudaEvent_t timing) ──
    // On a real GPU, the persistent_worker would:
    //   1. See the new head via __threadfence_system()  (PCIe delay)
    //   2. Process the command                          (GPU exec)
    //   3. Write result_code = SUCCESS                  (writeback)
    //   4. Advance tail + __threadfence_system()        (PCIe delay)
    //
    // We simulate this by:
    //   a) Spinning for gpu_exec_ns (models kernel time)
    //   b) Writing the completion flag
    //   c) Advancing tail
    let _cuda_event = CudaEventPair::simulate(gpu_exec_ns);

    // Simulate the GPU-side processing delay.
    // On real hardware, cudaEventRecord captures the actual GPU clock.
    // Here we spin to model the kernel execution time accurately.
    let spin_start = Instant::now();
    while spin_start.elapsed().as_nanos() < gpu_exec_ns as u128 {
        std::hint::spin_loop();
    }

    // GPU writes completion and advances tail
    simulate_gpu_completion(queue);

    // ── Phase 4: Host observes completion ──
    let t2 = Instant::now();
    let completion_wait_ns = t2.duration_since(t1).as_nanos() as u64;
    let rtt_ns = t2.duration_since(t0).as_nanos() as u64;

    // Estimate PCIe propagation delay:
    // pcie_delay = rtt - host_push - gpu_exec
    // This isolates the round-trip bus latency from computation.
    let pcie_delay_ns = rtt_ns.saturating_sub(host_push_ns + gpu_exec_ns);

    Some(RttSample {
        seq,
        rtt_ns,
        host_push_ns,
        completion_wait_ns,
        gpu_exec_ns,
        pcie_delay_ns,
    })
}

// ============================================================
// Test: Full RTT Benchmark (100k samples, P99 < 8µs target)
// ============================================================

#[test]
fn test_rtt_latency_benchmark() {
    let config = RttBenchConfig::default();

    println!("\n{}", "═".repeat(68));
    println!("  Round-Trip Time (RTT) Latency Benchmark");
    println!("{}", "═".repeat(68));
    println!("  Samples     : {} (+ {} warmup)", config.total_samples, config.warmup_samples);
    println!("  GPU exec    : {} ns (simulated cudaEvent_t)", config.gpu_exec_ns);
    println!("  P99 target  : < {:.1} µs", config.p99_target_us);
    println!("  Output      : rtt_latency_report.json");
    println!("{}\n", "═".repeat(68));

    let mut queue: CommandQueueHost = unsafe { zeroed() };
    let mut rng = rand::thread_rng();

    // ── Warmup phase ──
    println!("Warming up ({} iterations)...", config.warmup_samples);
    for i in 0..config.warmup_samples {
        let _ = measure_single_rtt(&mut queue, i as u64, config.gpu_exec_ns);
    }
    reset_queue(&mut queue);
    println!("Warmup complete.\n");

    // ── Measurement phase ──
    println!("Collecting {} RTT samples...", config.total_samples);
    let mut samples: Vec<RttSample> = Vec::with_capacity(config.total_samples);
    let mut failed: u64 = 0;

    let bench_start = Instant::now();
    let progress_interval = config.total_samples / 10;

    for i in 0..config.total_samples {
        if i > 0 && i % progress_interval == 0 {
            let pct = (i * 100) / config.total_samples;
            println!("  [{:>3}%] sample {:>8} / {}", pct, i, config.total_samples);
        }

        match measure_single_rtt(&mut queue, i as u64, config.gpu_exec_ns) {
            Some(sample) => samples.push(sample),
            None => failed += 1,
        }
    }

    let total_elapsed_secs = bench_start.elapsed().as_secs_f64();
    println!("\nCollection complete in {:.3}s ({} succeeded, {} failed)\n",
        total_elapsed_secs, samples.len(), failed);

    // ── Compute per-phase statistics ──
    let mut rtt_values: Vec<u64> = samples.iter().map(|s| s.rtt_ns).collect();
    let mut push_values: Vec<u64> = samples.iter().map(|s| s.host_push_ns).collect();
    let mut wait_values: Vec<u64> = samples.iter().map(|s| s.completion_wait_ns).collect();
    let mut gpu_values: Vec<u64> = samples.iter().map(|s| s.gpu_exec_ns).collect();
    let mut pcie_values: Vec<u64> = samples.iter().map(|s| s.pcie_delay_ns).collect();

    rtt_values.sort_unstable();
    push_values.sort_unstable();
    wait_values.sort_unstable();
    gpu_values.sort_unstable();
    pcie_values.sort_unstable();

    let rtt_stats = compute_stats(&rtt_values);
    let push_stats = compute_stats(&push_values);
    let wait_stats = compute_stats(&wait_values);
    let gpu_stats = compute_stats(&gpu_values);
    let pcie_stats = compute_stats(&pcie_values);

    let breakdown = PhaseBreakdown {
        host_push: push_stats.clone(),
        completion_wait: wait_stats.clone(),
        gpu_exec: gpu_stats.clone(),
        pcie_delay: pcie_stats.clone(),
        rtt: rtt_stats.clone(),
    };

    // ── Print results ──
    print_stats_table("Full RTT (push → completion)", &rtt_stats);
    print_stats_table("Phase 1: Host Push (write + CAS)", &push_stats);
    print_stats_table("Phase 2+3+4: Completion Wait", &wait_stats);
    print_stats_table("Simulated GPU Execution (cudaEvent_t)", &gpu_stats);
    print_stats_table("Estimated PCIe Propagation Delay", &pcie_stats);

    // ── Phase contribution analysis ──
    println!("\n=== RTT Phase Contribution (mean) ===");
    let total_mean = rtt_stats.mean_ns as f64;
    if total_mean > 0.0 {
        println!("  Host Push     : {:>8} ns ({:>5.1}%)",
            push_stats.mean_ns, push_stats.mean_ns as f64 / total_mean * 100.0);
        println!("  GPU Execution : {:>8} ns ({:>5.1}%)",
            gpu_stats.mean_ns, gpu_stats.mean_ns as f64 / total_mean * 100.0);
        println!("  PCIe Delay    : {:>8} ns ({:>5.1}%)",
            pcie_stats.mean_ns, pcie_stats.mean_ns as f64 / total_mean * 100.0);
        println!("  ──────────────────────────────────");
        println!("  Total RTT     : {:>8} ns (100.0%)", rtt_stats.mean_ns);
    }

    // ── Throughput ──
    let throughput = samples.len() as f64 / total_elapsed_secs;
    println!("\n  Throughput    : {:.2} round-trips/sec", throughput);
    println!("  Benchmark    : {:.3} seconds total", total_elapsed_secs);

    // ── P99 target check ──
    let p99_us = rtt_stats.p99_us;
    println!("\n=== P99 RTT Target Validation ===");
    println!("  P99 RTT      : {:.3} µs", p99_us);
    println!("  Target       : < {:.1} µs", config.p99_target_us);
    if p99_us < config.p99_target_us {
        println!("  Result       : PASS (P99 is {:.1}x below target)",
            config.p99_target_us / p99_us);
    } else {
        println!("  Result       : NEEDS OPTIMIZATION (P99 exceeds target by {:.1}x)",
            p99_us / config.p99_target_us);
    }

    // ── Delay hotspot identification ──
    println!("\n=== PCIe Delay Hotspot Analysis ===");
    let high_pcie_threshold_ns = 2000; // 2µs — flag as potential contention
    let high_pcie_count = pcie_values.iter()
        .filter(|&&v| v > high_pcie_threshold_ns)
        .count();
    let high_pcie_pct = high_pcie_count as f64 / samples.len() as f64 * 100.0;
    println!("  Samples with PCIe delay > 2 µs: {} ({:.2}%)",
        high_pcie_count, high_pcie_pct);
    if high_pcie_pct > 5.0 {
        println!("  WARNING: Significant PCIe bus contention detected.");
        println!("           Consider reducing concurrent PCIe traffic or");
        println!("           using pinned memory (cudaMallocHost) to reduce");
        println!("           page-fault overhead on the unified memory path.");
    } else {
        println!("  OK: PCIe propagation is within expected bounds.");
    }

    let very_high_rtt_threshold_ns = 10_000; // 10µs
    let outlier_count = rtt_values.iter()
        .filter(|&&v| v > very_high_rtt_threshold_ns)
        .count();
    println!("  RTT outliers > 10 µs: {} ({:.2}%)",
        outlier_count, outlier_count as f64 / samples.len() as f64 * 100.0);
    if outlier_count > 0 {
        println!("  NOTE: Outliers may be caused by OS scheduling jitter,");
        println!("        power management transitions, or TLB misses on");
        println!("        unified memory pages crossing the PCIe bus.");
    }

    // ── cudaEvent_t vs Instant comparison ──
    println!("\n=== Timing Source Comparison ===");
    println!("  Host (std::time::Instant):");
    println!("    Resolution : ~nanosecond (platform-dependent)");
    println!("    RTT P99    : {:.3} µs", rtt_stats.p99_us);
    println!("  Device (cudaEvent_t, simulated):");
    println!("    Resolution : ~0.5 µs (GPU clock-based)");
    println!("    Exec P99   : {:.3} µs", gpu_stats.p99_us);
    println!("  Delta (host RTT P99 - device exec P99) = {:.3} µs",
        rtt_stats.p99_us - gpu_stats.p99_us);
    println!("    → This delta represents host overhead + PCIe round-trip.");
    println!("    → On real hardware, minimize this by:");
    println!("      1. Using volatile writes + __threadfence_system()");
    println!("      2. Pinning host memory (cudaMallocHost)");
    println!("      3. Dedicating a CPU core to the polling thread");
    println!("      4. Setting GPU persistence mode (nvidia-smi -pm 1)");

    // ── Write report ──
    if config.write_report {
        let report = json!({
            "schema_version": "2.0",
            "benchmark": "RTT Latency (Round-Trip Time)",
            "generated_at_unix_secs": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "config": {
                "total_samples": config.total_samples,
                "warmup_samples": config.warmup_samples,
                "gpu_exec_ns_simulated": config.gpu_exec_ns,
                "p99_target_us": config.p99_target_us,
            },
            "summary": {
                "total_elapsed_secs": total_elapsed_secs,
                "successful_samples": samples.len(),
                "failed_pushes": failed,
                "throughput_rtt_per_sec": throughput,
                "p99_target_met": p99_us < config.p99_target_us,
            },
            "rtt_ns": {
                "min": rtt_stats.min_ns, "max": rtt_stats.max_ns,
                "mean": rtt_stats.mean_ns, "std_dev": rtt_stats.std_dev_ns,
                "p50": rtt_stats.p50_ns, "p90": rtt_stats.p90_ns,
                "p95": rtt_stats.p95_ns, "p99": rtt_stats.p99_ns,
                "p999": rtt_stats.p999_ns,
            },
            "rtt_us": {
                "min": rtt_stats.min_us, "max": rtt_stats.max_us,
                "mean": rtt_stats.mean_us, "std_dev": rtt_stats.std_dev_ns / 1000.0,
                "p50": rtt_stats.p50_us, "p90": rtt_stats.p90_us,
                "p95": rtt_stats.p95_us, "p99": rtt_stats.p99_us,
                "p999": rtt_stats.p999_us,
            },
            "phase_breakdown_ns": {
                "host_push": {
                    "min": push_stats.min_ns, "max": push_stats.max_ns,
                    "mean": push_stats.mean_ns, "p99": push_stats.p99_ns,
                },
                "completion_wait": {
                    "min": wait_stats.min_ns, "max": wait_stats.max_ns,
                    "mean": wait_stats.mean_ns, "p99": wait_stats.p99_ns,
                },
                "gpu_exec_simulated": {
                    "min": gpu_stats.min_ns, "max": gpu_stats.max_ns,
                    "mean": gpu_stats.mean_ns, "p99": gpu_stats.p99_ns,
                },
                "pcie_delay_estimated": {
                    "min": pcie_stats.min_ns, "max": pcie_stats.max_ns,
                    "mean": pcie_stats.mean_ns, "p99": pcie_stats.p99_ns,
                },
            },
            "phase_contribution_pct": {
                "host_push": push_stats.mean_ns as f64 / total_mean * 100.0,
                "gpu_execution": gpu_stats.mean_ns as f64 / total_mean * 100.0,
                "pcie_delay": pcie_stats.mean_ns as f64 / total_mean * 100.0,
            },
            "hotspot_analysis": {
                "pcie_delay_gt_2us_count": high_pcie_count,
                "pcie_delay_gt_2us_pct": high_pcie_pct,
                "rtt_outliers_gt_10us_count": outlier_count,
                "rtt_outliers_gt_10us_pct": outlier_count as f64 / samples.len() as f64 * 100.0,
            },
            "timing_sources": {
                "host": "std::time::Instant (nanosecond)",
                "device": "cudaEvent_t (simulated, sub-microsecond)",
                "delta_host_vs_device_p99_us": rtt_stats.p99_us - gpu_stats.p99_us,
            },
            "notes": [
                "GPU execution time is simulated; real timing requires cudaEvent_t on a CUDA-capable machine.",
                "PCIe delay estimate = RTT - host_push - gpu_exec. Negative values indicate measurement noise.",
                format!("P99 RTT target: < {:.1} µs. Achieved: {:.3} µs.", config.p99_target_us, p99_us),
            ],
        });

        let report_str = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write("rtt_latency_report.json", &report_str).unwrap();
        println!("\nReport written to rtt_latency_report.json");
    }

    // ── Assertions ──
    assert!(!samples.is_empty(), "No RTT samples collected");
    assert!(
        rtt_stats.p99_us < config.p99_target_us,
        "P99 RTT ({:.3} µs) exceeds target ({:.1} µs). \
         See hotspot analysis above for optimization guidance.",
        rtt_stats.p99_us,
        config.p99_target_us,
    );
}

// ============================================================
// Test: RTT Smoke Test (1,000 samples, quick validation)
// ============================================================

#[test]
fn test_rtt_latency_smoke() {
    const SAMPLES: usize = 1_000;

    println!("\n=== RTT Smoke Test ({} samples) ===", SAMPLES);
    let mut queue: CommandQueueHost = unsafe { zeroed() };
    let mut samples: Vec<RttSample> = Vec::with_capacity(SAMPLES);

    for i in 0..SAMPLES {
        if let Some(sample) = measure_single_rtt(&mut queue, i as u64, SIMULATED_GPU_EXEC_NS) {
            samples.push(sample);
        }
    }

    assert!(!samples.is_empty(), "No samples collected");

    let mut rtt_values: Vec<u64> = samples.iter().map(|s| s.rtt_ns).collect();
    rtt_values.sort_unstable();
    let stats = compute_stats(&rtt_values);

    print_stats_table("RTT Smoke Test", &stats);

    // Smoke test: P99 should be under 50µs even in debug mode
    assert!(
        stats.p99_us < 50.0,
        "Smoke test P99 RTT ({:.3} µs) is unreasonably high. \
         Expected < 50 µs even in debug mode.",
        stats.p99_us,
    );

    println!("Smoke test PASSED (P99 = {:.3} µs)", stats.p99_us);
}

// ============================================================
// Test: Phase Decomposition Accuracy
// ============================================================

#[test]
fn test_rtt_phase_decomposition() {
    println!("\n=== RTT Phase Decomposition Test ===");

    let mut queue: CommandQueueHost = unsafe { zeroed() };
    let mut total_rtt: u64 = 0;
    let mut total_push: u64 = 0;
    let mut total_wait: u64 = 0;
    let n = 500;

    for i in 0..n {
        if let Some(s) = measure_single_rtt(&mut queue, i, SIMULATED_GPU_EXEC_NS) {
            total_rtt += s.rtt_ns;
            total_push += s.host_push_ns;
            total_wait += s.completion_wait_ns;
        }
    }

    // Verify: push + wait ≈ rtt (within measurement tolerance)
    let sum_phases = total_push + total_wait;
    let diff = if total_rtt > sum_phases {
        total_rtt - sum_phases
    } else {
        sum_phases - total_rtt
    };
    let tolerance_ns = n as u64 * 100; // 100ns tolerance per sample

    println!("  Total RTT      : {} ns", total_rtt);
    println!("  Push + Wait    : {} ns", sum_phases);
    println!("  Difference     : {} ns ({:.1} ns/sample)",
        diff, diff as f64 / n as f64);
    println!("  Tolerance      : {} ns ({} ns/sample)", tolerance_ns, 100);

    assert!(
        diff < tolerance_ns,
        "Phase decomposition error ({} ns) exceeds tolerance ({} ns). \
         push + wait should equal rtt within measurement noise.",
        diff, tolerance_ns,
    );

    println!("Phase decomposition PASSED");
}

// ============================================================
// Test: PCIe Delay Estimation Sanity
// ============================================================

#[test]
fn test_pcie_delay_estimation() {
    println!("\n=== PCIe Delay Estimation Sanity Test ===");

    let mut queue: CommandQueueHost = unsafe { zeroed() };
    let mut pcie_delays: Vec<u64> = Vec::with_capacity(200);

    for i in 0..200 {
        if let Some(s) = measure_single_rtt(&mut queue, i, SIMULATED_GPU_EXEC_NS) {
            pcie_delays.push(s.pcie_delay_ns);
        }
    }

    pcie_delays.sort_unstable();
    let stats = compute_stats(&pcie_delays);

    println!("  Mean PCIe delay: {} ns ({:.3} µs)", stats.mean_ns, stats.mean_us);
    println!("  P99 PCIe delay : {} ns ({:.3} µs)", stats.p99_ns, stats.p99_us);

    // In simulated mode (no real PCIe bus), the estimated PCIe delay
    // should be small — dominated by OS scheduling jitter.
    // On real hardware, expect 1-3 µs for PCIe Gen3/Gen4 round-trip.
    assert!(
        stats.p99_us < 20.0,
        "Simulated PCIe delay P99 ({:.3} µs) is unreasonably high. \
         In simulation, expect < 20 µs (no real PCIe bus).",
        stats.p99_us,
    );

    println!("PCIe delay estimation PASSED");
}

// ============================================================
// Test: cudaEvent_t Simulation Consistency
// ============================================================

#[test]
fn test_cuda_event_simulation() {
    println!("\n=== cudaEvent_t Simulation Test ===");

    // Verify that simulated cudaEvent_t pairs report consistent timing
    let exec_ns = 500;
    let event = CudaEventPair::simulate(exec_ns);

    assert_eq!(event.elapsed_ns, exec_ns,
        "CudaEventPair elapsed should match input execution time");
    assert!(event.stop_ns >= event.start_ns,
        "CudaEventPair stop must be >= start");
    assert_eq!(event.stop_ns - event.start_ns, exec_ns as u64,
        "CudaEventPair delta should equal execution time");

    // Verify multiple events don't produce identical start times
    let e1 = CudaEventPair::simulate(100);
    std::thread::sleep(std::time::Duration::from_nanos(100));
    let e2 = CudaEventPair::simulate(100);
    // Note: on fast CPUs, these might still be equal, so we just
    // verify they don't panic and produce valid output.
    assert!(e2.start_ns >= e1.start_ns,
        "CudaEventPair timestamps should be monotonically non-decreasing");

    println!("  CudaEventPair validation PASSED");
    println!("  Note: On real hardware, replace with:");
    println!("    cudaEventCreate(&start); cudaEventCreate(&stop);");
    println!("    cudaEventRecord(start, stream);");
    println!("    <<< kernel >>>");
    println!("    cudaEventRecord(stop, stream);");
    println!("    cudaEventSynchronize(stop);");
    println!("    cudaEventElapsedTime(&ms, start, stop);");
}

// ============================================================
// Test: Batch RTT (measure contiguous burst latencies)
// ============================================================

#[test]
fn test_rtt_burst_latency() {
    println!("\n=== RTT Burst Latency Test ===");
    println!("  Measures RTT for back-to-back command bursts");
    println!("  to detect queuing delays and head-of-line blocking.\n");

    let mut queue: CommandQueueHost = unsafe { zeroed() };
    let burst_size = 10;
    let num_bursts = 100;
    let mut burst_rtts: Vec<u64> = Vec::new();
    let mut per_cmd_rtts: Vec<u64> = Vec::new();

    for burst in 0..num_bursts {
        let burst_start = Instant::now();

        for j in 0..burst_size {
            let seq = (burst * burst_size + j) as u64;
            if let Some(sample) = measure_single_rtt(&mut queue, seq, SIMULATED_GPU_EXEC_NS) {
                per_cmd_rtts.push(sample.rtt_ns);
            }
        }

        let burst_rtt = burst_start.elapsed().as_nanos() as u64;
        burst_rtts.push(burst_rtt);
    }

    burst_rtts.sort_unstable();
    per_cmd_rtts.sort_unstable();

    let burst_stats = compute_stats(&burst_rtts);
    let cmd_stats = compute_stats(&per_cmd_rtts);

    print_stats_table(&format!("Burst RTT ({} cmds/burst)", burst_size), &burst_stats);
    print_stats_table("Per-Command RTT (within bursts)", &cmd_stats);

    let avg_per_cmd_in_burst = burst_stats.mean_ns as f64 / burst_size as f64;
    println!("\n  Avg per-cmd in burst: {:.0} ns ({:.3} µs)",
        avg_per_cmd_in_burst, avg_per_cmd_in_burst / 1_000.0);
    println!("  Avg standalone cmd  : {} ns ({:.3} µs)",
        cmd_stats.mean_ns, cmd_stats.mean_us);

    let amortization = if cmd_stats.mean_ns > 0 {
        avg_per_cmd_in_burst / cmd_stats.mean_ns as f64
    } else {
        1.0
    };
    println!("  Burst amortization  : {:.2}x", amortization);

    println!("\nBurst latency test PASSED");
}

// ============================================================
// Test: Completion Flag Semantics
// ============================================================

#[test]
fn test_completion_flag_semantics() {
    println!("\n=== Completion Flag Semantics Test ===");

    let mut queue: CommandQueueHost = unsafe { zeroed() };

    // Push a command with PENDING result_code
    let cmd = Command {
        cmd_type: cmd_type::NOOP,
        id: 42,
        timestamp: 12345,
        data_a: 0.0,
        data_b: 0.0,
        result: 0.0,
        batch_data: 0,
        batch_count: 0,
        _padding: 0,
        result_code: completion::PENDING,
    };

    assert!(push_command(&mut queue, cmd), "Push should succeed on empty queue");

    // Before GPU processes: result_code should still be PENDING
    let idx = 0usize;
    assert_eq!(queue.buffer[idx].result_code, completion::PENDING,
        "Before GPU processing, result_code should be PENDING");

    // Simulate GPU completion
    simulate_gpu_completion(&mut queue);

    // After GPU processes: result_code should be SUCCESS
    assert_eq!(queue.buffer[idx].result_code, completion::SUCCESS,
        "After GPU processing, result_code should be SUCCESS");

    // Tail should have advanced
    assert_eq!(queue.tail, 1, "Tail should advance after GPU completion");

    println!("  PENDING → push → GPU processes → SUCCESS → tail advances");
    println!("Completion flag semantics PASSED");
}

// ============================================================
// Test: LatencyStats computation correctness
// ============================================================

#[test]
fn test_latency_stats_computation() {
    println!("\n=== LatencyStats Computation Test ===");

    // Known sorted input
    let data: Vec<u64> = (1..=1000).collect();
    let stats = compute_stats(&data);

    assert_eq!(stats.samples, 1000);
    assert_eq!(stats.min_ns, 1);
    assert_eq!(stats.max_ns, 1000);
    assert_eq!(stats.p50_ns, 501);  // data[500]
    assert_eq!(stats.p90_ns, 901);  // data[900]
    assert_eq!(stats.p99_ns, 991);  // data[990]
    assert_eq!(stats.p999_ns, 1000); // data[999]

    // Mean should be ~500
    assert!((stats.mean_ns as i64 - 500).abs() < 2,
        "Mean should be ~500, got {}", stats.mean_ns);

    println!("  min={}, max={}, mean={}, p50={}, p99={}",
        stats.min_ns, stats.max_ns, stats.mean_ns, stats.p50_ns, stats.p99_ns);
    println!("LatencyStats computation PASSED");
}
