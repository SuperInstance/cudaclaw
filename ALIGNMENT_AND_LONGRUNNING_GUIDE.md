# Memory Alignment and Long-Running Kernel Support

## Overview

This guide covers two critical aspects of the CudaClaw system:

1. **Memory Alignment Verification** - Ensuring `#[repr(C)]` compatibility between Rust and CUDA
2. **Long-Running Kernel Support** - Configuration and monitoring for persistent GPU kernels

## Part 1: Memory Alignment Verification

### Why Alignment Matters

When sharing memory between Rust and CUDA, exact memory layout compatibility is critical:

```
┌─────────────────────────────────────────────────────────────┐
│                   Memory Layout Mismatch                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Rust expects:                                               │
│  ┌────────┬────────┬────────┬────────┐                      │
│  │ type   │ id     │ ts     │ data   │                      │
│  │ (4B)   │ (4B)   │ (8B)   │ (32B)  │                      │
│  └────────┴────────┴────────┴────────┘                      │
│                                                              │
│  CUDA expects:                                              │
│  ┌────────┬────────┬────────┬────────┐                      │
│  │ type   │ id     │ ts     │ data   │                      │
│  │ (4B)   │ (8B) ❌ │ (8B)   │ (28B) ❌│                     │
│  └────────┴────────┴────────┴────────┘                      │
│                                                              │
│  Result: GPU KERNEL CRASH 💥                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### #[repr(C)] in Rust

`#[repr(C)]` guarantees C-compatible memory layout:

```rust
#[repr(C)]
pub struct Command {
    pub cmd_type: u32,          // offset 0,  4 bytes
    pub id: u32,                // offset 4,  4 bytes
    pub timestamp: u64,         // offset 8,  8 bytes
    pub data_a: f32,            // offset 16, 4 bytes
    pub data_b: f32,            // offset 20, 4 bytes
    pub result: f32,            // offset 24, 4 bytes
    pub batch_data: u64,        // offset 28, 8 bytes
    pub batch_count: u32,       // offset 36, 4 bytes
    pub _padding: u32,          // offset 40, 4 bytes
    pub result_code: u32,       // offset 44, 4 bytes
}
// Total: 48 bytes (padded to multiple of largest alignment)
```

**Without #[repr(C)]**:
- Rust is free to reorder fields
- Rust may add different padding
- Layout is unpredictable

**With #[repr(C)]**:
- Fields ordered exactly as declared
- Standard C padding rules apply
- Layout matches exactly what C/C++ expects

### Alignment Specifications

| Structure | Size | Alignment | Cache Line |
|-----------|------|-----------|------------|
| Command | 48 bytes | 32-byte | 1 (partial) |
| CommandQueue | 896 bytes | 128-byte | 7 (exact) |

**Why 32-byte alignment for Command?**
- Optimal for L1 cache line size (typically 32-128 bytes)
- Ensures efficient memory access patterns
- Prevents cache line splitting

**Why 128-byte alignment for CommandQueue?**
- Matches typical L2 cache line size
- Prevents false sharing between queues
- Aligns to memory controller granularity

### Verification System

#### Compile-Time Verification (CUDA)

```cpp
// Size verification
static_assert(sizeof(Command) == 48, "Command must be 48 bytes");
static_assert(sizeof(CommandQueue) == 896, "CommandQueue must be 896 bytes");

// Field offset verification
#define VERIFY_OFFSET(type, member, expected) \
    static_assert(offsetof(type, member) == (expected), \
        "Offset mismatch for " #member)

VERIFY_OFFSET(Command, type, 0);
VERIFY_OFFSET(Command, id, 4);
VERIFY_OFFSET(Command, timestamp, 8);
VERIFY_OFFSET(Command, result_code, 44);
```

#### Runtime Verification (Rust)

```rust
pub fn verify_alignment() -> AlignmentReport {
    let report = AlignmentReport {
        command_size_matches: std::mem::size_of::<Command>() == 48,
        command_queue_size_matches: std::mem::size_of::<CommandQueueHost>() == 896,
        // ... field offset verification
    };
    report
}

pub fn assert_alignment() {
    let report = verify_alignment();
    if !report.overall_valid {
        panic!("Memory layout mismatch detected!");
    }
}
```

### Verification Report

```
=== Memory Alignment Verification Report ===

Command Structure:
  Size matches (48 bytes): ✓
  Offset cmd_type: ✓
  Offset id: ✓
  Offset timestamp: ✓
  Offset data_a: ✓
  Offset result_code: ✓

CommandQueue Structure:
  Size matches (896 bytes): ✓
  Offset status: ✓
  Offset head: ✓
  Offset tail: ✓
  Offset commands_processed: ✓
  Offset total_cycles: ✓
  Offset idle_cycles: ✓
  Offset current_strategy: ✓

Overall Valid: ✓ PASS
```

## Part 2: Long-Running Kernel Support

### Kernel Configuration

The system supports three configuration modes:

```rust
pub struct KernelConfig {
    /// Maximum time kernel can run before watchdog check
    pub max_execution_time: Duration,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Watchdog timeout before kernel is considered hung
    pub watchdog_timeout: Duration,

    /// Enable automatic kernel restart on hang
    pub auto_restart: bool,

    /// Maximum restart attempts before giving up
    pub max_restarts: u32,

    /// Enable kernel health monitoring
    pub enable_health_monitoring: bool,

    /// Minimum cycles per second for healthy kernel
    pub min_cycles_per_second: u64,
}
```

### Configuration Presets

#### 1. Short-Running (Testing)

```rust
let config = KernelConfig::short_running();

// Characteristics:
// - Max execution time: 60 seconds
// - Health check interval: 10 ms
// - Watchdog timeout: 5 seconds
// - Auto-restart: DISABLED
// - Max restarts: 1
// - Min cycles/sec: 100

// Use case: Unit tests, integration tests, development
```

#### 2. Long-Running (Production)

```rust
let config = KernelConfig::long_running();

// Characteristics:
// - Max execution time: 24 hours
// - Health check interval: 1 second
// - Watchdog timeout: 30 seconds
// - Auto-restart: ENABLED
// - Max restarts: 10
// - Min cycles/sec: 1,000

// Use case: Batch processing, data pipelines, analytics
```

#### 3. Continuous (Server)

```rust
let config = KernelConfig::continuous();

// Characteristics:
// - Max execution time: UNLIMITED
// - Health check interval: 5 seconds
// - Watchdog timeout: 60 seconds
// - Auto-restart: ENABLED
// - Max restarts: 100
// - Min cycles/sec: 100

// Use case: Always-on services, real-time processing, web servers
```

### Health Monitoring

The system continuously monitors kernel health:

```
┌─────────────────────────────────────────────────────────────┐
│                   Health Monitoring Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Every health_check_interval:                               │
│  1. Read queue statistics (cycles, commands, idle)          │
│  2. Calculate metrics:                                      │
│     - Cycles per second = (current - last) / interval       │
│     - Idle percentage = (idle / total) * 100                │
│  3. Evaluate health:                                        │
│     ✓ Healthy: cycles/sec >= min AND idle < 90%             │
│     ⚠ Degraded: cycles/sec < min OR idle >= 90%            │
│     ✗ Unhealthy: cycles/sec very low OR idle very high     │
│     💀 Hung: cycles/sec = 0                                 │
│  4. Take action:                                            │
│     - If degraded: Log warning                              │
│     - If unhealthy: Log error, check for restart            │
│     - If hung: Trigger watchdog timeout                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Health Status Definitions

| Status | Description | Condition |
|--------|-------------|-----------|
| **Healthy** | Kernel operating normally | cycles/sec ≥ min AND idle < 90% |
| **Degraded** | Reduced performance | cycles/sec < min OR idle ≥ 90% |
| **Unhealthy** | Severely degraded | cycles/sec very low OR idle very high |
| **Hung** | No response | cycles/sec = 0 |
| **Crashed** | Kernel terminated | Communication lost |

### Watchdog Mechanism

The watchdog protects against hung kernels:

```
Watchdog State Machine:
┌──────────┐
│  START   │
└─────┬────┘
      │
      ▼
┌──────────┐    Health check passes
│  HEALTHY │─────────────────────────┐
└─────┬────┘                         │
      │                             │
      ▼                             ▼
┌──────────┐    Health check fails    │
│DEGRADED │◄──────────────────────────┘
└─────┬────┘
      │
      ▼
┌──────────┐    Consecutive failures >= 5
│UNHEALTHY │
└─────┬────┘
      │
      ▼
┌──────────┐    Watchdog timeout OR failures >= 10
│   HUNG   │─────────────────┐
└─────┬────┘                 │
      │                       │
      │   Auto-restart?       │
      │   enabled?            │
      ├─────────────┬─────────┘
      │ YES         │ NO
      ▼             ▼
┌──────────┐   ┌──────────┐
│ RESTART  │   │  SHUTDOWN│
└──────────┘   └──────────┘
```

### Lifecycle Management

```rust
// Create lifecycle manager
let config = KernelConfig::long_running();
let mut lifecycle_manager = KernelLifecycleManager::new(queue, config);

// Start monitoring
lifecycle_manager.start()?;

// Periodic health checks
loop {
    std::thread::sleep(Duration::from_secs(1));

    // Check health and restart if needed
    if lifecycle_manager.check_and_restart()? {
        println!("Kernel restarted successfully");
        break;
    }

    // Get current health metrics
    if let Some(metrics) = lifecycle_manager.health_metrics() {
        println!("Status: {:?}", metrics.status);
        println!("Cycles/sec: {:.2}", metrics.cycles_per_second);
        println!("Idle: {:.1}%", metrics.idle_percentage);
    }
}
```

### Kernel Health Metrics

```rust
pub struct KernelHealthMetrics {
    pub status: KernelHealth,
    pub uptime: Duration,
    pub last_cycle_count: u64,
    pub cycles_per_second: f64,
    pub idle_percentage: f64,
    pub last_health_check: Instant,
    pub consecutive_failed_checks: u32,
    pub restart_count: u32,
}
```

**Example Output:**

```
Health Check [1]:
  Status: Healthy
  Uptime: 1.234s
  Cycles/sec: 54321.45
  Idle percentage: 12.3%

Health Check [2]:
  Status: Healthy
  Uptime: 2.456s
  Cycles/sec: 56789.12
  Idle percentage: 10.5%

Health Check [3]:
  Status: Degraded
  Uptime: 3.678s
  Cycles/sec: 987.65
  Idle percentage: 85.2%

Health Check [4]:
  Status: Unhealthy
  Uptime: 4.890s
  Cycles/sec: 123.45
  Idle percentage: 95.1%

⚠  WARNING: Kernel health check failed (5 consecutive failures)
    Cycles/sec: 123.45 (minimum: 1000.00)
    Idle percentage: 95.1%
```

## Usage Examples

### Example 1: Verify Alignment Before Use

```rust
use alignment::{assert_alignment, verify_alignment, print_alignment_report};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Verify alignment at startup
    let report = verify_alignment();
    print_alignment_report(&report);

    // Assert alignment (panics if invalid)
    assert_alignment();

    // Now safe to use CUDA
    let executor = CudaClawExecutor::new()?;
    // ...
}
```

### Example 2: Short-Running Kernel (Testing)

```rust
use alignment::{KernelConfig, KernelLifecycleManager};

#[test]
fn test_kernel_execution() {
    let config = KernelConfig::short_running();
    let mut lifecycle = KernelLifecycleManager::new(queue, config);

    lifecycle.start()?;

    // Run test...
    let result = dispatcher.dispatch_sync(test_cmd)?;
    assert!(result.success);

    // Health check
    assert!(lifecycle.is_healthy());
}
```

### Example 3: Long-Running Kernel (Production)

```rust
use alignment::{KernelConfig, KernelLifecycleManager};

fn run_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    let config = KernelConfig::long_running();
    let mut lifecycle = KernelLifecycleManager::new(queue, config);

    lifecycle.start()?;

    // Process batches
    for batch in batches {
        // Check health before each batch
        if lifecycle.check_and_restart()? {
            println!("Kernel was restarted, continuing...");
        }

        // Process batch
        let results = dispatcher.dispatch_batch(batch)?;

        // Monitor progress
        if let Some(metrics) = lifecycle.health_metrics() {
            println!("Processed batch, status: {:?}", metrics.status);
        }
    }

    Ok(())
}
```

### Example 4: Continuous Kernel (Server)

```rust
use alignment::{KernelConfig, KernelLifecycleManager};

fn run_server() -> Result<(), Box<dyn std::error::Error>> {
    let config = KernelConfig::continuous();
    let mut lifecycle = KernelLifecycleManager::new(queue, config);

    lifecycle.start()?;

    // Run forever
    loop {
        std::thread::sleep(Duration::from_secs(5));

        // Periodic health check
        if let Some(metrics) = lifecycle.health_metrics() {
            match metrics.status {
                KernelHealth::Healthy => {
                    // Normal operation
                }
                KernelHealth::Degraded => {
                    eprintln!("Warning: Kernel performance degraded");
                }
                KernelHealth::Unhealthy | KernelHealth::Hung => {
                    eprintln!("Critical: Kernel unhealthy, attempting restart...");
                    lifecycle.check_and_restart()?;
                }
                _ => {}
            }
        }
    }
}
```

## Best Practices

### 1. Always Verify Alignment

```rust
// ✅ GOOD: Verify at startup
fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_alignment();
    // ... proceed with CUDA operations
}

// ❌ BAD: Skip verification
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // May crash if alignment is wrong!
    let executor = CudaClawExecutor::new()?;
}
```

### 2. Choose Appropriate Configuration

```rust
// ✅ GOOD: Use appropriate config for use case
let config = match env {
    Env::Test => KernelConfig::short_running(),
    Env::Production => KernelConfig::long_running(),
    Env::Server => KernelConfig::continuous(),
};

// ❌ BAD: Use wrong config for use case
let config = KernelConfig::short_running();  // For 24-hour job!
```

### 3. Monitor Health Regularly

```rust
// ✅ GOOD: Regular health checks
loop {
    process_batch()?;

    if let Some(metrics) = lifecycle.health_metrics() {
        if metrics.status != KernelHealth::Healthy {
            handle_degraded_performance(metrics)?;
        }
    }
}

// ❌ BAD: Never check health
loop {
    process_batch()?;  // May hang or crash!
}
```

### 4. Handle Failures Gracefully

```rust
// ✅ GOOD: Handle restarts
if lifecycle.check_and_restart()? {
    println!("Kernel restarted, continuing...");
    continue;  // Retry operation
}

// ❌ BAD: Ignore failures
let _ = lifecycle.check_and_restart();
// ... continue as if nothing happened
```

## Troubleshooting

### Issue: Alignment Mismatch Detected

**Symptoms:**
- Panic: "Memory layout mismatch detected!"
- Alignment report shows failures

**Solutions:**
1. Check `#[repr(C)]` is present on all shared structs
2. Verify field order matches between Rust and CUDA
3. Check padding fields are correctly placed
4. Rebuild both Rust and CUDA code

### Issue: Kernel Hung Detected

**Symptoms:**
- Status: Hung
- Cycles/sec: 0
- Watchdog timeout triggered

**Solutions:**
1. Increase watchdog timeout for heavy workloads
2. Check kernel logic for infinite loops
3. Verify GPU is not overloaded
4. Enable auto-restart

### Issue: Poor Performance Detected

**Symptoms:**
- Status: Degraded or Unhealthy
- High idle percentage
- Low cycles/sec

**Solutions:**
1. Check GPU utilization with `nvidia-smi`
2. Profile kernel with `nsys` to find bottlenecks
3. Reduce batch size if memory is constrained
4. Increase `min_cycles_per_second` threshold

### Issue: Frequent Restarts

**Symptoms:**
- High restart count
- Kernel keeps restarting

**Solutions:**
1. Lower `min_cycles_per_second` threshold
2. Increase watchdog timeout
3. Check for driver/hardware issues
4. Review kernel logs for errors

## Configuration Tuning

### Tuning health_check_interval

```rust
// Fast health checks (development)
config.health_check_interval = Duration::from_millis(10);

// Normal health checks (production)
config.health_check_interval = Duration::from_secs(1);

// Slow health checks (resource-constrained)
config.health_check_interval = Duration::from_secs(5);
```

**Trade-offs:**
- Too fast: High CPU overhead, potential false positives
- Too slow: Slow detection of issues, longer recovery time

### Tuning watchdog_timeout

```rust
// Fast watchdog (low-latency systems)
config.watchdog_timeout = Duration::from_secs(10);

// Normal watchdog (general purpose)
config.watchdog_timeout = Duration::from_secs(30);

// Slow watchdog (heavy workloads)
config.watchdog_timeout = Duration::from_secs(60);
```

**Trade-offs:**
- Too fast: May trigger on legitimate slow operations
- Too slow: Long time to detect hung kernel

### Tuning min_cycles_per_second

```rust
// High-performance GPU (e.g., A100, H100)
config.min_cycles_per_second = 100000;

// Mid-range GPU (e.g., RTX 3080, RTX 4090)
config.min_cycles_per_second = 10000;

// Low-end GPU (e.g., GTX 1660, integrated GPU)
config.min_cycles_per_second = 1000;
```

**Trade-offs:**
- Too high: False positives on slower systems
- Too low: May not detect performance issues

## Summary

### Memory Alignment
- ✅ Use `#[repr(C)]` on all shared structures
- ✅ Verify alignment at compile time (CUDA) and runtime (Rust)
- ✅ Match field order exactly between Rust and CUDA
- ✅ Add padding fields as needed for proper alignment

### Long-Running Kernels
- ✅ Choose appropriate configuration preset
- ✅ Enable health monitoring for production use
- ✅ Configure watchdog timeout based on workload
- ✅ Set appropriate cycles/sec threshold for GPU
- ✅ Handle restarts gracefully in application logic

---

**Last Updated:** 2026-03-16
**Status:** Production Ready ✅
**Compatibility:** CUDA 11.0+, Rust 1.70+
