// Memory Alignment Verification and Long-Running Kernel Support
//
// This module provides:
// 1. Compile-time and runtime alignment verification between Rust and CUDA
// 2. Long-running kernel configuration and health monitoring
// 3. Watchdog mechanisms for kernel timeout detection
// 4. Kernel lifecycle management

use crate::cuda_claw::{Command, CommandQueueHost, CommandType, QueueStatus};
use cust::memory::UnifiedBuffer;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;

// ============================================================
// ALIGNMENT VERIFICATION
// ============================================================

/// Alignment verification result
#[derive(Debug, Clone)]
pub struct AlignmentReport {
    pub command_size_matches: bool,
    pub command_queue_size_matches: bool,
    pub command_offset_matches: Vec<(&'static str, bool)>,
    pub command_queue_offset_matches: Vec<(&'static str, bool)>,
    pub overall_valid: bool,
}

/// Verify memory layout alignment between Rust and CUDA
pub fn verify_alignment() -> AlignmentReport {
    let mut report = AlignmentReport {
        command_size_matches: false,
        command_queue_size_matches: false,
        command_offset_matches: Vec::new(),
        command_queue_offset_matches: Vec::new(),
        overall_valid: false,
    };

    // Verify Command size (must be 48 bytes)
    report.command_size_matches = std::mem::size_of::<Command>() == 48;

    // Verify CommandQueueHost size (must be 896 bytes)
    report.command_queue_size_matches = std::mem::size_of::<CommandQueueHost>() == 896;

    // Verify Command field offsets
    macro_rules! check_offset {
        ($ty:ty, $field:ident, $expected:expr) => {
            let uninit = std::mem::MaybeUninit::<$ty>::uninit();
            let ptr = uninit.as_ptr();
            let offset = unsafe {
                (&(*ptr).$field as *const _ as usize) - (ptr as usize)
            };
            (offset == $expected, stringify!($field), offset, $expected)
        };
    }

    let cmd = unsafe { std::mem::MaybeUninit::<Command>::uninit().assume_init() };
    let cmd_ptr = &cmd as *const Command as usize;

    // Command field offset verification
    report.command_offset_matches.push((
        "cmd_type",
        unsafe { (&(*(&cmd as *const Command)).cmd_type as *const _ as usize) - cmd_ptr == 0 }
    ));
    report.command_offset_matches.push((
        "id",
        unsafe { (&(*(&cmd as *const Command)).id as *const _ as usize) - cmd_ptr == 4 }
    ));
    report.command_offset_matches.push((
        "timestamp",
        unsafe { (&(*(&cmd as *const Command)).timestamp as *const _ as usize) - cmd_ptr == 8 }
    ));
    report.command_offset_matches.push((
        "data_a",
        unsafe { (&(*(&cmd as *const Command)).data_a as *const _ as usize) - cmd_ptr == 16 }
    ));
    report.command_offset_matches.push((
        "result_code",
        unsafe { (&(*(&cmd as *const Command)).result_code as *const _ as usize) - cmd_ptr == 44 }
    ));

    // CommandQueueHost field offset verification
    let queue = unsafe { std::mem::MaybeUninit::<CommandQueueHost>::uninit().assume_init() };
    let queue_ptr = &queue as *const CommandQueueHost as usize;

    report.command_queue_offset_matches.push((
        "status",
        unsafe { (&(*(&queue as *const CommandQueueHost)).status as *const _ as usize) - queue_ptr == 0 }
    ));
    report.command_queue_offset_matches.push((
        "head",
        unsafe { (&(*(&queue as *const CommandQueueHost)).head as *const _ as usize) - queue_ptr == 772 }
    ));
    report.command_queue_offset_matches.push((
        "tail",
        unsafe { (&(*(&queue as *const CommandQueueHost)).tail as *const _ as usize) - queue_ptr == 776 }
    ));

    // Overall validity
    report.overall_valid = report.command_size_matches
        && report.command_queue_size_matches
        && report.command_offset_matches.iter().all(|(_, valid)| *valid)
        && report.command_queue_offset_matches.iter().all(|(_, valid)| *valid);

    report
}

/// Print alignment verification report
pub fn print_alignment_report(report: &AlignmentReport) {
    println!("=== Memory Alignment Verification Report ===\n");

    println!("Command Structure:");
    println!("  Size matches (48 bytes): {}", if report.command_size_matches { "✓" } else { "✗" });
    for (field, valid) in &report.command_offset_matches {
        println!("  Offset {}: {}", field, if *valid { "✓" } else { "✗" });
    }

    println!("\nCommandQueue Structure:");
    println!("  Size matches (896 bytes): {}", if report.command_queue_size_matches { "✓" } else { "✗" });
    for (field, valid) in &report.command_queue_offset_matches {
        println!("  Offset {}: {}", field, if *valid { "✓" } else { "✗" });
    }

    println!("\nOverall Valid: {}", if report.overall_valid { "✓ PASS" } else { "✗ FAIL" });
    println!();
}

/// Assert alignment at runtime (panics if invalid)
pub fn assert_alignment() {
    let report = verify_alignment();
    if !report.overall_valid {
        print_alignment_report(&report);
        panic!("Memory layout mismatch detected between Rust and CUDA!");
    }
}

// ============================================================
// LONG-RUNNING KERNEL CONFIGURATION
// ============================================================

/// Configuration for long-running kernel behavior
#[derive(Debug, Clone)]
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

impl Default for KernelConfig {
    fn default() -> Self {
        KernelConfig {
            max_execution_time: Duration::from_secs(3600),      // 1 hour
            health_check_interval: Duration::from_millis(100),   // 100 ms
            watchdog_timeout: Duration::from_secs(10),           // 10 seconds
            auto_restart: true,
            max_restarts: 3,
            enable_health_monitoring: true,
            min_cycles_per_second: 1000,                        // Minimum 1K cycles/sec
        }
    }
}

impl KernelConfig {
    /// Create configuration for short-running kernels (testing)
    pub fn short_running() -> Self {
        KernelConfig {
            max_execution_time: Duration::from_secs(60),
            health_check_interval: Duration::from_millis(10),
            watchdog_timeout: Duration::from_secs(5),
            auto_restart: false,
            max_restarts: 1,
            enable_health_monitoring: true,
            min_cycles_per_second: 100,
        }
    }

    /// Create configuration for long-running kernels (production)
    pub fn long_running() -> Self {
        KernelConfig {
            max_execution_time: Duration::from_secs(86400),     // 24 hours
            health_check_interval: Duration::from_secs(1),
            watchdog_timeout: Duration::from_secs(30),
            auto_restart: true,
            max_restarts: 10,
            enable_health_monitoring: true,
            min_cycles_per_second: 1000,
        }
    }

    /// Create configuration for continuous kernels (server)
    pub fn continuous() -> Self {
        KernelConfig {
            max_execution_time: Duration::from_secs(0),         // Unlimited
            health_check_interval: Duration::from_secs(5),
            watchdog_timeout: Duration::from_secs(60),
            auto_restart: true,
            max_restarts: 100,
            enable_health_monitoring: true,
            min_cycles_per_second: 100,
        }
    }
}

// ============================================================
// KERNEL HEALTH STATUS
// ============================================================

/// Health status of the kernel
#[derive(Debug, Clone, PartialEq)]
pub enum KernelHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Hung,
    Crashed,
    Unknown,
}

/// Detailed health metrics
#[derive(Debug, Clone)]
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

impl Default for KernelHealthMetrics {
    fn default() -> Self {
        KernelHealthMetrics {
            status: KernelHealth::Unknown,
            uptime: Duration::from_secs(0),
            last_cycle_count: 0,
            cycles_per_second: 0.0,
            idle_percentage: 0.0,
            last_health_check: Instant::now(),
            consecutive_failed_checks: 0,
            restart_count: 0,
        }
    }
}

// ============================================================
// WATCHDOG MONITOR
// ============================================================

/// Watchdog for monitoring long-running kernels
pub struct KernelWatchdog {
    config: KernelConfig,
    health_metrics: KernelHealthMetrics,
    start_time: Instant,
    last_check_time: Instant,
    queue: Arc<Mutex<UnifiedBuffer<CommandQueueHost>>>,
    running: bool,
}

impl KernelWatchdog {
    /// Create a new watchdog with default configuration
    pub fn new(
        queue: Arc<Mutex<UnifiedBuffer<CommandQueueHost>>>,
        config: KernelConfig,
    ) -> Self {
        KernelWatchdog {
            config,
            health_metrics: KernelHealthMetrics::default(),
            start_time: Instant::now(),
            last_check_time: Instant::now(),
            queue,
            running: false,
        }
    }

    /// Start the watchdog monitoring
    pub fn start(&mut self) {
        self.running = true;
        self.start_time = Instant::now();
        self.last_check_time = Instant::now();

        if self.config.enable_health_monitoring {
            self.spawn_health_monitor();
        }
    }

    /// Stop the watchdog monitoring
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Spawn background health monitor thread
    fn spawn_health_monitor(&self) {
        let queue = self.queue.clone();
        let config = self.config.clone();
        let interval = self.config.health_check_interval;

        thread::spawn(move || {
            let mut last_cycles = 0u64;
            let mut last_check = Instant::now();
            let mut consecutive_failures = 0u32;

            loop {
                thread::sleep(interval);

                // Read queue statistics
                let (current_cycles, total_cycles, idle_cycles) = {
                    let q = queue.lock().unwrap();
                    (q.commands_processed, q.total_cycles, q.idle_cycles)
                };

                let now = Instant::now();
                let elapsed = now.duration_since(last_check);
                last_check = now;

                // Calculate cycles per second
                let cycle_delta = current_cycles - last_cycles;
                let cycles_per_sec = if elapsed.as_secs() > 0 {
                    cycle_delta as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                // Calculate idle percentage
                let idle_pct = if total_cycles > 0 {
                    (idle_cycles as f64 / total_cycles as f64) * 100.0
                } else {
                    0.0
                };

                // Determine health status
                let healthy = cycles_per_sec >= config.min_cycles_per_second as f64
                    && idle_pct < 90.0;

                if !healthy {
                    consecutive_failures += 1;

                    if consecutive_failures >= 5 {
                        eprintln!("WARNING: Kernel health check failed ({} consecutive failures)", consecutive_failures);
                        eprintln!("  Cycles/sec: {:.2} (minimum: {:.2})", cycles_per_sec, config.min_cycles_per_second);
                        eprintln!("  Idle percentage: {:.1}%", idle_pct);

                        if consecutive_failures >= 10 {
                            eprintln!("ERROR: Kernel appears to be hung or crashed!");

                            if config.auto_restart {
                                eprintln!("Attempting kernel restart...");
                                // Signal watchdog that kernel needs restart
                                break;
                            }
                        }
                    }
                } else {
                    consecutive_failures = 0;
                }

                last_cycles = current_cycles;

                // Check for watchdog timeout
                if consecutive_failures > 0 {
                    let time_since_last_activity = elapsed;
                    if time_since_last_activity > config.watchdog_timeout {
                        eprintln!("ERROR: Watchdog timeout triggered!");
                        eprintln!("  No activity for {:?}", time_since_last_activity);
                        break;
                    }
                }
            }
        });
    }

    /// Perform a single health check
    pub fn check_health(&mut self) -> &KernelHealthMetrics {
        let queue = self.queue.lock().unwrap();

        let total_cycles = queue.total_cycles;
        let idle_cycles = queue.idle_cycles;
        let current_commands = queue.commands_processed;

        drop(queue);

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_check_time);
        self.last_check_time = now;

        // Calculate metrics
        let cycle_delta = current_commands - self.health_metrics.last_cycle_count;
        let cycles_per_second = if elapsed.as_secs_f64() > 0.0 {
            cycle_delta as f64 / elapsed.as_secs_f64()
        } else {
            self.health_metrics.cycles_per_second
        };

        let idle_percentage = if total_cycles > 0 {
            (idle_cycles as f64 / total_cycles as f64) * 100.0
        } else {
            0.0
        };

        // Update health metrics
        self.health_metrics.uptime = now.duration_since(self.start_time);
        self.health_metrics.last_cycle_count = current_commands;
        self.health_metrics.cycles_per_second = cycles_per_second;
        self.health_metrics.idle_percentage = idle_percentage;
        self.health_metrics.last_health_check = now;

        // Determine health status
        self.health_metrics.status = if cycles_per_second >= self.config.min_cycles_per_second as f64
            && idle_percentage < 90.0
        {
            KernelHealth::Healthy
        } else if cycles_per_second > 0 && idle_percentage < 95.0 {
            KernelHealth::Degraded
        } else if cycles_per_second > 0 {
            KernelHealth::Unhealthy
        } else {
            KernelHealth::Hung
        };

        &self.health_metrics
    }

    /// Get current health metrics
    pub fn health_metrics(&self) -> &KernelHealthMetrics {
        &self.health_metrics
    }

    /// Check if kernel is healthy
    pub fn is_healthy(&self) -> bool {
        self.health_metrics.status == KernelHealth::Healthy
    }

    /// Check if kernel should be restarted
    pub fn should_restart(&self) -> bool {
        self.config.auto_restart
            && (self.health_metrics.status == KernelHealth::Hung
                || self.health_metrics.status == KernelHealth::Crashed)
            && self.health_metrics.restart_count < self.config.max_restarts
    }
}

// ============================================================
// KERNEL LIFECYCLE MANAGER
// ============================================================

/// Manages the lifecycle of long-running kernels
pub struct KernelLifecycleManager {
    queue: Arc<Mutex<UnifiedBuffer<CommandQueueHost>>>,
    watchdog: Option<KernelWatchdog>,
    config: KernelConfig,
    running: bool,
}

impl KernelLifecycleManager {
    /// Create a new lifecycle manager
    pub fn new(
        queue: Arc<Mutex<UnifiedBuffer<CommandQueueHost>>>,
        config: KernelConfig,
    ) -> Self {
        KernelLifecycleManager {
            queue,
            watchdog: None,
            config,
            running: false,
        }
    }

    /// Start the kernel lifecycle management
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.running {
            return Ok(());
        }

        // Create and start watchdog
        let mut watchdog = KernelWatchdog::new(self.queue.clone(), self.config.clone());
        watchdog.start();
        self.watchdog = Some(watchdog);

        self.running = true;
        Ok(())
    }

    /// Stop the kernel lifecycle management
    pub fn stop(&mut self) {
        if let Some(mut watchdog) = self.watchdog.take() {
            watchdog.stop();
        }
        self.running = false;
    }

    /// Check if kernel needs restart
    pub fn check_and_restart(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(ref mut watchdog) = self.watchdog {
            watchdog.check_health();

            if watchdog.should_restart() {
                println!("Initiating kernel restart...");

                // Send shutdown command
                let shutdown_cmd = Command::new(CommandType::Shutdown, 999);
                let queue = self.queue.lock().unwrap();

                // Write command
                let idx = queue.head as usize;
                unsafe {
                    let queue_ptr = &*queue as *const CommandQueueHost as *mut CommandQueueHost;
                    (*queue_ptr).commands[idx] = shutdown_cmd;
                    (*queue_ptr).head = ((*queue_ptr).head + 1) % 16;
                    (*queue_ptr).status = QueueStatus::Ready as u32;
                }

                // Wait for graceful shutdown
                thread::sleep(Duration::from_millis(100));

                // Increment restart counter
                self.watchdog.as_mut().unwrap().health_metrics.restart_count += 1;

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get health metrics
    pub fn health_metrics(&self) -> Option<&KernelHealthMetrics> {
        self.watchdog.as_ref().map(|w| w.health_metrics())
    }

    /// Check if kernel is healthy
    pub fn is_healthy(&self) -> bool {
        self.watchdog.as_ref().map(|w| w.is_healthy()).unwrap_or(false)
    }
}

impl Drop for KernelLifecycleManager {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================
// ALIGNMENT TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_alignment() {
        let report = verify_alignment();
        assert!(report.command_size_matches, "Command size mismatch");
        assert!(report.overall_valid, "Overall alignment failed");
    }

    #[test]
    fn test_command_queue_alignment() {
        let report = verify_alignment();
        assert!(report.command_queue_size_matches, "CommandQueue size mismatch");
    }

    #[test]
    fn test_alignment_does_not_panic() {
        // This test ensures alignment verification doesn't panic
        assert_alignment();
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.max_execution_time, Duration::from_secs(3600));
        assert_eq!(config.watchdog_timeout, Duration::from_secs(10));
        assert!(config.auto_restart);
    }

    #[test]
    fn test_kernel_config_short_running() {
        let config = KernelConfig::short_running();
        assert_eq!(config.max_execution_time, Duration::from_secs(60));
        assert!(!config.auto_restart);
    }

    #[test]
    fn test_kernel_config_long_running() {
        let config = KernelConfig::long_running();
        assert_eq!(config.max_execution_time, Duration::from_secs(86400));
        assert!(config.auto_restart);
        assert_eq!(config.max_restarts, 10);
    }

    #[test]
    fn test_kernel_config_continuous() {
        let config = KernelConfig::continuous();
        assert_eq!(config.max_execution_time, Duration::from_secs(0));
        assert_eq!(config.max_restarts, 100);
    }
}
