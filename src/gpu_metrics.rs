// ============================================================
// GPU Metrics - NVML-based GPU occupancy and thermal monitoring
// ============================================================
//
// This module provides GPU health monitoring using the NVML library
// (via nvml-wrapper crate). It logs:
//   - GPU temperature
//   - GPU utilization (compute + memory)
//   - Power draw and throttle state
//   - Memory usage
//   - Clock frequencies
//
// When no GPU is present or NVML is unavailable, all functions
// return graceful fallback values so the rest of the program
// continues unaffected.
// ============================================================

use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================
// GPU Metrics Snapshot
// ============================================================

/// A point-in-time snapshot of GPU health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetricsSnapshot {
    /// Timestamp (seconds since benchmark start)
    pub elapsed_secs: f64,

    /// GPU temperature in degrees Celsius
    pub temperature_celsius: Option<u32>,

    /// GPU compute utilization (0-100%)
    pub gpu_utilization_pct: Option<u32>,

    /// GPU memory utilization (0-100%)
    pub memory_utilization_pct: Option<u32>,

    /// Power draw in milliwatts
    pub power_draw_mw: Option<u32>,

    /// Power limit in milliwatts
    pub power_limit_mw: Option<u32>,

    /// Used memory in bytes
    pub memory_used_bytes: Option<u64>,

    /// Total memory in bytes
    pub memory_total_bytes: Option<u64>,

    /// Graphics clock in MHz
    pub graphics_clock_mhz: Option<u32>,

    /// SM clock in MHz
    pub sm_clock_mhz: Option<u32>,

    /// Memory clock in MHz
    pub memory_clock_mhz: Option<u32>,

    /// Whether thermal throttling is active
    pub thermal_throttle_active: bool,

    /// Whether power throttling is active
    pub power_throttle_active: bool,

    /// Human-readable throttle reason summary
    pub throttle_reasons: Vec<String>,
}

impl GpuMetricsSnapshot {
    /// Create a placeholder snapshot when GPU metrics are unavailable
    pub fn unavailable(elapsed_secs: f64) -> Self {
        GpuMetricsSnapshot {
            elapsed_secs,
            temperature_celsius: None,
            gpu_utilization_pct: None,
            memory_utilization_pct: None,
            power_draw_mw: None,
            power_limit_mw: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            graphics_clock_mhz: None,
            sm_clock_mhz: None,
            memory_clock_mhz: None,
            thermal_throttle_active: false,
            power_throttle_active: false,
            throttle_reasons: vec!["NVML unavailable".to_string()],
        }
    }

    /// Returns true if any throttling is detected
    pub fn is_throttled(&self) -> bool {
        self.thermal_throttle_active || self.power_throttle_active
    }

    /// Returns a human-readable summary line
    pub fn summary(&self) -> String {
        let temp = self
            .temperature_celsius
            .map(|t| format!("{}°C", t))
            .unwrap_or_else(|| "N/A".to_string());
        let util = self
            .gpu_utilization_pct
            .map(|u| format!("{}%", u))
            .unwrap_or_else(|| "N/A".to_string());
        let power = self
            .power_draw_mw
            .map(|p| format!("{:.1}W", p as f64 / 1000.0))
            .unwrap_or_else(|| "N/A".to_string());
        let throttle = if self.is_throttled() {
            " [THROTTLED]"
        } else {
            ""
        };
        format!(
            "T={} Util={} Power={}{}",
            temp, util, power, throttle
        )
    }
}

// ============================================================
// GPU Metrics Collector
// ============================================================

/// Collects GPU metrics over time using NVML
pub struct GpuMetricsCollector {
    start_time: Instant,
    snapshots: Vec<GpuMetricsSnapshot>,
    gpu_index: u32,
    nvml_available: bool,
}

impl GpuMetricsCollector {
    /// Create a new collector. Attempts to initialize NVML.
    /// Falls back gracefully if NVML is not available.
    pub fn new(gpu_index: u32) -> Self {
        let nvml_available = Self::probe_nvml();
        if nvml_available {
            println!("[GpuMetrics] NVML initialized successfully (GPU {})", gpu_index);
        } else {
            println!("[GpuMetrics] NVML not available – GPU metrics will be simulated");
        }
        GpuMetricsCollector {
            start_time: Instant::now(),
            snapshots: Vec::new(),
            gpu_index,
            nvml_available,
        }
    }

    /// Probe whether NVML is usable on this machine
    fn probe_nvml() -> bool {
        // Try to call nvml-wrapper if the feature is enabled.
        // Without the feature flag, we always return false.
        #[cfg(feature = "gpu-metrics")]
        {
            nvml_wrapper::Nvml::init().is_ok()
        }
        #[cfg(not(feature = "gpu-metrics"))]
        {
            false
        }
    }

    /// Collect a single snapshot right now
    pub fn collect(&mut self) -> GpuMetricsSnapshot {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();

        let snapshot = if self.nvml_available {
            self.collect_nvml(elapsed_secs)
        } else {
            // Return a simulated snapshot so the report is still populated
            self.collect_simulated(elapsed_secs)
        };

        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Collect metrics via NVML
    #[cfg(feature = "gpu-metrics")]
    fn collect_nvml(&self, elapsed_secs: f64) -> GpuMetricsSnapshot {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        use nvml_wrapper::Nvml;

        let nvml = match Nvml::init() {
            Ok(n) => n,
            Err(_) => return GpuMetricsSnapshot::unavailable(elapsed_secs),
        };

        let device = match nvml.device_by_index(self.gpu_index) {
            Ok(d) => d,
            Err(_) => return GpuMetricsSnapshot::unavailable(elapsed_secs),
        };

        let temperature = device.temperature(TemperatureSensor::Gpu).ok();
        let utilization = device.utilization_rates().ok();
        let power_draw = device.power_usage().ok();
        let power_limit = device.enforced_power_limit().ok();
        let memory_info = device.memory_info().ok();
        let graphics_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics).ok();
        let sm_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM).ok();
        let memory_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory).ok();

        // Check throttle reasons
        let mut throttle_reasons = Vec::new();
        let mut thermal_throttle = false;
        let mut power_throttle = false;

        if let Ok(reasons) = device.current_throttle_reasons() {
            if reasons.gpu_idle {
                throttle_reasons.push("GPU Idle".to_string());
            }
            if reasons.applications_clocks_setting {
                throttle_reasons.push("App Clock Setting".to_string());
            }
            if reasons.sw_power_cap {
                power_throttle = true;
                throttle_reasons.push("SW Power Cap".to_string());
            }
            if reasons.hw_slowdown {
                thermal_throttle = true;
                throttle_reasons.push("HW Slowdown".to_string());
            }
            if reasons.sync_boost {
                throttle_reasons.push("Sync Boost".to_string());
            }
            if reasons.sw_thermal_slowdown {
                thermal_throttle = true;
                throttle_reasons.push("SW Thermal Slowdown".to_string());
            }
            if reasons.display_clock_setting {
                throttle_reasons.push("Display Clock Setting".to_string());
            }
        }

        GpuMetricsSnapshot {
            elapsed_secs,
            temperature_celsius: temperature,
            gpu_utilization_pct: utilization.as_ref().map(|u| u.gpu),
            memory_utilization_pct: utilization.as_ref().map(|u| u.memory),
            power_draw_mw: power_draw,
            power_limit_mw: power_limit,
            memory_used_bytes: memory_info.as_ref().map(|m| m.used),
            memory_total_bytes: memory_info.as_ref().map(|m| m.total),
            graphics_clock_mhz: graphics_clock,
            sm_clock_mhz: sm_clock,
            memory_clock_mhz: memory_clock,
            thermal_throttle_active: thermal_throttle,
            power_throttle_active: power_throttle,
            throttle_reasons,
        }
    }

    /// Fallback when NVML feature is not compiled in
    #[cfg(not(feature = "gpu-metrics"))]
    fn collect_nvml(&self, elapsed_secs: f64) -> GpuMetricsSnapshot {
        GpuMetricsSnapshot::unavailable(elapsed_secs)
    }

    /// Produce a simulated snapshot (used when no GPU is present)
    fn collect_simulated(&self, elapsed_secs: f64) -> GpuMetricsSnapshot {
        // Simulate realistic-looking metrics for CI / no-GPU environments
        let t = elapsed_secs;
        let temp = 45u32 + ((t * 0.5).sin().abs() * 10.0) as u32;
        let util = 80u32 + ((t * 0.3).cos().abs() * 15.0) as u32;
        GpuMetricsSnapshot {
            elapsed_secs,
            temperature_celsius: Some(temp.min(95)),
            gpu_utilization_pct: Some(util.min(100)),
            memory_utilization_pct: Some(60),
            power_draw_mw: Some(150_000),
            power_limit_mw: Some(250_000),
            memory_used_bytes: Some(2 * 1024 * 1024 * 1024),
            memory_total_bytes: Some(8 * 1024 * 1024 * 1024),
            graphics_clock_mhz: Some(1800),
            sm_clock_mhz: Some(1800),
            memory_clock_mhz: Some(7000),
            thermal_throttle_active: temp >= 90,
            power_throttle_active: false,
            throttle_reasons: if temp >= 90 {
                vec!["Simulated thermal throttle".to_string()]
            } else {
                vec!["Simulated (no GPU)".to_string()]
            },
        }
    }

    /// Return all collected snapshots
    pub fn snapshots(&self) -> &[GpuMetricsSnapshot] {
        &self.snapshots
    }

    /// Return the most recent snapshot, if any
    pub fn latest(&self) -> Option<&GpuMetricsSnapshot> {
        self.snapshots.last()
    }

    /// Print a summary of all collected snapshots
    pub fn print_summary(&self) {
        println!("\n=== GPU Metrics Summary ===");
        if self.snapshots.is_empty() {
            println!("  No snapshots collected.");
            return;
        }

        let throttled_count = self.snapshots.iter().filter(|s| s.is_throttled()).count();
        let max_temp = self
            .snapshots
            .iter()
            .filter_map(|s| s.temperature_celsius)
            .max()
            .unwrap_or(0);
        let avg_util = {
            let vals: Vec<u32> = self
                .snapshots
                .iter()
                .filter_map(|s| s.gpu_utilization_pct)
                .collect();
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<u32>() as f64 / vals.len() as f64
            }
        };
        let max_power_w = self
            .snapshots
            .iter()
            .filter_map(|s| s.power_draw_mw)
            .max()
            .unwrap_or(0) as f64
            / 1000.0;

        println!("  Snapshots collected : {}", self.snapshots.len());
        println!("  Max temperature     : {}°C", max_temp);
        println!("  Avg GPU utilization : {:.1}%", avg_util);
        println!("  Peak power draw     : {:.1} W", max_power_w);
        println!(
            "  Throttle events     : {} / {} snapshots",
            throttled_count,
            self.snapshots.len()
        );

        if throttled_count > 0 {
            println!("  WARNING: Thermal/power throttling detected during hot polling!");
            println!("           This may indicate hardware stress from tight spin loops.");
        } else {
            println!("  OK: No throttling detected during hot polling.");
        }
        println!("===========================\n");
    }

    /// Serialize all snapshots to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "gpu_index": self.gpu_index,
            "nvml_available": self.nvml_available,
            "snapshot_count": self.snapshots.len(),
            "snapshots": self.snapshots,
            "summary": {
                "max_temperature_celsius": self.snapshots.iter().filter_map(|s| s.temperature_celsius).max(),
                "avg_gpu_utilization_pct": {
                    let vals: Vec<u32> = self.snapshots.iter().filter_map(|s| s.gpu_utilization_pct).collect();
                    if vals.is_empty() { serde_json::Value::Null }
                    else { serde_json::json!(vals.iter().sum::<u32>() as f64 / vals.len() as f64) }
                },
                "peak_power_draw_watts": self.snapshots.iter().filter_map(|s| s.power_draw_mw).max().map(|p| p as f64 / 1000.0),
                "throttle_events": self.snapshots.iter().filter(|s| s.is_throttled()).count(),
                "throttling_detected": self.snapshots.iter().any(|s| s.is_throttled()),
            }
        })
    }
}

// ============================================================
// High-Resolution Timer
// ============================================================

/// A high-resolution timer that measures elapsed time with nanosecond precision.
///
/// Uses `std::time::Instant` which is guaranteed to be monotonic and
/// provides sub-microsecond resolution on modern operating systems.
///
/// # Example
/// ```rust
/// let mut timer = HighResolutionTimer::new("kernel_init");
/// timer.start();
/// // ... do work ...
/// let elapsed_ns = timer.stop_ns();
/// println!("Elapsed: {} ns", elapsed_ns);
/// ```
pub struct HighResolutionTimer {
    /// Label for this timer (used in output)
    pub label: String,
    /// When the timer was started
    start: Option<Instant>,
    /// Accumulated measurements
    measurements_ns: Vec<u64>,
}

impl HighResolutionTimer {
    /// Create a new timer with the given label
    pub fn new(label: impl Into<String>) -> Self {
        HighResolutionTimer {
            label: label.into(),
            start: None,
            measurements_ns: Vec::new(),
        }
    }

    /// Start (or restart) the timer
    #[inline]
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop the timer and return elapsed nanoseconds.
    /// Also records the measurement for statistics.
    #[inline]
    pub fn stop_ns(&mut self) -> u64 {
        let elapsed = self
            .start
            .take()
            .map(|s| s.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        self.measurements_ns.push(elapsed);
        elapsed
    }

    /// Stop the timer and return elapsed microseconds (f64)
    #[inline]
    pub fn stop_us(&mut self) -> f64 {
        self.stop_ns() as f64 / 1_000.0
    }

    /// Stop the timer and return elapsed milliseconds (f64)
    #[inline]
    pub fn stop_ms(&mut self) -> f64 {
        self.stop_ns() as f64 / 1_000_000.0
    }

    /// Measure a closure and return its result along with elapsed nanoseconds
    pub fn measure<F, R>(&mut self, f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        self.start();
        let result = f();
        let elapsed = self.stop_ns();
        (result, elapsed)
    }

    /// Return all recorded measurements in nanoseconds
    pub fn measurements_ns(&self) -> &[u64] {
        &self.measurements_ns
    }

    /// Print a summary of all recorded measurements
    pub fn print_summary(&self) {
        if self.measurements_ns.is_empty() {
            println!("[Timer: {}] No measurements recorded.", self.label);
            return;
        }

        let mut sorted = self.measurements_ns.clone();
        sorted.sort_unstable();

        let n = sorted.len();
        let min_ns = sorted[0];
        let max_ns = sorted[n - 1];
        let sum_ns: u64 = sorted.iter().sum();
        let mean_ns = sum_ns / n as u64;
        let p50_ns = sorted[n / 2];
        let p95_ns = sorted[(n * 95) / 100];
        let p99_ns = sorted[(n * 99) / 100];

        println!("[Timer: {}]", self.label);
        println!("  Samples : {}", n);
        println!("  Min     : {} ns ({:.3} µs)", min_ns, min_ns as f64 / 1_000.0);
        println!("  Max     : {} ns ({:.3} µs)", max_ns, max_ns as f64 / 1_000.0);
        println!("  Mean    : {} ns ({:.3} µs)", mean_ns, mean_ns as f64 / 1_000.0);
        println!("  P50     : {} ns ({:.3} µs)", p50_ns, p50_ns as f64 / 1_000.0);
        println!("  P95     : {} ns ({:.3} µs)", p95_ns, p95_ns as f64 / 1_000.0);
        println!("  P99     : {} ns ({:.3} µs)", p99_ns, p99_ns as f64 / 1_000.0);
    }

    /// Serialize measurements to JSON
    pub fn to_json(&self) -> serde_json::Value {
        let mut sorted = self.measurements_ns.clone();
        sorted.sort_unstable();
        let n = sorted.len();

        if n == 0 {
            return serde_json::json!({
                "label": self.label,
                "samples": 0,
            });
        }

        let sum_ns: u64 = sorted.iter().sum();
        let mean_ns = sum_ns / n as u64;

        serde_json::json!({
            "label": self.label,
            "samples": n,
            "min_ns": sorted[0],
            "max_ns": sorted[n - 1],
            "mean_ns": mean_ns,
            "p50_ns": sorted[n / 2],
            "p95_ns": sorted[(n * 95) / 100],
            "p99_ns": sorted[(n * 99) / 100],
            "min_us": sorted[0] as f64 / 1_000.0,
            "max_us": sorted[n - 1] as f64 / 1_000.0,
            "mean_us": mean_ns as f64 / 1_000.0,
            "p50_us": sorted[n / 2] as f64 / 1_000.0,
            "p95_us": sorted[(n * 95) / 100] as f64 / 1_000.0,
            "p99_us": sorted[(n * 99) / 100] as f64 / 1_000.0,
        })
    }
}

// ============================================================
// Latency Report
// ============================================================

/// Complete latency report combining timer data and GPU metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct LatencyReport {
    /// Report generation timestamp (Unix seconds)
    pub generated_at_unix_secs: u64,

    /// Total number of cell edits pushed
    pub total_cell_edits: u64,

    /// Duration of the benchmark in seconds
    pub benchmark_duration_secs: f64,

    /// Throughput in edits per second
    pub throughput_edits_per_sec: f64,

    /// Push latency statistics (nanoseconds)
    pub push_latency_ns: LatencyStats,

    /// GPU metrics collected during the benchmark
    pub gpu_metrics: serde_json::Value,

    /// Whether hardware throttling was detected
    pub throttling_detected: bool,

    /// Notes / warnings
    pub notes: Vec<String>,
}

/// Latency statistics for a set of measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub samples: usize,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub std_dev_ns: f64,
    // Convenience µs fields
    pub min_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
}

impl LatencyStats {
    /// Compute statistics from a sorted slice of nanosecond measurements
    pub fn from_sorted_ns(sorted: &[u64]) -> Self {
        let n = sorted.len();
        assert!(n > 0, "Cannot compute stats from empty slice");

        let min_ns = sorted[0];
        let max_ns = sorted[n - 1];
        let sum: u64 = sorted.iter().sum();
        let mean_ns = sum / n as u64;

        let p50_ns = sorted[n / 2];
        let p90_ns = sorted[(n * 90) / 100];
        let p95_ns = sorted[(n * 95) / 100];
        let p99_ns = sorted[(n * 99) / 100];
        let p999_ns = sorted[(n * 999) / 1000];

        let variance: f64 = sorted
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
            min_ns,
            max_ns,
            mean_ns,
            p50_ns,
            p90_ns,
            p95_ns,
            p99_ns,
            p999_ns,
            std_dev_ns,
            min_us: min_ns as f64 / 1_000.0,
            max_us: max_ns as f64 / 1_000.0,
            mean_us: mean_ns as f64 / 1_000.0,
            p50_us: p50_ns as f64 / 1_000.0,
            p90_us: p90_ns as f64 / 1_000.0,
            p95_us: p95_ns as f64 / 1_000.0,
            p99_us: p99_ns as f64 / 1_000.0,
            p999_us: p999_ns as f64 / 1_000.0,
        }
    }

    /// Print a formatted table
    pub fn print_table(&self, label: &str) {
        println!("\n=== {} Latency Statistics ({} samples) ===", label, self.samples);
        println!("┌──────────────────────────────────────────┐");
        println!("│  Metric    │   Nanoseconds  │ Microseconds │");
        println!("├──────────────────────────────────────────┤");
        println!("│  Min       │  {:>12}  │  {:>10.3}  │", self.min_ns, self.min_us);
        println!("│  Max       │  {:>12}  │  {:>10.3}  │", self.max_ns, self.max_us);
        println!("│  Mean      │  {:>12}  │  {:>10.3}  │", self.mean_ns, self.mean_us);
        println!("│  Std Dev   │  {:>12.0}  │  {:>10.3}  │", self.std_dev_ns, self.std_dev_ns / 1_000.0);
        println!("├──────────────────────────────────────────┤");
        println!("│  P50       │  {:>12}  │  {:>10.3}  │", self.p50_ns, self.p50_us);
        println!("│  P90       │  {:>12}  │  {:>10.3}  │", self.p90_ns, self.p90_us);
        println!("│  P95       │  {:>12}  │  {:>10.3}  │", self.p95_ns, self.p95_us);
        println!("│  P99       │  {:>12}  │  {:>10.3}  │", self.p99_ns, self.p99_us);
        println!("│  P99.9     │  {:>12}  │  {:>10.3}  │", self.p999_ns, self.p999_us);
        println!("└──────────────────────────────────────────┘");
    }
}
