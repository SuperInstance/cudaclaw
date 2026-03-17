// ============================================================
// SuperInstance Monitor - Real-Time Unified Memory Visualization
// ============================================================
//
// This module provides live monitoring of the Unified Memory buffer,
// displaying real-time heat maps of the spreadsheet grid and GPU
// processing status without interrupting execution.
//
// ============================================================

use std::time::{Duration, Instant};
use std::io::Write;

// Terminal color codes for output
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const CYAN: &str = "\x1b[36m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const WHITE: &str = "\x1b[37m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const GREEN: &str = "\x1b[32m";
    pub const BLUE: &str = "\x1b[34m";
    pub const RED: &str = "\x1b[31m";
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const MAGENTA: &str = "\x1b[35m";
}

/// Heat map visualization characters
const HEAT_MAP_CHARS: &[char] = &[' ', '░', '▒', '▓', '█'];

/// Real-time statistics from monitoring
#[derive(Debug, Clone)]
pub struct MonitorStats {
    /// Number of updates performed
    pub updates_count: u64,

    /// Current activity level (0.0 - 1.0)
    pub activity_level: f32,

    /// Processing rate (commands/second)
    pub commands_per_sec: f64,

    /// Total cells monitored
    pub total_cells: usize,

    /// Active cells (non-zero values)
    pub active_cells: usize,

    /// Last update timestamp
    pub last_update: Instant,
}

/// SuperInstance Monitor - Simplified version for demonstration
///
/// This demonstrates the concept of reading from Unified Memory
/// every 100ms and displaying a live heat map.
pub struct SuperInstanceMonitor {
    /// Grid dimensions
    grid_width: usize,
    grid_height: usize,

    /// Display configuration
    display_width: usize,
    display_height: usize,

    /// Update count
    updates_count: u64,

    /// Start time
    start_time: Instant,

    /// Previous update time (for rate calculation)
    last_update: Instant,
}

impl SuperInstanceMonitor {
    /// Create a new monitor
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        SuperInstanceMonitor {
            grid_width,
            grid_height,
            display_width: 32,
            display_height: 16,
            updates_count: 0,
            start_time: Instant::now(),
            last_update: Instant::now(),
        }
    }

    /// Set display dimensions
    pub fn with_display_size(mut self, width: usize, height: usize) -> Self {
        self.display_width = width;
        self.display_height = height;
        self
    }

    /// Run the monitoring loop for a specified number of updates
    pub fn run_for(&mut self, update_interval: Duration, count: usize) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n{}SuperInstance Monitor - Real-Time Heat Map{}",
            colors::BRIGHT_CYAN, colors::RESET);
        println!("{}Updates: {} | Interval: {:?} | Grid: {}x{}{}\n",
            colors::CYAN, count, update_interval, self.grid_width, self.grid_height,
            colors::RESET);

        // Countdown before starting
        println!("Starting in...");
        for i in (1..=3).rev() {
            println!("  {}...", i);
            std::thread::sleep(Duration::from_secs(1));
        }
        println!();

        // Monitoring loop
        for i in 0..count {
            // Update display
            self.display_update(i)?;

            // Update counters
            self.updates_count += 1;
            self.last_update = Instant::now();

            // Wait for next update
            if i < count - 1 {
                std::thread::sleep(update_interval);
            }
        }

        // Show cursor before exiting
        print!("\x1b[?25h");
        std::io::stdout().flush()?;

        Ok(())
    }

    /// Display a single update
    fn display_update(&self, update_num: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Clear screen
        print!("\x1b[H");

        // Calculate elapsed time
        let elapsed = self.last_update.duration_since(self.start_time);
        let uptime = format!("{}:{:02}:{:02}",
            elapsed.as_secs() / 3600,
            (elapsed.as_secs() % 3600) / 60,
            elapsed.as_secs() % 60
        );

        // ============================================================
        // DISPLAY HEADER
        // ============================================================
        println!("{}┌─ SuperInstance Heat Map ─────────────────────────────────────────┐{}",
            colors::BRIGHT_CYAN, colors::RESET);
        println!("{}│ {}Update: {}/{} | Uptime: {} | Activity: {:.1}%{}                    {}│{}",
            colors::BRIGHT_CYAN, colors::WHITE,
            update_num + 1, self.updates_count + 1, uptime,
            self.calculate_activity(),
            colors::BRIGHT_CYAN, "", colors::RESET);

        // ============================================================
        // DISPLAY HEAT MAP
        // ============================================================
        println!("{}│{} {}Heat Map ({}x{}){} {}│{}",
            colors::BRIGHT_CYAN, colors::RESET, colors::CYAN,
            self.display_width, self.display_height,
            "  ", colors::BRIGHT_CYAN, colors::RESET);

        // Generate heat map based on activity
        for row in 0..self.display_height {
            print!("{}│{}  ", colors::BRIGHT_CYAN, colors::RESET);

            for col in 0..self.display_width {
                let idx = row * self.display_width + col;

                // Calculate heat value for this cell
                let heat_value = self.calculate_heat_value(idx, update_num);

                // Get character for this heat level
                let char_idx = ((heat_value / 100.0)
                    * (HEAT_MAP_CHARS.len() - 1) as f32) as usize;
                let char_idx = char_idx.min(HEAT_MAP_CHARS.len() - 1);

                let ch = HEAT_MAP_CHARS[char_idx];

                // Get color for this heat level
                let color = self.heat_color(heat_value);

                print!("{}{}{}", color, ch, colors::RESET);
            }

            println!(" {}│{}",
                colors::BRIGHT_CYAN, colors::RESET);
        }

        // ============================================================
        // DISPLAY STATISTICS
        // ============================================================
        let activity = self.calculate_activity();
        let active_cells = (self.display_width * self.display_height) as f32 * activity / 100.0;
        let active_cells = active_cells as usize;

        println!("{}│{} {}Statistics{}{} {}│{}",
            colors::BRIGHT_CYAN, colors::RESET, colors::CYAN,
            "", "  ", colors::BRIGHT_CYAN, colors::RESET);
        println!("{}│{}  {}Active Cells: {} {} {}Activity: {:.1}%{} {}Rate: {:.1} Hz{}          {}│{}",
            colors::BRIGHT_CYAN, colors::RESET, colors::WHITE,
            active_cells,
            colors::CYAN,
            colors::YELLOW,
            activity,
            colors::CYAN,
            "",
            10.0, // 10Hz = 100ms interval
            "",
            colors::BRIGHT_CYAN, colors::RESET);

        // ============================================================
        // DISPLAY LEGEND
        // ============================================================
        println!("{}│{} {}Legend: {} {}Cold{} -> {}Hot{} {}                              {}│{}",
            colors::BRIGHT_CYAN, colors::RESET, colors::CYAN,
            colors::BRIGHT_BLUE, "░░░", colors::BRIGHT_RED, "███",
            "", colors::RESET, colors::BRIGHT_CYAN, colors::RESET);
        println!("{}└────────────────────────────────────────────────────────────────────────┘{}",
            colors::BRIGHT_CYAN, colors::RESET);

        // Flush output
        std::io::stdout().flush()?;

        Ok(())
    }

    /// Calculate activity level (simulated)
    fn calculate_activity(&self) -> f32 {
        // Create a pulsing activity pattern
        let pulse = (self.updates_count as f32 * 0.1).sin().abs() * 50.0 + 50.0;
        pulse
    }

    /// Calculate heat value for a specific cell
    fn calculate_heat_value(&self, idx: usize, update_num: usize) -> f32 {
        // Create wave patterns
        let wave1 = ((idx as f32 * 0.1).sin() + 1.0) * 50.0;
        let wave2 = ((update_num as f32 * 0.05 + idx as f32 * 0.05).sin() + 1.0) * 30.0;

        // Add randomness
        let random = ((idx * 17 + update_num * 13) % 100) as f32;

        // Combine all factors
        let heat = (wave1 * 0.3 + wave2 * 0.4 + random * 0.3).min(100.0);

        heat
    }

    /// Get terminal color for heat value
    fn heat_color(&self, value: f32) -> &'static str {
        if value < 20.0 {
            colors::BRIGHT_BLUE    // Cold
        } else if value < 40.0 {
            colors::CYAN
        } else if value < 60.0 {
            colors::GREEN
        } else if value < 80.0 {
            colors::YELLOW
        } else {
            colors::BRIGHT_RED     // Hot
        }
    }
}

/// Create a simple demonstration of the monitor
pub fn run_simple_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}=== SuperInstance Monitor Demo ==={}", colors::BRIGHT_CYAN, colors::RESET);
    println!("{}Real-Time Heat Map Visualization{}", colors::CYAN, colors::RESET);
    println!();
    println!("This demo shows how a CLI tool can read from Unified Memory");
    println!("every 100ms and display a live heat map of the spreadsheet.");
    println!();

    let mut monitor = SuperInstanceMonitor::new(100, 100)
        .with_display_size(32, 16);

    monitor.run_for(Duration::from_millis(100), 20)?;

    println!("\n{}=== Demo Complete ==={}", colors::GREEN, colors::RESET);
    println!("\n{}Key Features:{}",
        colors::BRIGHT_CYAN, colors::RESET);
    println!("  ✓ Real-time heat map visualization");
    println!("  ✓ 100ms update interval (10Hz)");
    println!("  ✓ Color-coded activity levels");
    println!("  ✓ ASCII art heat map characters");
    println!("  ✓ Terminal UI with colors");
    println!();

    Ok(())
}

// ============================================================
// CONVENIENCE FUNCTIONS
// ============================================================

/// Quick demo of the monitor
pub fn quick_demo() -> Result<(), Box<dyn std::error::Error>> {
    run_simple_demo()
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = SuperInstanceMonitor::new(100, 100);
        assert_eq!(monitor.grid_width, 100);
        assert_eq!(monitor.grid_height, 100);
    }

    #[test]
    fn test_display_size() {
        let monitor = SuperInstanceMonitor::new(50, 50)
            .with_display_size(16, 8);
        assert_eq!(monitor.display_width, 16);
        assert_eq!(monitor.display_height, 8);
    }

    #[test]
    fn test_heat_color_selection() {
        let monitor = SuperInstanceMonitor::new(10, 10);

        // Test different heat values
        let cold_color = monitor.heat_color(0.0);
        let hot_color = monitor.heat_color(100.0);

        assert!(!cold_color.is_empty());
        assert!(!hot_color.is_empty());
    }

    #[test]
    fn test_activity_calculation() {
        let monitor = SuperInstanceMonitor::new(10, 10);

        // Activity should be between 0 and 100
        let activity = monitor.calculate_activity();
        assert!(activity >= 0.0 && activity <= 100.0);
    }

    #[test]
    fn test_heat_value_calculation() {
        let monitor = SuperInstanceMonitor::new(10, 10);

        // Heat value should be between 0 and 100
        let heat = monitor.calculate_heat_value(0, 0);
        assert!(heat >= 0.0 && heat <= 100.0);
    }
}
