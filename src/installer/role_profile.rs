// ============================================================
// Ramified Role Profile System
// ============================================================
//
// Persists optimized configurations as reusable "Ramified Role"
// profiles in `.cudaclaw/roles/`. Each profile is hardware-locked
// (tied to a specific GPU + architecture) and contains:
//
//   1. The winning OptimizationSuggestion
//   2. Hardware fingerprint (GPU name + compute capability)
//   3. Role context metadata
//   4. Simulation results proving the configuration's merit
//   5. Provenance (which LLM, which round, timestamp)
//
// DIRECTORY STRUCTURE:
//   .cudaclaw/
//     roles/
//       spreadsheet_engine/
//         rtx4090_sm89.json          ← hardware-locked profile
//         rtx3080_sm86.json
//       realtime_crdt/
//         a100_sm80.json
//       batch_processor/
//         h100_sm90.json
//
// Profiles can be loaded at runtime to configure kernel launch
// parameters without re-running the optimization loop.
//
// ============================================================

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::installer::hardware_probe::HardwareProfile;
use crate::installer::llm_optimizer::{
    LlmConfig, OptimizationSuggestion, RoleContext, SimulationResult,
};
use crate::installer::simulated_finetuning::SimulationReport;

// ============================================================
// Role Profile
// ============================================================

/// A complete, persisted optimization profile.
/// This is the final artifact of the intelligent installer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleProfile {
    /// Profile format version (for forward compatibility)
    pub version: u32,

    /// Role name (e.g., "spreadsheet_engine")
    pub role_name: String,

    /// Hardware fingerprint this profile is optimized for
    pub hardware_fingerprint: HardwareFingerprint,

    /// The winning optimization configuration
    pub configuration: OptimizationSuggestion,

    /// Simulation results that validated this configuration
    pub simulation_results: Vec<SimulationResult>,

    /// Best score achieved
    pub best_score: f64,

    /// Whether the P99 target (8µs) was met
    pub meets_p99_target: bool,

    /// P99 RTT achieved (microseconds)
    pub achieved_p99_rtt_us: f64,

    /// Achieved throughput (commands/sec)
    pub achieved_throughput_cmds_per_sec: f64,

    /// Provenance: how this profile was created
    pub provenance: ProfileProvenance,

    /// Role context used during optimization
    pub role_context: RoleContext,

    /// Creation timestamp (Unix seconds)
    pub created_at: u64,

    /// Last validation timestamp (Unix seconds)
    pub last_validated: u64,

    /// Number of times this profile has been loaded
    pub load_count: u64,
}

// ============================================================
// Hardware Fingerprint
// ============================================================

/// Uniquely identifies a GPU hardware configuration.
/// Used to match profiles to the current hardware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareFingerprint {
    /// GPU name (e.g., "NVIDIA RTX 4090")
    pub gpu_name: String,

    /// Compute capability (e.g., "8.9")
    pub compute_capability: String,

    /// SM count (uniquely identifies GPU SKU within arch)
    pub sm_count: u32,

    /// Architecture family (e.g., "Ada Lovelace")
    pub architecture: String,

    /// Whether the hardware was simulated during profiling
    pub was_simulated: bool,

    /// Short identifier for file naming (e.g., "rtx4090_sm89")
    pub short_id: String,
}

impl HardwareFingerprint {
    /// Create a fingerprint from a hardware profile.
    pub fn from_profile(hw: &HardwareProfile) -> Self {
        let short_id = Self::make_short_id(&hw.gpu_name, &hw.compute_capability);
        HardwareFingerprint {
            gpu_name: hw.gpu_name.clone(),
            compute_capability: hw.compute_capability.clone(),
            sm_count: hw.sm_count,
            architecture: hw.architecture.clone(),
            was_simulated: hw.is_simulated,
            short_id,
        }
    }

    /// Generate a filesystem-safe short identifier.
    fn make_short_id(gpu_name: &str, compute_cap: &str) -> String {
        // Extract key parts: "NVIDIA RTX 4090 (Simulated)" → "rtx4090"
        let name_part: String = gpu_name
            .to_lowercase()
            .replace("nvidia", "")
            .replace("(simulated)", "")
            .replace("geforce", "")
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ')
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join("");

        // Compute cap: "8.9" → "sm89"
        let sm_part = format!("sm{}", compute_cap.replace('.', ""));

        format!("{}_{}", name_part, sm_part)
    }

    /// Check if this fingerprint matches another hardware profile.
    pub fn matches(&self, hw: &HardwareProfile) -> bool {
        self.compute_capability == hw.compute_capability
            && self.sm_count == hw.sm_count
    }
}

// ============================================================
// Profile Provenance
// ============================================================

/// Records how a profile was created for auditability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProvenance {
    /// LLM provider used (or "heuristic" if no LLM)
    pub llm_provider: String,

    /// LLM model used (or "none")
    pub llm_model: String,

    /// Number of optimization rounds completed
    pub optimization_rounds: u32,

    /// Total number of configurations evaluated
    pub configurations_evaluated: u32,

    /// How the profile was created
    pub method: OptimizationMethod,

    /// CudaClaw version
    pub cudaclaw_version: String,
}

/// How the optimization was performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    /// Full LLM optimization loop with simulated fine-tuning
    LlmOptimized,
    /// Heuristic baseline (no LLM, hardware-aware defaults)
    HeuristicBaseline,
    /// Manual configuration by user
    ManualConfig,
}

// ============================================================
// Profile Manager
// ============================================================

/// Manages reading, writing, and discovering role profiles
/// in the `.cudaclaw/roles/` directory.
pub struct ProfileManager {
    /// Root directory for role profiles
    roles_dir: PathBuf,
}

impl ProfileManager {
    /// Create a new ProfileManager rooted at the given project directory.
    /// Profiles are stored in `{project_root}/.cudaclaw/roles/`.
    pub fn new(project_root: &Path) -> Self {
        ProfileManager {
            roles_dir: project_root.join(".cudaclaw").join("roles"),
        }
    }

    /// Create a ProfileManager with a custom roles directory.
    pub fn with_dir(roles_dir: PathBuf) -> Self {
        ProfileManager { roles_dir }
    }

    /// Get the path to the roles directory.
    pub fn roles_dir(&self) -> &Path {
        &self.roles_dir
    }

    /// Save a role profile to disk.
    /// Creates the directory structure if it doesn't exist.
    pub fn save_profile(&self, profile: &RoleProfile) -> Result<PathBuf, ProfileError> {
        let role_dir = self.roles_dir.join(&profile.role_name);
        std::fs::create_dir_all(&role_dir).map_err(|e| ProfileError::IoError {
            path: role_dir.display().to_string(),
            message: format!("Failed to create role directory: {}", e),
        })?;

        let filename = format!("{}.json", profile.hardware_fingerprint.short_id);
        let filepath = role_dir.join(&filename);

        let json = serde_json::to_string_pretty(profile).map_err(|e| {
            ProfileError::SerializationError(format!("Failed to serialize profile: {}", e))
        })?;

        std::fs::write(&filepath, json).map_err(|e| ProfileError::IoError {
            path: filepath.display().to_string(),
            message: format!("Failed to write profile: {}", e),
        })?;

        println!("  Profile saved: {}", filepath.display());
        Ok(filepath)
    }

    /// Load a specific role profile by role name and hardware fingerprint.
    pub fn load_profile(
        &self,
        role_name: &str,
        hardware: &HardwareProfile,
    ) -> Result<RoleProfile, ProfileError> {
        let fingerprint = HardwareFingerprint::from_profile(hardware);
        let filename = format!("{}.json", fingerprint.short_id);
        let filepath = self.roles_dir.join(role_name).join(&filename);

        if !filepath.exists() {
            return Err(ProfileError::NotFound {
                role: role_name.to_string(),
                hardware: fingerprint.short_id,
            });
        }

        let json = std::fs::read_to_string(&filepath).map_err(|e| ProfileError::IoError {
            path: filepath.display().to_string(),
            message: format!("Failed to read profile: {}", e),
        })?;

        let mut profile: RoleProfile = serde_json::from_str(&json).map_err(|e| {
            ProfileError::SerializationError(format!("Failed to deserialize profile: {}", e))
        })?;

        profile.load_count += 1;

        // Write back the incremented load count
        if let Ok(updated_json) = serde_json::to_string_pretty(&profile) {
            let _ = std::fs::write(&filepath, updated_json);
        }

        Ok(profile)
    }

    /// List all available profiles for a role.
    pub fn list_profiles(&self, role_name: &str) -> Result<Vec<String>, ProfileError> {
        let role_dir = self.roles_dir.join(role_name);
        if !role_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = std::fs::read_dir(&role_dir).map_err(|e| ProfileError::IoError {
            path: role_dir.display().to_string(),
            message: format!("Failed to read role directory: {}", e),
        })?;

        let profiles: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
            })
            .filter_map(|e| {
                e.path()
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
            })
            .collect();

        Ok(profiles)
    }

    /// List all available roles.
    pub fn list_roles(&self) -> Result<Vec<String>, ProfileError> {
        if !self.roles_dir.exists() {
            return Ok(Vec::new());
        }

        let entries = std::fs::read_dir(&self.roles_dir).map_err(|e| ProfileError::IoError {
            path: self.roles_dir.display().to_string(),
            message: format!("Failed to read roles directory: {}", e),
        })?;

        let roles: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                e.path()
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
            })
            .collect();

        Ok(roles)
    }

    /// Delete a specific profile.
    pub fn delete_profile(
        &self,
        role_name: &str,
        hardware_id: &str,
    ) -> Result<(), ProfileError> {
        let filename = format!("{}.json", hardware_id);
        let filepath = self.roles_dir.join(role_name).join(&filename);

        if !filepath.exists() {
            return Err(ProfileError::NotFound {
                role: role_name.to_string(),
                hardware: hardware_id.to_string(),
            });
        }

        std::fs::remove_file(&filepath).map_err(|e| ProfileError::IoError {
            path: filepath.display().to_string(),
            message: format!("Failed to delete profile: {}", e),
        })?;

        Ok(())
    }

    /// Find the best matching profile for the current hardware.
    /// Returns None if no matching profile exists.
    pub fn find_matching_profile(
        &self,
        role_name: &str,
        hardware: &HardwareProfile,
    ) -> Option<RoleProfile> {
        self.load_profile(role_name, hardware).ok()
    }

    /// Create a RoleProfile from optimization results.
    pub fn create_profile(
        hardware: &HardwareProfile,
        role: &RoleContext,
        best_suggestion: &OptimizationSuggestion,
        best_result: &SimulationResult,
        all_results: Vec<SimulationResult>,
        llm_config: Option<&LlmConfig>,
    ) -> RoleProfile {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let provenance = if let Some(config) = llm_config {
            ProfileProvenance {
                llm_provider: format!("{:?}", config.provider),
                llm_model: config.model.clone(),
                optimization_rounds: config.optimization_rounds,
                configurations_evaluated: all_results.len() as u32,
                method: OptimizationMethod::LlmOptimized,
                cudaclaw_version: env!("CARGO_PKG_VERSION").to_string(),
            }
        } else {
            ProfileProvenance {
                llm_provider: "none".to_string(),
                llm_model: "none".to_string(),
                optimization_rounds: 0,
                configurations_evaluated: all_results.len() as u32,
                method: OptimizationMethod::HeuristicBaseline,
                cudaclaw_version: env!("CARGO_PKG_VERSION").to_string(),
            }
        };

        RoleProfile {
            version: 1,
            role_name: role.role_name.clone(),
            hardware_fingerprint: HardwareFingerprint::from_profile(hardware),
            configuration: best_suggestion.clone(),
            simulation_results: all_results,
            best_score: best_result.overall_score,
            meets_p99_target: best_result.meets_p99_target,
            achieved_p99_rtt_us: best_result.p99_rtt_us,
            achieved_throughput_cmds_per_sec: best_result.throughput_cmds_per_sec,
            provenance,
            role_context: role.clone(),
            created_at: timestamp,
            last_validated: timestamp,
            load_count: 0,
        }
    }

    /// Print a summary of a role profile.
    pub fn print_profile_summary(profile: &RoleProfile) {
        println!("\n{}", "═".repeat(64));
        println!("  Ramified Role Profile: {}", profile.role_name);
        println!("{}", "═".repeat(64));
        println!("  Hardware   : {} ({})",
            profile.hardware_fingerprint.gpu_name,
            profile.hardware_fingerprint.compute_capability);
        println!("  Architecture: {}", profile.hardware_fingerprint.architecture);
        println!("  Simulated  : {}", profile.hardware_fingerprint.was_simulated);
        println!("  Score      : {:.1}", profile.best_score);
        println!("  P99 RTT    : {:.2} µs (target: 8.0 µs) — {}",
            profile.achieved_p99_rtt_us,
            if profile.meets_p99_target { "MET" } else { "NOT MET" });
        println!("  Throughput : {:.0} cmd/s", profile.achieved_throughput_cmds_per_sec);

        println!("\n  --- Configuration ---");
        let c = &profile.configuration;
        println!("  Block Size       : {}", c.block_size);
        println!("  Grid Size        : {}", c.grid_size);
        println!("  Shared Memory    : {} KB", c.shared_memory_bytes / 1024);
        println!("  Loop Unroll      : {}x", c.loop_unroll_factor);
        println!("  Batch Size       : {}", c.command_batch_size);
        println!("  Idle Sleep       : {} ns", c.idle_sleep_ns);
        println!("  Warps/Block      : {}", c.warps_per_block);
        println!("  L1 Cache Pref    : {}", c.l1_cache_preference);
        println!("  Warp Aggregation : {}", c.enable_warp_aggregation);
        println!("  SoA Layout       : {}", c.use_soa_layout);
        println!("  Frontier Batch   : {}", c.frontier_batch_size);
        println!("  CAS Backoff      : {} - {} ns", c.cas_backoff_initial_ns, c.cas_backoff_max_ns);

        println!("\n  --- Provenance ---");
        println!("  Method     : {:?}", profile.provenance.method);
        println!("  LLM        : {} ({})",
            profile.provenance.llm_provider, profile.provenance.llm_model);
        println!("  Rounds     : {}", profile.provenance.optimization_rounds);
        println!("  Configs    : {} evaluated", profile.provenance.configurations_evaluated);
        println!("  Version    : {}", profile.provenance.cudaclaw_version);

        println!("\n  Reasoning: {}", c.reasoning);
        println!("{}\n", "═".repeat(64));
    }
}

// ============================================================
// Error Types
// ============================================================

#[derive(Debug)]
pub enum ProfileError {
    IoError { path: String, message: String },
    SerializationError(String),
    NotFound { role: String, hardware: String },
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::IoError { path, message } =>
                write!(f, "I/O error at {}: {}", path, message),
            ProfileError::SerializationError(msg) =>
                write!(f, "Serialization error: {}", msg),
            ProfileError::NotFound { role, hardware } =>
                write!(f, "No profile found for role '{}' on hardware '{}'", role, hardware),
        }
    }
}

impl std::error::Error for ProfileError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::installer::hardware_probe::HardwareProber;
    use crate::installer::llm_optimizer::LlmOptimizer;
    use tempfile::TempDir;

    fn setup() -> (HardwareProfile, RoleContext) {
        let prober = HardwareProber::simulated(0);
        let hw = prober.probe();
        let role = RoleContext::spreadsheet_engine();
        (hw, role)
    }

    #[test]
    fn test_hardware_fingerprint() {
        let (hw, _) = setup();
        let fp = HardwareFingerprint::from_profile(&hw);

        assert!(!fp.short_id.is_empty());
        assert!(fp.short_id.contains("sm89"), "Short ID should contain sm89, got: {}", fp.short_id);
        assert!(fp.was_simulated);
        assert!(fp.matches(&hw));
    }

    #[test]
    fn test_save_and_load_profile() {
        let tmp = TempDir::new().unwrap();
        let manager = ProfileManager::new(tmp.path());

        let (hw, role) = setup();
        let suggestion = LlmOptimizer::generate_heuristic_suggestion(&hw, &role);

        let result = crate::installer::llm_optimizer::SimulationResult {
            suggestion_id: suggestion.suggestion_id.clone(),
            round: 0,
            p99_rtt_us: 5.0,
            throughput_cmds_per_sec: 1_000_000.0,
            power_efficiency_score: 75.0,
            overall_score: 80.0,
            meets_p99_target: true,
            notes: "Test result".to_string(),
        };

        let profile = ProfileManager::create_profile(
            &hw, &role, &suggestion, &result, vec![result.clone()], None);

        // Save
        let path = manager.save_profile(&profile).unwrap();
        assert!(path.exists());

        // Load
        let loaded = manager.load_profile("spreadsheet_engine", &hw).unwrap();
        assert_eq!(loaded.role_name, "spreadsheet_engine");
        assert_eq!(loaded.load_count, 1);
        assert!(loaded.meets_p99_target);
    }

    #[test]
    fn test_list_roles_and_profiles() {
        let tmp = TempDir::new().unwrap();
        let manager = ProfileManager::new(tmp.path());

        let (hw, role) = setup();
        let suggestion = OptimizationSuggestion::default();
        let result = crate::installer::llm_optimizer::SimulationResult {
            suggestion_id: "test".to_string(),
            round: 0,
            p99_rtt_us: 5.0,
            throughput_cmds_per_sec: 1_000_000.0,
            power_efficiency_score: 75.0,
            overall_score: 80.0,
            meets_p99_target: true,
            notes: "Test".to_string(),
        };

        let profile = ProfileManager::create_profile(
            &hw, &role, &suggestion, &result, vec![result], None);
        manager.save_profile(&profile).unwrap();

        let roles = manager.list_roles().unwrap();
        assert!(roles.contains(&"spreadsheet_engine".to_string()));

        let profiles = manager.list_profiles("spreadsheet_engine").unwrap();
        assert!(!profiles.is_empty());
    }

    #[test]
    fn test_profile_not_found() {
        let tmp = TempDir::new().unwrap();
        let manager = ProfileManager::new(tmp.path());
        let (hw, _) = setup();

        let result = manager.load_profile("nonexistent", &hw);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_profile() {
        let tmp = TempDir::new().unwrap();
        let manager = ProfileManager::new(tmp.path());

        let (hw, role) = setup();
        let suggestion = OptimizationSuggestion::default();
        let result = crate::installer::llm_optimizer::SimulationResult {
            suggestion_id: "test".to_string(),
            round: 0,
            p99_rtt_us: 5.0,
            throughput_cmds_per_sec: 1_000_000.0,
            power_efficiency_score: 75.0,
            overall_score: 80.0,
            meets_p99_target: true,
            notes: "Test".to_string(),
        };

        let profile = ProfileManager::create_profile(
            &hw, &role, &suggestion, &result, vec![result], None);
        manager.save_profile(&profile).unwrap();

        let fp = HardwareFingerprint::from_profile(&hw);
        manager.delete_profile("spreadsheet_engine", &fp.short_id).unwrap();

        let profiles = manager.list_profiles("spreadsheet_engine").unwrap();
        assert!(profiles.is_empty());
    }
}
