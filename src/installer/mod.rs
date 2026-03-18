// ============================================================
// CudaClaw Intelligent Installer
// ============================================================
//
// Orchestrates the full optimization pipeline:
//
//   1. Hardware Probe → detect GPU attributes
//   2. LLM Optimization → get tuning suggestions
//   3. Simulated Fine-Tuning → evaluate suggestions
//   4. Role Profile → persist the best configuration
//
// USAGE (CLI):
//   # Full optimization with LLM (OpenAI):
//   cargo run -- install --role spreadsheet_engine \
//       --llm-provider openai --api-key $OPENAI_API_KEY --model gpt-4
//
//   # Full optimization with Deepseek:
//   cargo run -- install --role spreadsheet_engine \
//       --llm-provider deepseek --api-key $DEEPSEEK_API_KEY
//
//   # Full optimization with Claude:
//   cargo run -- install --role spreadsheet_engine \
//       --llm-provider anthropic --api-key $ANTHROPIC_API_KEY
//
//   # Custom OpenAI-compatible endpoint (Ollama, vLLM):
//   cargo run -- install --role spreadsheet_engine \
//       --base-url http://localhost:11434/v1 --model llama3
//
//   # Heuristic-only (no LLM, hardware-aware defaults):
//   cargo run -- install --role spreadsheet_engine --heuristic-only
//
//   # View saved profiles:
//   cargo run -- install --list-profiles
//
//   # Load and display a profile:
//   cargo run -- install --role spreadsheet_engine --show-profile
//
// ============================================================

pub mod hardware_probe;
pub mod llm_optimizer;
pub mod micro_simulation;
pub mod nvrtc_muscle_compiler;
pub mod role_profile;
pub mod simulated_finetuning;

use hardware_probe::{HardwareProfile, HardwareProber};
use llm_optimizer::{LlmConfig, LlmOptimizer, LlmProvider, OptimizationSuggestion, RoleContext};
use micro_simulation::{MicroSimConfig, MicroSimEngine, MicroSimResult};
use nvrtc_muscle_compiler::MuscleFiberCompiler;
use role_profile::{ProfileManager, RoleProfile};
use simulated_finetuning::{SimulationEngine, SimulationReport};

use crate::dna::RamifiedRole;

// ============================================================
// Installer Configuration
// ============================================================

/// Configuration for the intelligent installer.
pub struct InstallerConfig {
    /// Role to optimize for
    pub role: RoleContext,

    /// LLM configuration (None = heuristic-only mode)
    pub llm_config: Option<LlmConfig>,

    /// GPU index to probe
    pub gpu_index: u32,

    /// Project root directory (for .cudaclaw/roles/)
    pub project_root: std::path::PathBuf,

    /// Whether to skip the hardware probe (use simulated data)
    pub skip_probe: bool,

    /// Number of additional heuristic variations to test
    pub heuristic_variations: u32,

    /// Simulation noise factor (0.0 = deterministic)
    pub simulation_noise: f64,
}

impl Default for InstallerConfig {
    fn default() -> Self {
        InstallerConfig {
            role: RoleContext::spreadsheet_engine(),
            llm_config: None,
            gpu_index: 0,
            project_root: std::path::PathBuf::from("."),
            skip_probe: false,
            heuristic_variations: 5,
            simulation_noise: 0.02,
        }
    }
}

// ============================================================
// Installer
// ============================================================

/// The intelligent installer orchestrator.
pub struct Installer {
    config: InstallerConfig,
}

impl Installer {
    pub fn new(config: InstallerConfig) -> Self {
        Installer { config }
    }

    /// Run the full installation pipeline.
    /// Returns the saved RoleProfile path on success.
    pub async fn run(&self) -> Result<(RoleProfile, std::path::PathBuf), InstallerError> {
        println!("\n{}", "╔══════════════════════════════════════════════════════════════╗");
        println!("║            CudaClaw Intelligent Installer v{}          ║",
            env!("CARGO_PKG_VERSION"));
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // ── Step 1: Hardware Probe ──────────────────────────
        println!("Step 1/6: Hardware Probe");
        let hardware = self.run_probe();
        HardwareProber::print_profile(&hardware);

        // ── Step 2: Generate Optimization Suggestions ───────
        println!("\nStep 2/6: Generating Optimization Suggestions");
        let suggestions = self.generate_suggestions(&hardware).await?;
        println!("  Generated {} suggestions", suggestions.len());

        // ── Step 3: Simulated Fine-Tuning ───────────────────
        println!("\nStep 3/6: Simulated Fine-Tuning (model-driven)");
        let engine = SimulationEngine::new(
            hardware.clone(),
            self.config.role.clone(),
        ).with_noise(self.config.simulation_noise);

        let results = engine.evaluate_all(&suggestions);
        SimulationEngine::print_comparison(&results);

        let (best_idx, best_result) = engine.find_best(&suggestions)
            .ok_or(InstallerError::NoSuggestions)?;

        let best_suggestion = &suggestions[best_idx];

        // ── Step 3.5: NVRTC Muscle Fiber Compilation ──────────
        println!("\nStep 3.5/6: NVRTC Muscle Fiber Compilation");
        println!("  Compiling LLM-suggested constants into PTX via NVRTC...");
        let mut muscle_compiler = MuscleFiberCompiler::new(hardware.clone());
        if muscle_compiler.is_nvrtc_available() {
            println!("  NVRTC available — real compilation");
        } else {
            println!("  NVRTC not available — simulated compilation");
        }
        let muscle_compilations = muscle_compiler.compile_all(&suggestions);
        MuscleFiberCompiler::print_compilation_summary(&muscle_compilations);
        println!(
            "  Compiled {} / {} suggestions into PTX muscle fibers",
            muscle_compilations.len(),
            suggestions.len()
        );

        // ── Step 4: Micro-Simulation (5s kernel benchmarks) ─
        println!("\nStep 4/6: Micro-Simulation (5-second kernel benchmarks)");
        let micro_engine = MicroSimEngine::new(hardware.clone());

        // Build candidate configs from top suggestions (up to 3)
        let top_n = results.len().min(3);
        let top_suggestion_ids: Vec<String> = results[..top_n]
            .iter()
            .map(|r| r.suggestion_id.clone())
            .collect();

        let micro_candidates: Vec<MicroSimConfig> = suggestions
            .iter()
            .filter(|s| top_suggestion_ids.contains(&s.suggestion_id))
            .map(MicroSimConfig::from_suggestion)
            .collect();

        let narrowing = micro_engine.narrow(&micro_candidates);
        MicroSimEngine::print_comparison(&narrowing.candidates);

        // Determine the overall winner: prefer micro-sim winner if it
        // beats baseline, otherwise fall back to model-driven best.
        let (final_suggestion, final_micro_result) = if narrowing.beats_baseline {
            // Find the suggestion matching the micro-sim winner
            let winner = suggestions
                .iter()
                .find(|s| s.suggestion_id == narrowing.best_config_id)
                .unwrap_or(best_suggestion);
            println!("  Micro-simulation winner: {} (empirical)", narrowing.best_config_id);
            (winner, Some(narrowing.best_result.clone()))
        } else {
            println!("  Baseline held — using model-driven best: {}", best_suggestion.suggestion_id);
            (best_suggestion, None)
        };

        // ── Step 5: Save Role Profile ───────────────────────
        println!("\nStep 5/6: Saving Ramified Role Profile");
        let manager = ProfileManager::new(&self.config.project_root);

        let profile = ProfileManager::create_profile(
            &hardware,
            &self.config.role,
            final_suggestion,
            &best_result,
            results,
            self.config.llm_config.as_ref(),
        );

        let path = manager.save_profile(&profile).map_err(|e| {
            InstallerError::ProfileError(format!("{}", e))
        })?;

        // Save simulation report alongside the profile
        let report = SimulationReport::from_results(
            profile.simulation_results.clone(),
            &hardware,
            &self.config.role,
        );
        let report_path = path.with_extension("report.json");
        if let Ok(report_json) = serde_json::to_string_pretty(&report) {
            let _ = std::fs::write(&report_path, report_json);
        }

        ProfileManager::print_profile_summary(&profile);

        // ── Step 6: Ramify DNA (if micro-sim beat baseline) ─
        println!("\nStep 6/6: Ramify DNA");
        if narrowing.beats_baseline {
            println!("  Micro-simulation winner beats baseline — ramifying DNA...");

            // Find the NVRTC compilation for the winning suggestion
            let winning_compilation = muscle_compilations
                .iter()
                .find(|c| c.suggestion_id == final_suggestion.suggestion_id);

            if let Some(compilation) = winning_compilation {
                println!(
                    "  NVRTC muscle fiber for winner: {} ({} bytes PTX, {})",
                    compilation.suggestion_id,
                    compilation.compilation.ptx.len(),
                    if compilation.simulated { "simulated" } else { "real NVRTC" }
                );
            }

            match self.update_claw_dna(
                &hardware,
                final_suggestion,
                final_micro_result.as_ref(),
                winning_compilation,
            ) {
                Ok(dna_path) => {
                    println!("  .claw-dna ramified: {}", dna_path);
                }
                Err(e) => {
                    println!("  WARNING: Failed to ramify .claw-dna: {}", e);
                    println!("  (Profile was still saved successfully.)");
                }
            }
        } else {
            println!("  No DNA update — baseline was not beaten.");
            println!("  Current .claw-dna constants remain unchanged.");
        }

        println!("\n{}", "═".repeat(64));
        println!("  Installation complete!");
        println!("  Profile: {}", path.display());
        if report_path.exists() {
            println!("  Report:  {}", report_path.display());
        }
        println!("{}\n", "═".repeat(64));

        Ok((profile, path))
    }

    /// Run only the hardware probe step.
    fn run_probe(&self) -> HardwareProfile {
        if self.config.skip_probe {
            HardwareProber::simulated(self.config.gpu_index).probe()
        } else {
            HardwareProber::new(self.config.gpu_index).probe()
        }
    }

    /// Generate optimization suggestions (LLM + heuristic variations).
    async fn generate_suggestions(
        &self,
        hardware: &HardwareProfile,
    ) -> Result<Vec<OptimizationSuggestion>, InstallerError> {
        let mut suggestions = Vec::new();

        // Always include the heuristic baseline
        println!("  Generating heuristic baseline...");
        let baseline = LlmOptimizer::generate_heuristic_suggestion(
            hardware, &self.config.role);
        suggestions.push(baseline);

        // Generate heuristic variations by sweeping key parameters
        println!("  Generating {} heuristic variations...", self.config.heuristic_variations);
        let variations = self.generate_heuristic_variations(hardware);
        suggestions.extend(variations);

        // If LLM is configured, run the optimization loop
        if let Some(ref llm_config) = self.config.llm_config {
            println!("  Running LLM optimization loop ({}, {})...",
                llm_config.model, format!("{:?}", llm_config.provider));

            let optimizer = LlmOptimizer::new(llm_config.clone());
            match optimizer.optimize(hardware, &self.config.role, None).await {
                Ok(llm_suggestions) => {
                    println!("  LLM provided {} suggestions", llm_suggestions.len());
                    suggestions.extend(llm_suggestions);
                }
                Err(e) => {
                    println!("  WARNING: LLM optimization failed: {}", e);
                    println!("  Continuing with heuristic suggestions only.");
                }
            }
        } else {
            println!("  No LLM configured — using heuristic-only mode.");
        }

        Ok(suggestions)
    }

    /// Generate heuristic variations by sweeping key parameters.
    /// These complement the LLM suggestions to ensure good coverage
    /// of the parameter space.
    fn generate_heuristic_variations(
        &self,
        hardware: &HardwareProfile,
    ) -> Vec<OptimizationSuggestion> {
        let mut variations = Vec::new();
        let base = LlmOptimizer::generate_heuristic_suggestion(
            hardware, &self.config.role);

        // Variation 1: Ultra-low latency (minimal sleep, single warp)
        variations.push(OptimizationSuggestion {
            suggestion_id: "heuristic-ultra-low-latency".to_string(),
            round: 0,
            block_size: 32,
            grid_size: 1,
            idle_sleep_ns: 10,
            warps_per_block: 1,
            loop_unroll_factor: 8,
            command_batch_size: 1,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            l1_cache_preference: 1,
            reasoning: "Ultra-low latency: minimal sleep, single warp, max unroll".to_string(),
            estimated_improvement_pct: 0.0,
            ..base.clone()
        });

        // Variation 2: High throughput (multi-warp, batching)
        variations.push(OptimizationSuggestion {
            suggestion_id: "heuristic-high-throughput".to_string(),
            round: 0,
            block_size: 128,
            grid_size: 4,
            idle_sleep_ns: 200,
            warps_per_block: 4,
            loop_unroll_factor: 4,
            command_batch_size: 8,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            l1_cache_preference: 0,
            reasoning: "High throughput: 4 warps, 4 blocks, batch-8".to_string(),
            estimated_improvement_pct: 0.0,
            ..base.clone()
        });

        // Variation 3: Power efficient (higher sleep, fewer warps)
        variations.push(OptimizationSuggestion {
            suggestion_id: "heuristic-power-efficient".to_string(),
            round: 0,
            block_size: 32,
            grid_size: 1,
            idle_sleep_ns: 1000,
            warps_per_block: 1,
            loop_unroll_factor: 2,
            command_batch_size: 4,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            l1_cache_preference: 2, // Prefer shared memory
            reasoning: "Power efficient: higher sleep, lower unroll".to_string(),
            estimated_improvement_pct: 0.0,
            ..base.clone()
        });

        // Variation 4: Balanced multi-block
        variations.push(OptimizationSuggestion {
            suggestion_id: "heuristic-balanced-multi".to_string(),
            round: 0,
            block_size: 64,
            grid_size: 2,
            idle_sleep_ns: 100,
            warps_per_block: 2,
            loop_unroll_factor: 4,
            command_batch_size: 2,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            l1_cache_preference: 1,
            reasoning: "Balanced: 2 warps, 2 blocks, moderate batch".to_string(),
            estimated_improvement_pct: 0.0,
            ..base.clone()
        });

        // Variation 5: AoS layout test (for comparison)
        variations.push(OptimizationSuggestion {
            suggestion_id: "heuristic-aos-layout".to_string(),
            round: 0,
            use_soa_layout: false,
            enable_warp_aggregation: false,
            l1_cache_preference: 0,
            reasoning: "AoS layout comparison: no SoA, no warp aggregation".to_string(),
            estimated_improvement_pct: 0.0,
            ..base.clone()
        });

        // Only return the requested number of variations
        variations.truncate(self.config.heuristic_variations as usize);
        variations
    }

    /// Update the `.claw-dna` file with improved muscle fiber settings.
    ///
    /// Called when the micro-simulation proves a candidate config beats
    /// the baseline. Loads or creates the DNA file, updates only the
    /// muscle fiber parameters, and writes back in-place.
    ///
    /// If an NVRTC compilation was produced for the winning suggestion,
    /// the compiled PTX is stored in the DNA's muscle fiber kernel_source
    /// field so it can be loaded directly at runtime without recompilation.
    fn update_claw_dna(
        &self,
        hardware: &HardwareProfile,
        suggestion: &OptimizationSuggestion,
        micro_result: Option<&MicroSimResult>,
        nvrtc_compilation: Option<&nvrtc_muscle_compiler::MuscleFiberCompilation>,
    ) -> Result<String, InstallerError> {
        let dna_dir = self.config.project_root.join(".cudaclaw").join("dna");
        std::fs::create_dir_all(&dna_dir).map_err(|e| {
            InstallerError::ProfileError(format!("Failed to create dna dir: {}", e))
        })?;

        let role_name = &self.config.role.role_name;
        let hw_short = hardware.gpu_name
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
        let sm_tag = format!("sm{}", hardware.compute_capability.replace('.', ""));
        let filename = format!("{}_{}.claw-dna", role_name, format!("{}_{}", hw_short, sm_tag));
        let dna_path = dna_dir.join(&filename);
        let path_str = dna_path.display().to_string();

        // Load existing DNA or create a new default
        let mut dna = if dna_path.exists() {
            RamifiedRole::load_from_file(&path_str).unwrap_or_else(|_| {
                RamifiedRole::default_spreadsheet_engine()
            })
        } else {
            let mut d = RamifiedRole::default_spreadsheet_engine();
            d.role = role_name.clone();
            d.name = format!("{}_{}", role_name, format!("{}_{}", hw_short, sm_tag));
            d
        };

        // Update muscle fibers from the winning suggestion
        self.apply_suggestion_to_fibers(&mut dna, suggestion, micro_result);

        // If NVRTC compilation is available, store the compiled PTX
        // in the winning fiber's kernel_source so it can be loaded
        // at runtime without recompilation.
        if let Some(compilation) = nvrtc_compilation {
            // Store NVRTC-compiled PTX in the cell_update fiber
            // (the primary persistent-worker kernel)
            if let Some(fiber) = dna.muscle_fibers.get_mut("cell_update") {
                fiber.kernel_source = crate::dna::DnaKernelSource::NvrtcSource {
                    source: compilation.compilation.ptx.clone(),
                    program_name: format!("muscle_{}.cu", compilation.suggestion_id),
                    gpu_arch: compilation.target_arch.clone(),
                    defines: compilation.injected_constants.clone(),
                    extra_flags: Vec::new(),
                    max_registers: 0,
                    use_fast_math: false,
                };
            }
        }

        // Record the mutation with NVRTC context
        let nvrtc_info = nvrtc_compilation
            .map(|c| format!(", NVRTC={} ({} bytes PTX)",
                if c.simulated { "simulated" } else { "real" },
                c.compilation.ptx.len()))
            .unwrap_or_default();

        dna.record_mutation(&format!(
            "Installer micro-simulation win: {} (P99={:.2}us, Throughput={:.0}cmd/s{})",
            suggestion.suggestion_id,
            micro_result.map(|r| r.p99_latency_us).unwrap_or(0.0),
            micro_result.map(|r| r.throughput_cmds_per_sec).unwrap_or(0.0),
            nvrtc_info,
        ));

        // Write back in-place
        dna.save_to_file(&path_str).map_err(|e| {
            InstallerError::ProfileError(format!("Failed to save DNA: {}", e))
        })?;

        Ok(path_str)
    }

    /// Apply an OptimizationSuggestion's parameters to the DNA's muscle fibers.
    ///
    /// Updates existing fibers in-place (block_size, shared_memory_bytes, etc.)
    /// without changing the kernel source, description, or name.
    fn apply_suggestion_to_fibers(
        &self,
        dna: &mut RamifiedRole,
        suggestion: &OptimizationSuggestion,
        micro_result: Option<&MicroSimResult>,
    ) {
        // Map suggestion fields to the relevant fiber task IDs.
        // The persistent-worker kernel maps to "idle_poll" and "cell_update".
        let fiber_ids = ["cell_update", "crdt_merge", "idle_poll"];

        for task_id in &fiber_ids {
            if let Some(fiber) = dna.muscle_fibers.get_mut(task_id) {
                // Update launch parameters from the suggestion
                fiber.block_size = suggestion.block_size;
                fiber.shared_memory_bytes = suggestion.shared_memory_bytes;
                fiber.uses_warp_aggregation = suggestion.enable_warp_aggregation;

                // Task-specific overrides
                match *task_id {
                    "idle_poll" => {
                        // Persistent poller: always single warp
                        fiber.block_size = 32;
                        fiber.is_persistent = true;
                    }
                    "crdt_merge" => {
                        // CRDT merge always needs PTX CAS
                        fiber.uses_ptx_cas = true;
                    }
                    _ => {}
                }

                // Record micro-simulation measurements if available
                if let Some(result) = micro_result {
                    fiber.measurement_count += 1;
                    if result.p99_latency_us < fiber.best_measured_latency_us {
                        fiber.best_measured_latency_us = result.p99_latency_us;
                    }
                    if result.p99_latency_us > fiber.worst_measured_latency_us {
                        fiber.worst_measured_latency_us = result.p99_latency_us;
                    }
                    fiber.expected_p99_us = result.p99_latency_us;
                    fiber.expected_throughput_ops = result.throughput_cmds_per_sec;
                }
            }
        }

        // Also update the "batch_process" fiber with throughput-oriented params
        if let Some(fiber) = dna.muscle_fibers.get_mut("batch_process") {
            // Batch processor benefits from larger block sizes
            fiber.block_size = suggestion.block_size.max(256);
            fiber.shared_memory_bytes = suggestion.shared_memory_bytes;
            fiber.uses_warp_aggregation = suggestion.enable_warp_aggregation;
        }
    }

    /// List all saved profiles.
    pub fn list_profiles(&self) -> Result<(), InstallerError> {
        let manager = ProfileManager::new(&self.config.project_root);

        let roles = manager.list_roles().map_err(|e| {
            InstallerError::ProfileError(format!("{}", e))
        })?;

        if roles.is_empty() {
            println!("No saved profiles found in {}", manager.roles_dir().display());
            return Ok(());
        }

        println!("\n{}", "═".repeat(64));
        println!("  Saved Role Profiles");
        println!("{}", "═".repeat(64));

        for role in &roles {
            let profiles = manager.list_profiles(role).map_err(|e| {
                InstallerError::ProfileError(format!("{}", e))
            })?;

            println!("\n  Role: {}", role);
            for profile_id in &profiles {
                println!("    └─ {}", profile_id);
            }
        }

        println!("{}\n", "═".repeat(64));
        Ok(())
    }

    /// Show a specific profile.
    pub fn show_profile(&self) -> Result<(), InstallerError> {
        let manager = ProfileManager::new(&self.config.project_root);
        let prober = HardwareProber::simulated(self.config.gpu_index);
        let hardware = prober.probe();

        let profile = manager.load_profile(&self.config.role.role_name, &hardware)
            .map_err(|e| InstallerError::ProfileError(format!("{}", e)))?;

        ProfileManager::print_profile_summary(&profile);
        Ok(())
    }
}

// ============================================================
// CLI Argument Parsing
// ============================================================

/// Parse installer CLI arguments from the command line.
/// Returns None if the arguments don't match the installer subcommand.
pub fn parse_installer_args(args: &[String]) -> Option<InstallerConfig> {
    if args.is_empty() || args[0] != "install" {
        return None;
    }

    let mut config = InstallerConfig::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--role" => {
                i += 1;
                if i < args.len() {
                    config.role = match args[i].as_str() {
                        "spreadsheet_engine" => RoleContext::spreadsheet_engine(),
                        "realtime_crdt" => RoleContext::realtime_crdt(),
                        "batch_processor" => RoleContext::batch_processor(),
                        name => {
                            // Custom role with default weights
                            let mut role = RoleContext::spreadsheet_engine();
                            role.role_name = name.to_string();
                            role
                        }
                    };
                }
            }
            "--llm-provider" | "--provider" => {
                i += 1;
                if i < args.len() {
                    let provider = args[i].as_str();
                    let api_key = find_arg(args, "--api-key").unwrap_or_default();
                    let model = find_arg(args, "--model")
                        .unwrap_or_else(|| default_model(provider));

                    config.llm_config = Some(match provider {
                        "openai" => LlmConfig::openai(&api_key, &model),
                        "deepseek" => LlmConfig::deepseek(&api_key, &model),
                        "anthropic" | "claude" => LlmConfig::anthropic(&api_key, &model),
                        _ => {
                            let base_url = find_arg(args, "--base-url")
                                .unwrap_or_else(|| provider.to_string());
                            LlmConfig::custom(&base_url, &api_key, &model)
                        }
                    });
                }
            }
            "--base-url" => {
                i += 1;
                if i < args.len() {
                    let base_url = &args[i];
                    let api_key = find_arg(args, "--api-key").unwrap_or_default();
                    let model = find_arg(args, "--model")
                        .unwrap_or_else(|| "default".to_string());
                    config.llm_config = Some(LlmConfig::custom(base_url, &api_key, &model));
                }
            }
            "--heuristic-only" => {
                config.llm_config = None;
            }
            "--skip-probe" => {
                config.skip_probe = true;
            }
            "--gpu" => {
                i += 1;
                if i < args.len() {
                    config.gpu_index = args[i].parse().unwrap_or(0);
                }
            }
            "--project-root" => {
                i += 1;
                if i < args.len() {
                    config.project_root = std::path::PathBuf::from(&args[i]);
                }
            }
            "--variations" => {
                i += 1;
                if i < args.len() {
                    config.heuristic_variations = args[i].parse().unwrap_or(5);
                }
            }
            "--rounds" => {
                i += 1;
                if i < args.len() {
                    if let Some(ref mut llm) = config.llm_config {
                        llm.optimization_rounds = args[i].parse().unwrap_or(3);
                    }
                }
            }
            _ => {} // Skip unknown args
        }
        i += 1;
    }

    Some(config)
}

/// Find a named argument value in the args list.
fn find_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
}

/// Default model for each provider.
fn default_model(provider: &str) -> String {
    match provider {
        "openai" => "gpt-4".to_string(),
        "deepseek" => "deepseek-coder".to_string(),
        "anthropic" | "claude" => "claude-3-opus-20240229".to_string(),
        _ => "default".to_string(),
    }
}

/// Print installer usage help.
pub fn print_installer_help() {
    println!(r#"
CudaClaw Intelligent Installer
==============================

USAGE:
    cargo run -- install [OPTIONS]

OPTIONS:
    --role <ROLE>              Role to optimize for (default: spreadsheet_engine)
                               Built-in: spreadsheet_engine, realtime_crdt, batch_processor

    --llm-provider <PROVIDER>  LLM provider: openai, deepseek, anthropic
    --base-url <URL>           Custom OpenAI-compatible endpoint URL
    --api-key <KEY>            API key for the LLM provider
    --model <MODEL>            LLM model name (provider-specific defaults apply)
    --rounds <N>               Number of LLM optimization rounds (default: 3)

    --heuristic-only           Skip LLM, use hardware-aware heuristics only
    --skip-probe               Skip hardware probe, use simulated GPU data
    --gpu <INDEX>              GPU device index to probe (default: 0)

    --project-root <PATH>      Project root for .cudaclaw/roles/ (default: .)
    --variations <N>           Number of heuristic variations to test (default: 5)

    --list-profiles            List all saved role profiles
    --show-profile             Show the profile for the current role + hardware

EXAMPLES:
    # Quick heuristic-only optimization:
    cargo run -- install --role spreadsheet_engine --heuristic-only

    # Full LLM optimization with OpenAI:
    cargo run -- install --role spreadsheet_engine \
        --llm-provider openai --api-key $OPENAI_API_KEY

    # Using a local Ollama instance:
    cargo run -- install --role spreadsheet_engine \
        --base-url http://localhost:11434/v1 --model llama3

    # List saved profiles:
    cargo run -- install --list-profiles
"#);
}

// ============================================================
// Error Types
// ============================================================

#[derive(Debug)]
pub enum InstallerError {
    ProbeError(String),
    LlmError(String),
    SimulationError(String),
    ProfileError(String),
    NoSuggestions,
}

impl std::fmt::Display for InstallerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallerError::ProbeError(msg) => write!(f, "Probe error: {}", msg),
            InstallerError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            InstallerError::SimulationError(msg) => write!(f, "Simulation error: {}", msg),
            InstallerError::ProfileError(msg) => write!(f, "Profile error: {}", msg),
            InstallerError::NoSuggestions => write!(f, "No suggestions generated"),
        }
    }
}

impl std::error::Error for InstallerError {}

impl From<llm_optimizer::LlmError> for InstallerError {
    fn from(e: llm_optimizer::LlmError) -> Self {
        InstallerError::LlmError(format!("{}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_installer_args_basic() {
        let args = vec![
            "install".to_string(),
            "--role".to_string(),
            "spreadsheet_engine".to_string(),
            "--heuristic-only".to_string(),
        ];

        let config = parse_installer_args(&args).unwrap();
        assert_eq!(config.role.role_name, "spreadsheet_engine");
        assert!(config.llm_config.is_none());
    }

    #[test]
    fn test_parse_installer_args_with_llm() {
        let args = vec![
            "install".to_string(),
            "--role".to_string(),
            "realtime_crdt".to_string(),
            "--llm-provider".to_string(),
            "openai".to_string(),
            "--api-key".to_string(),
            "test-key".to_string(),
            "--model".to_string(),
            "gpt-4-turbo".to_string(),
        ];

        let config = parse_installer_args(&args).unwrap();
        assert_eq!(config.role.role_name, "realtime_crdt");
        let llm = config.llm_config.unwrap();
        assert!(llm.base_url.contains("openai.com"));
        assert_eq!(llm.model, "gpt-4-turbo");
    }

    #[test]
    fn test_parse_installer_args_not_install() {
        let args = vec!["benchmark".to_string()];
        assert!(parse_installer_args(&args).is_none());
    }

    #[test]
    fn test_parse_installer_args_custom_url() {
        let args = vec![
            "install".to_string(),
            "--base-url".to_string(),
            "http://localhost:11434/v1".to_string(),
            "--model".to_string(),
            "llama3".to_string(),
        ];

        let config = parse_installer_args(&args).unwrap();
        let llm = config.llm_config.unwrap();
        assert_eq!(llm.base_url, "http://localhost:11434/v1");
        assert_eq!(llm.model, "llama3");
        assert_eq!(llm.provider, LlmProvider::OpenAiCompatible);
    }

    #[tokio::test]
    async fn test_installer_heuristic_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = InstallerConfig {
            role: RoleContext::spreadsheet_engine(),
            llm_config: None,
            gpu_index: 0,
            project_root: tmp.path().to_path_buf(),
            skip_probe: true,
            heuristic_variations: 3,
            simulation_noise: 0.0,
        };

        let installer = Installer::new(config);
        let (profile, path) = installer.run().await.unwrap();

        assert_eq!(profile.role_name, "spreadsheet_engine");
        assert!(path.exists());
        assert!(profile.best_score > 0.0);
        assert!(!profile.simulation_results.is_empty());
    }

    #[test]
    fn test_list_profiles_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = InstallerConfig {
            project_root: tmp.path().to_path_buf(),
            ..Default::default()
        };

        let installer = Installer::new(config);
        installer.list_profiles().unwrap();
    }
}
