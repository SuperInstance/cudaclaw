pub mod llm;
pub mod simulation;
pub mod hardware_probe;
pub mod role_profile;

use serde::{Deserialize, Serialize};
use std::fs;

use crate::installer::hardware_probe::{HardwareProber};
use crate::installer::llm::LlmClient;
use crate::installer::role_profile::{ProfileManager, RoleProfile};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MuscleFibers {
    pub block_size: u32,
    pub warps_per_block: u32,
    pub unroll_factor: u32,
}

#[derive(Debug, Default)]
pub struct InstallerConfig {
    pub role_name: Option<String>,
    pub profile_name: Option<String>,
}

pub struct Installer {
    config: InstallerConfig,
}

impl Installer {
    pub fn new(config: InstallerConfig) -> Self {
        Self { config }
    }

    pub async fn run(&self) -> Result<(RoleProfile, std::path::PathBuf), Box<dyn std::error::Error>> {
        let task_description = self.config.role_name.as_deref().unwrap_or("Persistent Spreadsheet Recalculation");

        let role_profile = bootstrap_hardware(task_description).await?;

        Ok((role_profile, std::path::PathBuf::from(".claw-dna")))
    }

    pub fn list_profiles(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Listing profiles...");
        Ok(())
    }

    pub fn show_profile(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Showing profile...");
        Ok(())
    }
}

pub fn print_installer_help() {
    println!("Installer help...");
}

pub fn parse_installer_args(_args: &[String]) -> Option<InstallerConfig> {
    Some(InstallerConfig::default())
}

/// Bootstraps the hardware, triggering the LLM-Optimization loop if necessary.
pub async fn bootstrap_hardware(
    task_description: &str,
) -> Result<RoleProfile, Box<dyn std::error::Error>> {
    let prober = HardwareProber::new(0);
    let hardware_profile = prober.probe();

    let profile_manager = ProfileManager::new(std::env::current_dir()?.as_path());

    let role_profile = match profile_manager.find_matching_profile("cudaclaw_role", &hardware_profile) {
        Some(profile) => {
            println!("Found matching profile: {}", profile.hardware_fingerprint.short_id);
            profile
        }
        None => {
            println!("No matching profile found. Triggering LLM-Optimization loop...");

            let llm_client = LlmClient::new()?;
            let optimized_params = simulation::narrowing_loop(
                &llm_client, 
                &hardware_profile, 
                task_description
            ).await?;

            let muscle_fibers = MuscleFibers {
                block_size: optimized_params.block_size,
                warps_per_block: optimized_params.warps_per_block,
                unroll_factor: optimized_params.unroll_factor,
            };

            let role_profile = RoleProfile {
                version: 1,
                role_name: "cudaclaw_role".to_string(),
                hardware_fingerprint: role_profile::HardwareFingerprint::from_profile(&hardware_profile),
                configuration: Default::default(),
                simulation_results: vec![],
                best_score: 0.0,
                meets_p99_target: false,
                achieved_p99_rtt_us: 0.0,
                achieved_throughput_cmds_per_sec: 0.0,
                provenance: Default::default(),
                role_context: Default::default(),
                created_at: 0,
                last_validated: 0,
                load_count: 0,
                muscle_fibers,
            };

            // Save the new profile
            profile_manager.save_profile(&role_profile)?;

            // Output the Role Profile
            let role_profile_json = serde_json::to_string_pretty(&role_profile)?;
            println!("\nRole Profile:\n{}", role_profile_json);

            role_profile
        }
    };

    // Serialize the final constants to .claw-dna
    let claw_dna_content = serde_json::to_string_pretty(&role_profile.muscle_fibers)?;
    fs::write(".claw-dna", claw_dna_content)?;
    println!("\n.claw-dna file created successfully.");

    Ok(role_profile)
}
