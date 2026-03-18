// PTX module for loading compiled CUDA kernels
// This module provides utilities to load PTX files compiled by build.rs

use std::fs;
use std::path::PathBuf;
use std::env;

/// Get the output directory where PTX files are compiled
fn get_out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

/// Load a PTX file by name (without .ptx extension)
pub fn load_ptx(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let out_dir = get_out_dir();
    let ptx_path = out_dir.join(format!("{}.ptx", name));

    if !ptx_path.exists() {
        return Err(format!("PTX file not found: {:?}", ptx_path).into());
    }

    fs::read_to_string(&ptx_path)
        .map_err(|e| format!("Failed to read PTX file {:?}: {}", ptx_path, e).into())
}

/// List all available PTX files
pub fn list_ptx_files() -> Vec<String> {
    let out_dir = get_out_dir();
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".ptx") {
                    files.push(name.to_string());
                }
            }
        }
    }

    files.sort();
    files
}
