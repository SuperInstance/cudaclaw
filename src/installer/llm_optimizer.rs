// ============================================================
// LLM Optimization Loop — Configurable AI-Driven Tuning
// ============================================================
//
// Connects to a user-provided LLM (Deepseek, Claude, or OpenAI)
// via a configurable base_url. Sends hardware probe data and the
// cudaclaw role context (e.g., "spreadsheet_engine") to the LLM,
// which suggests optimized PTX/CUDA constants.
//
// SUPPORTED PROVIDERS:
//   - OpenAI-compatible (base_url: https://api.openai.com/v1)
//   - Deepseek         (base_url: https://api.deepseek.com/v1)
//   - Anthropic/Claude (base_url: https://api.anthropic.com/v1)
//   - Any OpenAI-compatible endpoint (Ollama, vLLM, etc.)
//
// PROTOCOL:
//   1. Build a structured prompt with hardware profile + role context
//   2. POST to {base_url}/chat/completions (OpenAI format)
//      or {base_url}/messages (Anthropic format)
//   3. Parse the JSON response for suggested constants
//   4. Validate suggestions against hardware constraints
//   5. Return an OptimizationSuggestion for simulated fine-tuning
//
// ============================================================

use serde::{Deserialize, Serialize};
use crate::installer::hardware_probe::HardwareProfile;

// ============================================================
// LLM Provider Configuration
// ============================================================

/// Supported LLM provider types.
/// Determines the API format used for requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LlmProvider {
    /// OpenAI-compatible API (also works for Deepseek, Ollama, vLLM)
    OpenAiCompatible,
    /// Anthropic Claude API (uses /messages endpoint)
    Anthropic,
}

/// Configuration for the LLM connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Base URL for the LLM API (e.g., "https://api.openai.com/v1")
    pub base_url: String,

    /// API key for authentication
    pub api_key: String,

    /// Model identifier (e.g., "gpt-4", "deepseek-coder", "claude-3-opus")
    pub model: String,

    /// Provider type (determines API format)
    pub provider: LlmProvider,

    /// Maximum tokens for the response
    pub max_tokens: u32,

    /// Temperature for generation (0.0 = deterministic, 1.0 = creative)
    /// Lower values preferred for optimization tasks.
    pub temperature: f32,

    /// Number of optimization rounds to run
    pub optimization_rounds: u32,
}

impl LlmConfig {
    /// Create an OpenAI-compatible configuration.
    pub fn openai(api_key: &str, model: &str) -> Self {
        LlmConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            provider: LlmProvider::OpenAiCompatible,
            max_tokens: 4096,
            temperature: 0.2,
            optimization_rounds: 3,
        }
    }

    /// Create a Deepseek configuration.
    pub fn deepseek(api_key: &str, model: &str) -> Self {
        LlmConfig {
            base_url: "https://api.deepseek.com/v1".to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            provider: LlmProvider::OpenAiCompatible,
            max_tokens: 4096,
            temperature: 0.2,
            optimization_rounds: 3,
        }
    }

    /// Create an Anthropic/Claude configuration.
    pub fn anthropic(api_key: &str, model: &str) -> Self {
        LlmConfig {
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            provider: LlmProvider::Anthropic,
            max_tokens: 4096,
            temperature: 0.2,
            optimization_rounds: 3,
        }
    }

    /// Create a custom OpenAI-compatible configuration (Ollama, vLLM, etc.)
    pub fn custom(base_url: &str, api_key: &str, model: &str) -> Self {
        LlmConfig {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            provider: LlmProvider::OpenAiCompatible,
            max_tokens: 4096,
            temperature: 0.2,
            optimization_rounds: 3,
        }
    }
}

// ============================================================
// Optimization Suggestion
// ============================================================

/// A set of optimized CUDA/PTX constants suggested by the LLM.
/// These are the tunable parameters that the simulated fine-tuning
/// module will evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Unique identifier for this suggestion
    pub suggestion_id: String,

    /// Which optimization round produced this (1-indexed)
    pub round: u32,

    /// Block size for persistent worker kernel launch
    pub block_size: u32,

    /// Number of blocks to launch (grid size)
    pub grid_size: u32,

    /// Shared memory per block (bytes) to request via
    /// cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
    pub shared_memory_bytes: u32,

    /// Loop unrolling factor for inner processing loops
    /// (passed as #pragma unroll N in CUDA)
    pub loop_unroll_factor: u32,

    /// Number of commands to batch before warp-broadcast
    /// (affects queue drain rate vs. latency)
    pub command_batch_size: u32,

    /// __nanosleep delay when queue is empty (nanoseconds).
    /// Lower = lower latency but higher power.
    /// Higher = lower power but higher latency.
    pub idle_sleep_ns: u32,

    /// Number of warps per block to use for persistent polling
    pub warps_per_block: u32,

    /// L1 cache preference: 0 = default, 1 = prefer L1,
    /// 2 = prefer shared memory, 3 = prefer equal
    pub l1_cache_preference: u32,

    /// Whether to enable warp-aggregated writes
    /// (beneficial when contention_sensitivity_ratio > 5x)
    pub enable_warp_aggregation: bool,

    /// Whether to use SoA (Structure-of-Arrays) memory layout
    /// (beneficial when sequential_read_bandwidth >> random_read_bandwidth)
    pub use_soa_layout: bool,

    /// Prefix-sum frontier compaction batch size for formula recalc
    pub frontier_batch_size: u32,

    /// atomicCAS backoff initial delay (nanoseconds)
    pub cas_backoff_initial_ns: u32,

    /// atomicCAS backoff maximum delay (nanoseconds)
    pub cas_backoff_max_ns: u32,

    /// LLM's reasoning for these choices (for human review)
    pub reasoning: String,

    /// Estimated performance improvement (LLM's prediction)
    pub estimated_improvement_pct: f64,
}

impl Default for OptimizationSuggestion {
    fn default() -> Self {
        OptimizationSuggestion {
            suggestion_id: "default".to_string(),
            round: 0,
            block_size: 32,
            grid_size: 1,
            shared_memory_bytes: 49152,
            loop_unroll_factor: 4,
            command_batch_size: 1,
            idle_sleep_ns: 100,
            warps_per_block: 1,
            l1_cache_preference: 1,
            enable_warp_aggregation: true,
            use_soa_layout: true,
            frontier_batch_size: 256,
            cas_backoff_initial_ns: 100,
            cas_backoff_max_ns: 10000,
            reasoning: "Default conservative configuration".to_string(),
            estimated_improvement_pct: 0.0,
        }
    }
}

// ============================================================
// Role Context
// ============================================================

/// Describes the cudaclaw role for the LLM optimization context.
/// Different roles have different performance priorities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleContext {
    /// Role name (e.g., "spreadsheet_engine", "realtime_crdt", "batch_processor")
    pub role_name: String,

    /// Priority weights for optimization objectives (0.0 - 1.0)
    pub latency_weight: f64,
    pub throughput_weight: f64,
    pub power_efficiency_weight: f64,

    /// Expected workload characteristics
    pub avg_commands_per_second: u64,
    pub avg_cells_per_edit: u32,
    pub formula_dependency_depth: u32,

    /// Whether this role uses persistent kernels
    pub uses_persistent_kernel: bool,

    /// Whether CRDT merge operations are frequent
    pub heavy_crdt_merges: bool,

    /// Additional context for the LLM (free-form)
    pub additional_context: String,
}

impl RoleContext {
    /// Create a "spreadsheet_engine" role context.
    /// Optimizes for low-latency cell edits with formula recalculation.
    pub fn spreadsheet_engine() -> Self {
        RoleContext {
            role_name: "spreadsheet_engine".to_string(),
            latency_weight: 0.7,
            throughput_weight: 0.2,
            power_efficiency_weight: 0.1,
            avg_commands_per_second: 100_000,
            avg_cells_per_edit: 1,
            formula_dependency_depth: 5,
            uses_persistent_kernel: true,
            heavy_crdt_merges: false,
            additional_context: "Low-latency interactive spreadsheet editing with \
                formula dependency chains. P99 RTT target < 8 microseconds. \
                Uses lock-free SPSC queue in unified memory with persistent \
                GPU kernel polling via __threadfence_system().".to_string(),
        }
    }

    /// Create a "realtime_crdt" role context.
    /// Optimizes for high-throughput CRDT merges across distributed nodes.
    pub fn realtime_crdt() -> Self {
        RoleContext {
            role_name: "realtime_crdt".to_string(),
            latency_weight: 0.4,
            throughput_weight: 0.5,
            power_efficiency_weight: 0.1,
            avg_commands_per_second: 500_000,
            avg_cells_per_edit: 32,
            formula_dependency_depth: 3,
            uses_persistent_kernel: true,
            heavy_crdt_merges: true,
            additional_context: "Multi-node CRDT synchronization with frequent \
                merge operations. Warp-aggregated CAS writes are critical. \
                Batch processing of 32+ edits per warp cycle preferred.".to_string(),
        }
    }

    /// Create a "batch_processor" role context.
    /// Optimizes for bulk operations (imports, recalculations).
    pub fn batch_processor() -> Self {
        RoleContext {
            role_name: "batch_processor".to_string(),
            latency_weight: 0.1,
            throughput_weight: 0.8,
            power_efficiency_weight: 0.1,
            avg_commands_per_second: 1_000_000,
            avg_cells_per_edit: 256,
            formula_dependency_depth: 10,
            uses_persistent_kernel: false,
            heavy_crdt_merges: false,
            additional_context: "Bulk spreadsheet import/recalculation. \
                Maximize throughput over latency. Multi-block kernel launch. \
                Full SoA layout with coalesced memory access critical.".to_string(),
        }
    }
}

// ============================================================
// LLM Optimizer
// ============================================================

/// The LLM optimization loop engine.
/// Sends hardware profile + role context to the LLM and
/// parses optimization suggestions.
pub struct LlmOptimizer {
    config: LlmConfig,
}

impl LlmOptimizer {
    pub fn new(config: LlmConfig) -> Self {
        LlmOptimizer { config }
    }

    /// Run the full optimization loop.
    /// Performs config.optimization_rounds rounds, each building
    /// on the previous suggestion's results.
    ///
    /// Returns all suggestions from each round (the caller picks
    /// the best after simulated fine-tuning).
    pub async fn optimize(
        &self,
        hardware: &HardwareProfile,
        role: &RoleContext,
        previous_results: Option<&[SimulationResult]>,
    ) -> Result<Vec<OptimizationSuggestion>, LlmError> {
        let mut suggestions = Vec::new();
        let mut prev_results = previous_results.map(|r| r.to_vec());

        for round in 1..=self.config.optimization_rounds {
            println!("  [LLM Round {}/{}] Requesting optimization...",
                round, self.config.optimization_rounds);

            let prompt = self.build_prompt(hardware, role, round, prev_results.as_deref());
            let response = self.call_llm(&prompt).await?;
            let suggestion = self.parse_suggestion(&response, round)?;

            println!("    Block size: {}, Grid: {}, Shared mem: {} KB",
                suggestion.block_size, suggestion.grid_size,
                suggestion.shared_memory_bytes / 1024);
            println!("    Unroll: {}x, Sleep: {} ns, Warps/block: {}",
                suggestion.loop_unroll_factor, suggestion.idle_sleep_ns,
                suggestion.warps_per_block);
            println!("    Warp aggregation: {}, SoA: {}",
                suggestion.enable_warp_aggregation, suggestion.use_soa_layout);
            println!("    Estimated improvement: {:.1}%",
                suggestion.estimated_improvement_pct);

            suggestions.push(suggestion);

            // In a real loop, we'd run the simulation here and feed
            // results back into the next round. The caller handles
            // this via the simulated_finetuning module.
            if round < self.config.optimization_rounds {
                println!("    (Next round will incorporate simulation results)");
            }
        }

        Ok(suggestions)
    }

    /// Run a single optimization round (for external simulation loop).
    pub async fn optimize_single_round(
        &self,
        hardware: &HardwareProfile,
        role: &RoleContext,
        round: u32,
        previous_results: Option<&[SimulationResult]>,
    ) -> Result<OptimizationSuggestion, LlmError> {
        let prompt = self.build_prompt(hardware, role, round, previous_results);
        let response = self.call_llm(&prompt).await?;
        self.parse_suggestion(&response, round)
    }

    /// Build the structured prompt for the LLM.
    fn build_prompt(
        &self,
        hardware: &HardwareProfile,
        role: &RoleContext,
        round: u32,
        previous_results: Option<&[SimulationResult]>,
    ) -> String {
        let hardware_json = serde_json::to_string_pretty(hardware)
            .unwrap_or_else(|_| "{}".to_string());
        let role_json = serde_json::to_string_pretty(role)
            .unwrap_or_else(|_| "{}".to_string());

        let mut prompt = format!(
r#"You are an expert CUDA performance engineer optimizing a GPU-accelerated spreadsheet engine called "cudaclaw".

## Hardware Profile
The target GPU has been probed with micro-benchmarks. Here is the complete hardware profile:
```json
{hardware_json}
```

## Role Context
The application role being optimized:
```json
{role_json}
```

## Optimization Round {round}

Based on the hardware characteristics and role requirements, suggest optimal values for the following CUDA/PTX constants. Your suggestions should maximize the role's priority weights (latency: {latency_w:.1}, throughput: {throughput_w:.1}, power: {power_w:.1}).

## Key Constraints
- block_size must be a multiple of 32 (warp size) and <= {max_threads_per_block}
- shared_memory_bytes must not exceed {max_shared_mem} bytes
- warps_per_block = block_size / 32
- idle_sleep_ns: lower = better latency but more power. Consider the PCIe __threadfence_system overhead of {fence_ns:.0} ns.
- enable_warp_aggregation: recommended if contention_sensitivity_ratio > 5x (currently {contention_ratio:.1}x)
- use_soa_layout: recommended if sequential bandwidth ({seq_bw:.0} GB/s) >> random bandwidth ({rand_bw:.0} GB/s)
- L1 cache is {l1_kb} KB per SM; shared memory can be configured up to {shared_max_kb} KB per block
- The persistent worker uses a lock-free SPSC queue in unified memory

## L1/L2 Cache Profile — PTX Optimization Focus
The following cache metrics are critical for your PTX constant choices. Your suggestions
MUST be specifically tuned to this GPU's cache hierarchy:

- **L1 cache per SM**: {l1_kb} KB — this determines the maximum working set that fits
  in L1 without spilling to L2. Set shared_memory_bytes to leave enough L1 capacity
  for the hot polling path. On this GPU, the L1/shared memory split is configurable:
  l1_cache_preference=1 maximizes L1, l1_cache_preference=2 maximizes shared memory.
- **L2 cache total**: {l2_kb} KB ({l2_mb:.1} MB) — the L2 must hold the command ring
  buffer (12 KB), the cell grid working set, and dependency graph pages. If the
  working set exceeds L2, performance drops catastrophically due to DRAM round-trips.
- **L1 hit latency**: {l1_hit_ns:.1} ns — baseline for coalesced SoA reads.
  Choose loop_unroll_factor so that the unrolled loop body fits within
  {l1_kb} KB of instruction cache.
- **L2 hit latency**: {l2_hit_ns:.1} ns — fallback for AoS reads or L1 misses.
  If l2_hit_ns/l1_hit_ns > 3x, strongly prefer SoA layout and L1-biased cache config.
- **L1→L2 latency ratio**: {l1_l2_ratio:.1}x — higher ratio means L1 misses are
  expensive. Favor smaller block_size to reduce L1 pressure per SM.
- **Global memory latency**: {global_latency_ns:.1} ns — DRAM round-trip cost.
  This is the penalty for L2 misses. Keep the persistent worker's hot data
  (queue head, cell values) within L2 at all times.

Your PTX constants should specifically account for:
1. Set block_size so that block_size * per-thread-register-footprint fits within
   the SM's register file without spilling to local memory.
2. Set shared_memory_bytes to cache exactly the hot cells being edited, leaving
   the remaining L1 capacity for the polling loop's instruction/data cache.
3. Choose loop_unroll_factor based on L1 instruction cache capacity — over-unrolling
   on a {l1_kb} KB L1 causes I-cache thrashing.
4. Set l1_cache_preference based on whether the workload is read-heavy (prefer L1=1)
   or write-heavy with reuse (prefer shared=2)."#,
            hardware_json = hardware_json,
            role_json = role_json,
            round = round,
            latency_w = role.latency_weight,
            throughput_w = role.throughput_weight,
            power_w = role.power_efficiency_weight,
            max_threads_per_block = hardware.max_threads_per_block,
            max_shared_mem = hardware.max_shared_memory_per_block,
            fence_ns = hardware.pcie_profile.threadfence_system_overhead_ns,
            contention_ratio = hardware.atomic_throughput.contention_sensitivity_ratio,
            seq_bw = hardware.memory_latency.sequential_read_bandwidth_gbps,
            rand_bw = hardware.memory_latency.random_read_bandwidth_gbps,
            l1_kb = hardware.l1_cache_size_bytes / 1024,
            l2_kb = hardware.l2_cache_size_bytes / 1024,
            l2_mb = hardware.l2_cache_size_bytes as f64 / (1024.0 * 1024.0),
            shared_max_kb = hardware.max_shared_memory_per_block / 1024,
            l1_hit_ns = hardware.memory_latency.l1_hit_latency_ns,
            l2_hit_ns = hardware.memory_latency.l2_hit_latency_ns,
            l1_l2_ratio = if hardware.memory_latency.l1_hit_latency_ns > 0.0 {
                hardware.memory_latency.l2_hit_latency_ns / hardware.memory_latency.l1_hit_latency_ns
            } else { 1.0 },
            global_latency_ns = hardware.memory_latency.global_memory_latency_ns,
        );

        // Add previous results if available
        if let Some(results) = previous_results {
            let results_json = serde_json::to_string_pretty(results)
                .unwrap_or_else(|_| "[]".to_string());
            prompt.push_str(&format!(
                r#"

## Previous Round Results
The following configurations were tested in simulation. Use these results to refine your suggestion:
```json
{results_json}
```
Analyze what worked and what didn't. Adjust your suggestion to improve on the best result."#,
                results_json = results_json,
            ));
        }

        prompt.push_str(r#"

## Response Format
Respond with ONLY a JSON object (no markdown fences, no explanation outside the JSON). Use this exact schema:
```json
{
  "block_size": <u32, multiple of 32>,
  "grid_size": <u32>,
  "shared_memory_bytes": <u32>,
  "loop_unroll_factor": <u32, 1|2|4|8|16>,
  "command_batch_size": <u32>,
  "idle_sleep_ns": <u32>,
  "warps_per_block": <u32>,
  "l1_cache_preference": <u32, 0-3>,
  "enable_warp_aggregation": <bool>,
  "use_soa_layout": <bool>,
  "frontier_batch_size": <u32>,
  "cas_backoff_initial_ns": <u32>,
  "cas_backoff_max_ns": <u32>,
  "reasoning": "<string explaining your choices>",
  "estimated_improvement_pct": <f64>
}
```"#);

        prompt
    }

    /// Call the LLM API and return the raw response text.
    async fn call_llm(&self, prompt: &str) -> Result<String, LlmError> {
        // Build the HTTP request based on provider type
        let (url, body, headers) = match self.config.provider {
            LlmProvider::OpenAiCompatible => {
                let url = format!("{}/chat/completions", self.config.base_url);
                let body = serde_json::json!({
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a CUDA performance optimization expert. \
                                       Respond only with valid JSON matching the requested schema."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                });
                let mut headers = Vec::new();
                headers.push(("Authorization".to_string(),
                    format!("Bearer {}", self.config.api_key)));
                headers.push(("Content-Type".to_string(),
                    "application/json".to_string()));
                (url, body, headers)
            }
            LlmProvider::Anthropic => {
                let url = format!("{}/messages", self.config.base_url);
                let body = serde_json::json!({
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "system": "You are a CUDA performance optimization expert. \
                              Respond only with valid JSON matching the requested schema.",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                });
                let mut headers = Vec::new();
                headers.push(("x-api-key".to_string(), self.config.api_key.clone()));
                headers.push(("anthropic-version".to_string(), "2023-06-01".to_string()));
                headers.push(("Content-Type".to_string(), "application/json".to_string()));
                (url, body, headers)
            }
        };

        // Use reqwest to make the HTTP call
        let client = reqwest::Client::new();
        let mut request = client.post(&url).json(&body);
        for (key, value) in &headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request.send().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to connect to {}: {}", url, e))
        })?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            LlmError::NetworkError(format!("Failed to read response: {}", e))
        })?;

        if !status.is_success() {
            return Err(LlmError::ApiError {
                status_code: status.as_u16(),
                message: response_text,
            });
        }

        // Extract the content from the response based on provider
        let content = match self.config.provider {
            LlmProvider::OpenAiCompatible => {
                let resp: serde_json::Value = serde_json::from_str(&response_text)
                    .map_err(|e| LlmError::ParseError(format!(
                        "Invalid JSON response: {}", e)))?;
                resp["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
            LlmProvider::Anthropic => {
                let resp: serde_json::Value = serde_json::from_str(&response_text)
                    .map_err(|e| LlmError::ParseError(format!(
                        "Invalid JSON response: {}", e)))?;
                resp["content"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            }
        };

        if content.is_empty() {
            return Err(LlmError::ParseError(
                "Empty content in LLM response".to_string()));
        }

        Ok(content)
    }

    /// Parse the LLM's response into an OptimizationSuggestion.
    fn parse_suggestion(
        &self,
        response: &str,
        round: u32,
    ) -> Result<OptimizationSuggestion, LlmError> {
        // Strip markdown code fences if present
        let cleaned = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(cleaned)
            .map_err(|e| LlmError::ParseError(format!(
                "Failed to parse LLM response as JSON: {}. Response was: {}",
                e, &cleaned[..cleaned.len().min(200)])))?;

        let suggestion_id = format!("llm-round-{}-{}", round,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis());

        Ok(OptimizationSuggestion {
            suggestion_id,
            round,
            block_size: parsed["block_size"].as_u64().unwrap_or(32) as u32,
            grid_size: parsed["grid_size"].as_u64().unwrap_or(1) as u32,
            shared_memory_bytes: parsed["shared_memory_bytes"].as_u64().unwrap_or(49152) as u32,
            loop_unroll_factor: parsed["loop_unroll_factor"].as_u64().unwrap_or(4) as u32,
            command_batch_size: parsed["command_batch_size"].as_u64().unwrap_or(1) as u32,
            idle_sleep_ns: parsed["idle_sleep_ns"].as_u64().unwrap_or(100) as u32,
            warps_per_block: parsed["warps_per_block"].as_u64().unwrap_or(1) as u32,
            l1_cache_preference: parsed["l1_cache_preference"].as_u64().unwrap_or(1) as u32,
            enable_warp_aggregation: parsed["enable_warp_aggregation"].as_bool().unwrap_or(true),
            use_soa_layout: parsed["use_soa_layout"].as_bool().unwrap_or(true),
            frontier_batch_size: parsed["frontier_batch_size"].as_u64().unwrap_or(256) as u32,
            cas_backoff_initial_ns: parsed["cas_backoff_initial_ns"].as_u64().unwrap_or(100) as u32,
            cas_backoff_max_ns: parsed["cas_backoff_max_ns"].as_u64().unwrap_or(10000) as u32,
            reasoning: parsed["reasoning"].as_str().unwrap_or("No reasoning provided").to_string(),
            estimated_improvement_pct: parsed["estimated_improvement_pct"].as_f64().unwrap_or(0.0),
        })
    }

    /// Generate a default suggestion without calling the LLM.
    /// Uses hardware-aware heuristics based on the probe data.
    pub fn generate_heuristic_suggestion(
        hardware: &HardwareProfile,
        role: &RoleContext,
    ) -> OptimizationSuggestion {
        // Heuristic: persistent worker uses 1 warp for low-latency roles,
        // more warps for throughput-oriented roles.
        let warps = if role.latency_weight > role.throughput_weight {
            1
        } else if role.throughput_weight > 0.6 {
            4
        } else {
            2
        };

        let block_size = warps * hardware.warp_size;

        // Shared memory: use as much as possible for working set cache
        let shared_mem = hardware.max_shared_memory_per_block.min(65536);

        // Sleep: lower for latency-sensitive, higher for power-efficient
        let sleep_ns = if role.latency_weight > 0.5 {
            50
        } else if role.power_efficiency_weight > 0.3 {
            500
        } else {
            100
        };

        // Warp aggregation: enable if contention is high
        let enable_warp_agg = hardware.atomic_throughput.contention_sensitivity_ratio > 5.0;

        // SoA: enable if sequential bandwidth >> random bandwidth
        let use_soa = hardware.memory_latency.sequential_read_bandwidth_gbps
            > hardware.memory_latency.random_read_bandwidth_gbps * 2.0;

        // Unroll factor: higher for compute-bound, lower for memory-bound
        let unroll = if hardware.memory_latency.global_memory_latency_ns > 400.0 {
            8 // Memory-bound: hide latency with ILP
        } else {
            4
        };

        // CAS backoff: scale with contention ratio
        let cas_initial = if hardware.atomic_throughput.contention_sensitivity_ratio > 10.0 {
            200
        } else {
            100
        };

        OptimizationSuggestion {
            suggestion_id: "heuristic-baseline".to_string(),
            round: 0,
            block_size,
            grid_size: 1,
            shared_memory_bytes: shared_mem,
            loop_unroll_factor: unroll,
            command_batch_size: if role.avg_cells_per_edit > 16 { 4 } else { 1 },
            idle_sleep_ns: sleep_ns,
            warps_per_block: warps,
            l1_cache_preference: 1, // Prefer L1 for latency-sensitive
            enable_warp_aggregation: enable_warp_agg,
            use_soa_layout: use_soa,
            frontier_batch_size: (role.formula_dependency_depth * 64).max(256),
            cas_backoff_initial_ns: cas_initial,
            cas_backoff_max_ns: cas_initial * 100,
            reasoning: format!(
                "Heuristic baseline for {} role on {}. \
                 Latency-weight={:.1}, throughput-weight={:.1}. \
                 {} warps for {} workload. \
                 Warp aggregation {} (contention ratio {:.1}x). \
                 SoA {} (seq/rand BW ratio {:.1}x).",
                role.role_name, hardware.gpu_name,
                role.latency_weight, role.throughput_weight,
                warps, if role.latency_weight > 0.5 { "low-latency" } else { "throughput" },
                if enable_warp_agg { "ON" } else { "OFF" },
                hardware.atomic_throughput.contention_sensitivity_ratio,
                if use_soa { "ON" } else { "OFF" },
                hardware.memory_latency.sequential_read_bandwidth_gbps
                    / hardware.memory_latency.random_read_bandwidth_gbps,
            ),
            estimated_improvement_pct: 0.0,
        }
    }
}

// ============================================================
// Simulation Result (fed back to LLM for next round)
// ============================================================

/// Result from simulated fine-tuning, fed back to the LLM
/// for iterative improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Which suggestion was tested
    pub suggestion_id: String,

    /// Round number
    pub round: u32,

    /// Simulated P99 RTT (microseconds)
    pub p99_rtt_us: f64,

    /// Simulated throughput (commands/sec)
    pub throughput_cmds_per_sec: f64,

    /// Simulated power efficiency score (0-100)
    pub power_efficiency_score: f64,

    /// Overall weighted score based on role priorities
    pub overall_score: f64,

    /// Whether the P99 target was met
    pub meets_p99_target: bool,

    /// Notes about the simulation
    pub notes: String,
}

// ============================================================
// Error Types
// ============================================================

#[derive(Debug)]
pub enum LlmError {
    /// Network or connection error
    NetworkError(String),
    /// API returned an error status code
    ApiError { status_code: u16, message: String },
    /// Failed to parse the LLM response
    ParseError(String),
    /// Suggestion violates hardware constraints
    ValidationError(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::NetworkError(msg) => write!(f, "LLM Network Error: {}", msg),
            LlmError::ApiError { status_code, message } =>
                write!(f, "LLM API Error ({}): {}", status_code, message),
            LlmError::ParseError(msg) => write!(f, "LLM Parse Error: {}", msg),
            LlmError::ValidationError(msg) => write!(f, "LLM Validation Error: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_suggestion_spreadsheet() {
        let prober = crate::installer::hardware_probe::HardwareProber::simulated(0);
        let hardware = prober.probe();
        let role = RoleContext::spreadsheet_engine();

        let suggestion = LlmOptimizer::generate_heuristic_suggestion(&hardware, &role);

        assert_eq!(suggestion.block_size % 32, 0, "Block size must be warp-aligned");
        assert!(suggestion.block_size <= hardware.max_threads_per_block);
        assert!(suggestion.shared_memory_bytes <= hardware.max_shared_memory_per_block);
        assert!(suggestion.idle_sleep_ns <= 500, "Latency-sensitive role should have low sleep");
        assert!(suggestion.enable_warp_aggregation, "High contention ratio should enable aggregation");
        assert!(suggestion.use_soa_layout, "High seq/rand BW ratio should enable SoA");
    }

    #[test]
    fn test_heuristic_suggestion_batch() {
        let prober = crate::installer::hardware_probe::HardwareProber::simulated(0);
        let hardware = prober.probe();
        let role = RoleContext::batch_processor();

        let suggestion = LlmOptimizer::generate_heuristic_suggestion(&hardware, &role);

        assert!(suggestion.warps_per_block >= 2,
            "Throughput role should use multiple warps");
        assert!(suggestion.command_batch_size > 1,
            "Batch processor should batch commands");
    }

    #[test]
    fn test_parse_suggestion_valid_json() {
        let config = LlmConfig::openai("test-key", "gpt-4");
        let optimizer = LlmOptimizer::new(config);

        let response = r#"{
            "block_size": 64,
            "grid_size": 2,
            "shared_memory_bytes": 65536,
            "loop_unroll_factor": 8,
            "command_batch_size": 2,
            "idle_sleep_ns": 50,
            "warps_per_block": 2,
            "l1_cache_preference": 1,
            "enable_warp_aggregation": true,
            "use_soa_layout": true,
            "frontier_batch_size": 512,
            "cas_backoff_initial_ns": 200,
            "cas_backoff_max_ns": 20000,
            "reasoning": "Optimized for low latency",
            "estimated_improvement_pct": 15.5
        }"#;

        let suggestion = optimizer.parse_suggestion(response, 1).unwrap();
        assert_eq!(suggestion.block_size, 64);
        assert_eq!(suggestion.grid_size, 2);
        assert_eq!(suggestion.loop_unroll_factor, 8);
        assert!(suggestion.enable_warp_aggregation);
        assert!((suggestion.estimated_improvement_pct - 15.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_suggestion_with_markdown_fences() {
        let config = LlmConfig::openai("test-key", "gpt-4");
        let optimizer = LlmOptimizer::new(config);

        let response = "```json\n{\"block_size\": 128, \"grid_size\": 1}\n```";
        let suggestion = optimizer.parse_suggestion(response, 1).unwrap();
        assert_eq!(suggestion.block_size, 128);
    }

    #[test]
    fn test_llm_config_presets() {
        let openai = LlmConfig::openai("key", "gpt-4");
        assert!(openai.base_url.contains("openai.com"));
        assert_eq!(openai.provider, LlmProvider::OpenAiCompatible);

        let deepseek = LlmConfig::deepseek("key", "deepseek-coder");
        assert!(deepseek.base_url.contains("deepseek.com"));

        let anthropic = LlmConfig::anthropic("key", "claude-3");
        assert!(anthropic.base_url.contains("anthropic.com"));
        assert_eq!(anthropic.provider, LlmProvider::Anthropic);
    }
}
