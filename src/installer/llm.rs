
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::env;

use crate::installer::hardware_probe::HardwareProfile;
use crate::installer::role_profile::HardwareFingerprint;

#[derive(Serialize, Deserialize, Debug)]
pub struct OptimizationParameters {
    pub block_size: u32,
    pub warps_per_block: u32,
    pub unroll_factor: u32,
}

pub struct LlmClient {
    client: Client,
    base_url: String,
    model_name: String,
    api_key: String,
}

impl LlmClient {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        dotenv::dotenv().ok();

        let base_url = env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model_name = env::var("LLM_MODEL_NAME").unwrap_or_else(|_| "gpt-3.5-turbo".to_string());
        let api_key = env::var("LLM_API_KEY")?;

        let client = Client::new();

        Ok(Self {
            client,
            base_url,
            model_name,
            api_key,
        })
    }

    pub async fn generate_optimization_plan(
        &self,
        hardware_profile: &HardwareProfile,
        task_description: &str,
    ) -> Result<OptimizationParameters, Box<dyn std::error::Error>> {
        let hardware_fingerprint = HardwareFingerprint::from_profile(hardware_profile);

        let prompt = format!(
            "Given the following hardware specifications:
- GPU: {}
- Compute Capability: {}
- SM Count: {}
- Architecture: {}
- Max Threads per Block: {}

And the task description: '{}',

Please provide the optimal CUDA kernel launch parameters in JSON format.
The JSON object should contain the following keys: 'block_size', 'warps_per_block', and 'unroll_factor'.",
            hardware_fingerprint.gpu_name,
            hardware_fingerprint.compute_capability,
            hardware_fingerprint.sm_count,
            hardware_fingerprint.architecture,
            hardware_profile.max_threads_per_block,
            task_description
        );

        let response = self.client
            .post(&format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides CUDA optimization parameters."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": { "type": "json_object" }
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let choices = response["choices"].as_array().ok_or("Invalid response format: missing 'choices'")?;
        let message = choices[0]["message"]["content"].as_str().ok_or("Invalid response format: missing 'content'")?;

        let params: OptimizationParameters = serde_json::from_str(message)?;

        // Constraint Guard
        if params.block_size > hardware_profile.max_threads_per_block {
            return Err(format!(
                "LLM suggested block_size {} exceeds hardware limit of {}",
                params.block_size,
                hardware_profile.max_threads_per_block
            ).into());
        }

        Ok(params)
    }
}
