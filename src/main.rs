use cuda_claw::CudaClawExecutor;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod agent;
mod cuda_claw;

use agent::{AgentDispatcher, AgentOperation, AgentType, CellRef, SuperInstance};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CudaClaw - GPU-Accelerated SmartCRDT Orchestrator");
    println!("==============================================\n");

    // Initialize CUDA
    let _ctx = cust::quick_init().expect("Failed to initialize CUDA");

    println!("CUDA initialized successfully");
    println!("GPUs available: {}", cust::device::get_device_count()?);

    // Create the CudaClaw executor
    println!("\nInitializing CudaClaw executor...");
    let mut executor = CudaClawExecutor::new()?;

    // Initialize the command queue
    println!("Initializing command queue...");
    executor.init_queue()?;

    // Get the unified memory command queue for AgentDispatcher
    let command_queue = Arc::new(Mutex::new(executor.queue.clone()));

    // Start the persistent kernel
    println!("Starting persistent GPU kernel...");
    executor.start()?;

    println!("Persistent kernel is now running on GPU\n");

    // Run latency tests
    run_latency_tests(&mut executor)?;

    // Run functional tests
    run_functional_tests(&mut executor)?;

    // Demonstrate AgentDispatcher
    run_agent_dispatcher_demo(command_queue.clone())?;

    // Shutdown the persistent kernel
    println!("\nShutting down persistent kernel...");
    executor.shutdown()?;

    println!("\nCudaClaw executor shut down successfully");

    Ok(())
}

fn run_latency_tests(executor: &mut CudaClawExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Latency Tests ===");
    println!("Testing round-trip latency between host and GPU kernel...\n");

    const NUM_ITERATIONS: usize = 100;
    let mut latencies = Vec::with_capacity(NUM_ITERATIONS);

    // Warmup
    for _ in 0..10 {
        let _ = executor.execute_no_op();
    }

    // Actual latency test
    for i in 0..NUM_ITERATIONS {
        let latency = executor.execute_no_op()?;
        latencies.push(latency);

        if (i + 1) % 20 == 0 {
            println!("  Completed {} iterations", i + 1);
        }
    }

    // Calculate statistics
    latencies.sort();
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let sum: Duration = latencies.iter().sum();
    let avg = sum / NUM_ITERATIONS as u32;
    let median = latencies[NUM_ITERATIONS / 2];
    let p95 = latencies[(NUM_ITERATIONS * 95) / 100];
    let p99 = latencies[(NUM_ITERATIONS * 99) / 100];

    println!("\nLatency Statistics ({} iterations):", NUM_ITERATIONS);
    println!("  Min:     {:8.2?}", min);
    println!("  Max:     {:8.2?}", max);
    println!("  Average: {:8.2?}", avg);
    println!("  Median:  {:8.2?}", median);
    println!("  P95:     {:8.2?}", p95);
    println!("  P99:     {:8.2?}", p99);

    Ok(())
}

fn run_functional_tests(executor: &mut CudaClawExecutor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Functional Tests ===");

    // Test Add command
    println!("\nTesting Add command:");
    let test_cases = vec![
        (1.0, 2.0, 3.0),
        (-5.0, 3.0, -2.0),
        (0.0, 0.0, 0.0),
        (3.14159, 2.71828, 5.85987),
        (1000.0, 2000.0, 3000.0),
    ];

    for (a, b, expected) in test_cases {
        let (latency, result) = executor.execute_add(a, b)?;
        let epsilon = 0.0001;
        let success = (result - expected).abs() < epsilon;
        println!("  {:.5} + {:.5} = {:.5} [{} - {:?}]",
            a, b, result,
            if success { "OK" } else { "FAIL" },
            latency
        );
    }

    // Get statistics
    let stats = executor.get_stats();
    println!("\nQueue Statistics:");
    println!("  Commands processed: {}", stats.commands_processed);
    println!("  Total cycles:       {}", stats.total_cycles);
    println!("  Queue head:         {}", stats.head);
    println!("  Queue tail:         {}", stats.tail);
    println!("  Current status:     {:?}", stats.status);

    Ok(())
}

fn run_agent_dispatcher_demo(
    command_queue: Arc<Mutex<cust::memory::UnifiedBuffer<cuda_claw::CommandQueueHost>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== AgentDispatcher Demo ===");
    println!("Demonstrating SuperInstance agent management...\n");

    // Create AgentDispatcher with a 100x100 spreadsheet
    const GRID_ROWS: u32 = 100;
    const GRID_COLS: u32 = 100;
    const NODE_ID: u32 = 1;

    println!("Creating AgentDispatcher with {}x{} spreadsheet grid", GRID_ROWS, GRID_COLS);
    let mut dispatcher = AgentDispatcher::new(GRID_ROWS, GRID_COLS, command_queue, NODE_ID)?;

    // Register various SuperInstance agents
    println!("\nRegistering SuperInstance agents...");

    let claw_agent = SuperInstance::new("claw_001".to_string(), AgentType::Claw)
        .with_model("deepseek-chat".to_string())
        .with_equipment(vec!["MEMORY".to_string(), "REASONING".to_string()]);

    let smpclaw_agent = SuperInstance::new("smpclaw_001".to_string(), AgentType::SMPclaw)
        .with_model("deepseek-coder".to_string())
        .with_equipment(vec!["MEMORY".to_string(), "SPREADSHEET".to_string()]);

    let bot_agent = SuperInstance::new("bot_001".to_string(), AgentType::Bot);

    dispatcher.register_agent(claw_agent)?;
    dispatcher.register_agent(smpclaw_agent)?;
    dispatcher.register_agent(bot_agent)?;

    println!("  Registered {} agents", dispatcher.agents.len());

    // List registered agents
    println!("\nRegistered Agents:");
    for agent in dispatcher.list_agents() {
        println!("  - {}: {:?} (model: {:?}, equipment: {} items)",
            agent.id,
            agent.agent_type,
            agent.model,
            agent.equipment.len()
        );
    }

    // Demonstrate cell operations
    println!("\n=== Cell Operations ===");

    // Set some initial values
    println!("\nSetting cell values...");
    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(0, 0),
        value: 42.0,
        timestamp: 1,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(0, 1),
        value: 3.14159,
        timestamp: 2,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    let set_result = dispatcher.dispatch_agent_op(AgentOperation::SetCell {
        cell: CellRef::new(1, 0),
        value: 2.71828,
        timestamp: 3,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&set_result)?);

    // Get cell values
    println!("\nGetting cell values...");
    let get_result = dispatcher.dispatch_agent_op(AgentOperation::GetCell {
        cell: CellRef::new(0, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&get_result)?);

    // Add cells
    println!("\nAdding cells...");
    let add_result = dispatcher.dispatch_agent_op(AgentOperation::AddCells {
        a: CellRef::new(0, 0),
        b: CellRef::new(0, 1),
        result: CellRef::new(2, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&add_result)?);

    // Multiply cells
    println!("\nMultiplying cells...");
    let mul_result = dispatcher.dispatch_agent_op(AgentOperation::MultiplyCells {
        a: CellRef::new(0, 0),
        b: CellRef::new(1, 0),
        result: CellRef::new(3, 0),
    })?;
    println!("  {}", serde_json::to_string_pretty(&mul_result)?);

    // Apply formula
    println!("\nApplying formula (sum of range)...");
    let formula_result = dispatcher.dispatch_agent_op(AgentOperation::ApplyFormula {
        inputs: vec![
            CellRef::new(0, 0),
            CellRef::new(0, 1),
            CellRef::new(1, 0),
        ],
        output: CellRef::new(4, 0),
        formula: "SUM".to_string(),
    })?;
    println!("  {}", serde_json::to_string_pretty(&formula_result)?);

    // Batch update
    println!("\nBatch updating cells...");
    let batch_updates = vec![
        (CellRef::new(5, 0), 1.0),
        (CellRef::new(5, 1), 2.0),
        (CellRef::new(5, 2), 3.0),
        (CellRef::new(5, 3), 4.0),
        (CellRef::new(5, 4), 5.0),
    ];
    let batch_result = dispatcher.dispatch_agent_op(AgentOperation::BatchUpdate {
        updates: batch_updates.clone(),
        timestamp: 100,
        node_id: NODE_ID,
    })?;
    println!("  {}", serde_json::to_string_pretty(&batch_result)?);

    // Agent operation
    println!("\nDispatching agent operation...");
    let agent_result = dispatcher.dispatch_agent_op(AgentOperation::AgentOp {
        agent_id: "claw_001".to_string(),
        operation: "add".to_string(),
        params: serde_json::json!({"a": 10.0, "b": 20.0}),
    })?;
    println!("  {}", serde_json::to_string_pretty(&agent_result)?);

    // Demonstrate JSON command parsing
    println!("\n=== JSON Command Parsing ===");

    let json_cmd = r#"{
        "op": "SetCell",
        "cell": {"row": 10, "col": 10},
        "value": 99.99,
        "timestamp": 200,
        "node_id": 1
    }"#;

    println!("\nParsing JSON command:");
    println!("  {}", json_cmd);

    let parsed_op = agent::parse_command(json_cmd)?;
    let result = dispatcher.dispatch_agent_op(parsed_op)?;
    println!("  Result: {}", serde_json::to_string_pretty(&result)?);

    // Command validation demo
    println!("\n=== Command Validation ===");

    // Invalid command (out of bounds)
    let invalid_cmd = AgentOperation::SetCell {
        cell: CellRef::new(999999, 999999),
        value: 1.0,
        timestamp: 1,
        node_id: 1,
    };

    match invalid_cmd.validate() {
        Ok(_) => println!("  Command validated (unexpected)"),
        Err(e) => println!("  Validation error (expected): {}", e),
    }

    // Get dispatcher statistics
    println!("\n=== Dispatcher Statistics ===");
    let stats = dispatcher.get_stats();
    println!("  Total agents: {}", stats.total_agents);
    println!("  Grid size: {} x {} ({} cells)", stats.grid_size.0, stats.grid_size.1, stats.total_cells);
    println!("  Current timestamp: {}", stats.current_timestamp);
    println!("  Node ID: {}", stats.node_id);
    println!("  Agents by type:");
    for (agent_type, count) in stats.agents_by_type {
        println!("    - {:?}: {}", agent_type, count);
    }

    println!("\n=== AgentDispatcher Demo Complete ===");

    Ok(())
}
