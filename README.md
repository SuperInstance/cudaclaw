# CudaClaw - GPU-Accelerated Agent Orchestrator

**High-performance cellular agent orchestration using CUDA + Rust.** Sub-10ms latency for 10,000+ concurrent agents through CUDA persistent kernels and warp-level parallelism.

## Brand Line

> 10,000 agents at 400K ops/s — warp-level consensus for fleet-scale coordination.

## Installation

```bash
git clone https://github.com/SuperInstance/cudaclaw.git
cd cudaclaw
cargo build --release
```

## Quick Start

```rust
use cudaclaw::{CudaClawExecutor, KernelVariant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut executor = CudaClawExecutor::with_variant(
        KernelVariant::PersistentWorker
    )?;

    executor.init_queue()?;
    executor.start()?;

    // Execute commands on GPU
    executor.execute_add(10.0, 20.0)?;
    executor.execute_multiply(5.0, 6.0)?;

    let stats = executor.get_worker_stats()?;
    println!("Commands processed: {}", stats.commands_processed);

    executor.shutdown()?;
    Ok(())
}
```

## Why GPU Acceleration?

| Approach | Max Agents | Latency | Throughput |
|----------|------------|---------|------------|
| **CPU Single-Threaded** | ~100 | 1-5ms | 10K ops/s |
| **CPU Multi-Threaded** | ~1,000 | 5-20ms | 50K ops/s |
| **CudaClaw GPU** | **10,000+** | **<10ms** | **400K ops/s** |

## Architecture

- **Rust Host** — Safe, high-level command dispatch and monitoring
- **CUDA Kernels** — Persistent GPU workers with warp-level parallelism
- **SmartCRDT** — Distributed state synchronization with Lamport timestamps
- **Lock-Free Queues** — Zero-copy CPU-GPU communication via Unified Memory

## Fleet Context

Part of the Cocapn fleet. Related repos:
- [bordercollie](https://github.com/SuperInstance/bordercollie) — Fleet task herding and orchestration
- [agentic-compiler](https://github.com/SuperInstance/agentic-compiler) — Markdown-to-runtime compilation
- [ai-character-sdk](https://github.com/SuperInstance/ai-character-sdk) — AI character SDK with memory
- [crab-traps](https://github.com/SuperInstance/crab-traps) — Lure collection for fleet learning

---
🦐 Cocapn fleet — lighthouse keeper architecture