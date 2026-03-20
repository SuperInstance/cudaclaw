# CudaClaw — Onboarding Report

**Date**: 2026-03-20
**Version**: 0.1.0 (Production Alpha)
**Stack**: Rust 1.70+ · CUDA 11.0+ · NVIDIA GPU Compute Capability 7.0+

---

## What Is CudaClaw?

CudaClaw is a GPU-accelerated agent orchestrator. Instead of running thousands of software agents on the CPU, it offloads coordination, state management, and computation to an NVIDIA GPU using CUDA persistent kernels and warp-level parallelism.

The practical pitch: where a CPU multi-threaded system tops out around 1,000 agents at 5-20ms latency, CudaClaw targets **10,000+ concurrent agents at <10ms latency and 400K operations/second**.

Primary use cases it was built for:
- Massively parallel agent systems (e.g., game AI, simulation)
- Real-time multi-agent coordination
- GPU-accelerated spreadsheet engines (10,000+ formula cells recalculating live)
- Constraint-theory spatial queries (KD-tree at 400K ops/s)
- ML feedback loops driving self-optimization

---

## System Architecture (Mental Model)

```
CPU (Rust Host)                          GPU (CUDA Device)
─────────────────────────────────        ─────────────────────────────────────
GpuDispatcher                            persistent_worker kernel
  │  priority-ordered commands           (1 block, 256 threads = 8 warps)
  ▼                                        Warp 0: polls CommandQueue
CommandQueue  ◄──── Unified Memory ───►    Warps 1-7: execute commands
  (49KB ring,                            │
   1024 slots)                           ▼
                                        crdt_engine.cuh
SystemMonitor                             Section 1: Warp-Aggregated Merge
  health checks                          Section 2: Dependency-Graph Parallelizer
  GPU metrics (NVML)                     Section 3: Shared Memory Working Set
```

The key insight is **Unified Memory**: the `CommandQueue` ring buffer lives in memory accessible by both CPU and GPU simultaneously with zero-copy. The CPU writes commands; the persistent GPU kernel spins polling for them and processes in batches.

---

## Technology Decisions

| Decision | Why |
|----------|-----|
| **Persistent kernels** instead of per-call kernel launches | Kernel launch overhead is ~5-10μs. Persistent kernels eliminate this, enabling 2-5μs command latency |
| **Warp-level parallelism** (32-lane SIMT) | 32 agents update simultaneously in a single warp cycle |
| **SmartCRDT** for state sync | Conflict-free Replicated Data Types give eventual consistency without locking. Last-write-wins with Lamport timestamps for deterministic resolution |
| **Unified Memory** for the queue | Zero-copy communication; no explicit cudaMemcpy needed between host and device |
| **Rust host** | Memory safety for the CPU side without GC pauses; `cust` crate provides CUDA FFI |
| **NVRTC JIT** (Ramify engine) | Compile CUDA kernels at runtime from DNA-specified sources, enabling self-modifying agent behavior |

---

## Codebase Tour

### Entry Point

`src/main.rs` (~2,186 lines) — CLI with 10+ subcommands via clap:

```
cudaclaw                   # main GPU demo
cudaclaw dna               # RamifiedRole DNA manager
cudaclaw constraint        # Constraint-theory DNA
cudaclaw agent             # GPU cell agent (demo/status)
cudaclaw feedback          # ML feedback loop
cudaclaw ramify            # NVRTC JIT engine
cudaclaw runtime           # DNA-driven runtime
cudaclaw spreadsheet       # Spreadsheet bridge
cudaclaw install           # LLM hardware profiler
cudaclaw monitor-tree      # ASCII/HTML tree dashboard
cudaclaw benchmark         # P99 latency benchmark
```

### Core GPU Pipeline

| File | Role |
|------|------|
| `src/cuda_claw.rs` | `CudaClawExecutor` — initializes GPU context, loads PTX, launches kernels |
| `src/cuda_claw/ptx.rs` | PTX module loading via `cust` |
| `kernels/executor.cu` | Persistent worker kernel entry point |
| `kernels/crdt_engine.cuh` | The heart of the system — 3,366 lines, 4 sections (see below) |
| `kernels/shared_types.h` | Struct definitions shared between Rust and CUDA |

### Host Support

| File | Role |
|------|------|
| `src/dispatcher.rs` | `GpuDispatcher` — thread-safe, priority-ordered batch submission |
| `src/bridge.rs` | `UnifiedMemoryBridge` — allocates and manages Unified Memory |
| `src/monitor.rs` | `SystemMonitor` — health checks, watchdog, stats collection |
| `src/lock_free_queue.rs` | Host-side lock-free SPSC ring buffer (mirrors the GPU queue) |
| `src/alignment.rs` | Runtime verification that Rust and CUDA struct layouts match |
| `src/gpu_metrics.rs` | NVML telemetry with graceful fallback when no GPU present |
| `src/agent.rs` | `AgentDispatcher` — manages SuperInstance agent types (Claw/Bot/Seed/SMPclaw) |

### Advanced Subsystems

| Directory/File | What It Does |
|----------------|--------------|
| `src/ramify/` | **Ramify engine** — NVRTC JIT compiler. Compiles CUDA kernels at runtime from PTX or C++ source. DNA-driven pipeline. |
| `src/constraint_theory/` | Constraint-theory DNA + `GeometricTwinMap` for spatial topology |
| `src/gpu_cell_agent/` | GPU cell agent + `MuscleFiber` abstraction |
| `src/ml_feedback/` | DNA mutation loop: `ExecutionLog` → `SuccessAnalyzer` → `DnaMutator` |
| `src/installer/` | LLM-driven hardware profiler: probes GPU, calls OpenAI API, generates role profiles |
| `src/spreadsheet_bridge.rs` | Maps spreadsheet cells to GPU agents; handles formula registration |
| `src/dna.rs` | `RamifiedRole` DNA manager |
| `src/runtime.rs` | DNA-driven NVRTC runtime |

---

## crdt_engine.cuh Deep Dive

This is the most important file. Know its structure before touching it.

```
Lines 1–1933        Core Engine
                    ├─ CRDTCell (32B): value f64, timestamp u64, node_id u32, state enum
                    ├─ Atomic ops: atomic_cas_cell_64, compare_timestamps
                    ├─ Device functions: crdt_read_cell, crdt_write_cell, crdt_merge_conflict
                    ├─ Warp primitives: warp_broadcast_command, warp_parallel_update_32
                    └─ 4 kernels: process_commands, batch_update, recalculate, persistent_worker

Lines 1935–2297     Section 1: Warp-Aggregated Merge
                    ├─ PendingUpdate (32B): cell_idx, value, timestamp per lane
                    ├─ Bitonic sort for deduplication by cell_idx
                    ├─ Single atomicCAS per unique target (reduces contention)
                    └─ crdt_warp_merge_kernel

Lines 2298–2909     Section 2: Dependency-Graph Parallelizer
                    ├─ FormulaOp enum (12 ops: ADD SUB MUL DIV SUM MIN MAX IF COUNT AVG ABS POWER)
                    ├─ FormulaCell (128B): formula + 6 dependency cell indices
                    ├─ Kogge-Stone prefix sum scan for topological levels
                    └─ crdt_parallel_recalc_with_deps_kernel

Lines 2908–3366     Section 3: Shared Memory Working Set
                    ├─ CacheStrategy enum: DIRTY_ONLY, DIRTY_AND_DEPS, FULL_ROW
                    ├─ ActiveWorkingSet (~37KB): up to 1024 cells in L1 shared memory
                    └─ crdt_smart_recalc_kernel (unified entry point)
```

---

## Known Bugs (Blocking GPU Compilation)

These are pre-existing. Fix them before trying `cargo build --features cuda`.

### 1. Duplicate kernel definitions in `crdt_engine.cuh` (~lines 1400-1638)
Four kernels appear twice. The block at lines ~1400-1638 is identical to lines ~1012-1230.
**Fix**: Delete the duplicate block.

### 2. Statistics kernel scalar-as-array bug (~line 1309)
`crdt_collect_statistics_kernel` does `block_max[tid + stride]` on a `__shared__ double` (scalar).
**Fix**: Replace with `__shared__ double block_max_arr[256]` and index into that.

### 3. Stale field references in `lock_free_queue.cuh`
`pop_command()` uses `queue->commands[index]` and `queue->commands_popped` which don't exist. Correct field is `queue->buffer[index]`. This file is not actively used.
**Fix**: Update field names to match current `CommandQueue` layout, or remove the file.

### 4. VolatileDispatcher disabled (`src/main.rs` ~line 790)
`cust` 0.3 requires `UnifiedBuffer` wrapped in `Arc<Mutex<>>`. The round-trip benchmark depends on this.
**Fix**: Wrap the buffer and re-enable the dispatcher.

### 5. Formula recalculation disabled
`smartcrdt_trigger_recalc` is intentionally off. Leave until the formula engine is validated.

---

## State of the Project

### What Works (CPU, no GPU required)

All CLI subcommands that don't touch GPU hardware run today:

```bash
cargo run -- dna demo
cargo run -- constraint show-dna
cargo run -- agent demo
cargo run -- feedback demo
cargo run -- ramify demo
cargo run -- runtime demo
cargo run -- spreadsheet demo
cargo run -- monitor-tree demo
cargo run -- install --show-profile
```

All CPU-side tests pass:
```bash
cargo test
```

### What Needs a Real GPU

```bash
cargo build --features cuda        # won't compile until bugs 1-3 are fixed
cargo test --features cuda         # end-to-end GPU tests
cargo run --release -- benchmark   # P99 latency measurement
```

### Feature Completeness

| Subsystem | Status |
|-----------|--------|
| Persistent GPU kernel | Implemented, needs hardware validation |
| SmartCRDT engine | Implemented, 3 compile bugs block GPU build |
| Lock-free queue (CPU side) | Complete |
| Lock-free queue (GPU device) | Stale (Bug 3) |
| Warp-aggregated merge | Implemented |
| Dependency-graph parallelizer | Implemented |
| Shared memory working set | Implemented |
| Formula engine trigger | Disabled (Bug 5) |
| Ramify NVRTC JIT | Implemented, some paths simulated |
| ML feedback loop | Implemented |
| Constraint theory | Implemented |
| Spreadsheet bridge | Implemented |
| LLM installer | Implemented (needs OPENAI_API_KEY) |
| GPU metrics (NVML) | Implemented with fallback |
| VolatileDispatcher | Disabled (Bug 4) |
| Multi-node CRDT merge | Framework exists, untested end-to-end |

---

## MVP Checklist

In priority order:

- [ ] **Bug 1**: Remove duplicate kernel block in `crdt_engine.cuh` lines ~1400-1638
- [ ] **Bug 2**: Fix `block_max` scalar-as-array in `crdt_collect_statistics_kernel`
- [ ] **Bug 3**: Fix or remove stale `lock_free_queue.cuh` field refs
- [ ] **Validate**: `cargo build --features cuda` succeeds
- [ ] **Validate**: `cargo test --features cuda` all green
- [ ] **Bug 4**: Re-enable `VolatileDispatcher` with `Arc<Mutex<>>` wrapping
- [ ] **Validate**: Round-trip benchmark produces latency numbers
- [ ] **Bug 5**: Re-enable `smartcrdt_trigger_recalc`, validate formula recalc end-to-end
- [ ] **Scale test**: 10,000 agent batch, confirm 400K ops/s throughput
- [ ] **Profile**: NSight Compute, identify bottlenecks

---

## Dependency Map

```
Cargo.toml
├── cust 0.3          (optional, --features cuda)   CUDA FFI
├── tokio 1.42        (full)                        async runtime
├── serde 1.0         (derive)                      JSON serialization
├── serde_json 1.0
├── nvml-wrapper 0.10 (optional, --features gpu-metrics)  GPU telemetry
├── rand 0.8                                        test data
├── reqwest 0.11      (json)                        LLM HTTP calls (installer)
├── tempfile 3.9                                    test temp dirs
├── dotenv 0.15                                     .env loading
└── openai-api-rs 0.1                               OpenAI API (installer)
```

---

## First Session Checklist

If you're picking this up on a machine with an NVIDIA GPU:

1. **Check GPU**: `nvidia-smi` — confirm compute capability ≥ 7.0
2. **Check CUDA**: `nvcc --version` — need 11.0+
3. **Check Rust**: `rustup show` — need 1.70+
4. **CPU smoke test**: `cargo test` — should all pass
5. **Fix Bug 1** (duplicate kernels in `crdt_engine.cuh`)
6. **Fix Bug 2** (statistics kernel)
7. **Fix Bug 3** (lock_free_queue.cuh)
8. **GPU build**: `cargo build --features cuda`
9. **GPU tests**: `cargo test --features cuda`
10. **Benchmark**: `cargo run --release -- benchmark`

---

## Where to Find Things

| "I need to..." | Go to |
|----------------|-------|
| Change how commands are submitted to the GPU | `src/dispatcher.rs` |
| Change persistent kernel behavior | `kernels/executor.cu` + `kernels/crdt_engine.cuh` lines 1-1933 |
| Add a new formula operation | `crdt_engine.cuh` Section 2 (~line 2320+) |
| Add a new cache strategy | `crdt_engine.cuh` Section 3 (~line 2936+) |
| Change shared CPU/GPU struct layouts | `kernels/shared_types.h` + `src/alignment.rs` |
| Add a CLI subcommand | `src/main.rs` |
| Change agent types or operations | `src/agent.rs` |
| Modify JIT kernel compilation | `src/ramify/nvrtc_compiler.rs` |
| Change the ML feedback loop | `src/ml_feedback/` |
| Change the installer/profiler | `src/installer/` |
| Tune GPU metrics collection | `src/gpu_metrics.rs` |

---

## Performance Reference

| Kernel Variant | Latency | Throughput | Best For |
|----------------|---------|------------|----------|
| Adaptive | 5-10μs | 10K ops/s | Balanced workloads |
| Spin | 1-2μs | 50K ops/s | Lowest latency |
| Timed | 50-100μs | 5K ops/s | Power saving |
| PersistentWorker | 2-5μs | 100K ops/s | High throughput |
| MultiBlockWorker | 5-10μs | 400K ops/s | Maximum throughput |

Batch sweet spot: 4-16 commands per dispatch (2-4x speedup over single dispatch).

Hardware limits to keep in mind:
- Shared memory per block: 37KB usable (48KB hardware)
- Max working set cells: 1024
- Max formula dependencies: 6 per cell
- Queue depth: 1024 commands
- Minimum GPU: Compute Capability 7.0 (Volta/Turing/Ampere)
