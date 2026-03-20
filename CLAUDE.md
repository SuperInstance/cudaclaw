# CudaClaw — Claude Code Instructions

## What This Project Is

CudaClaw is a GPU-accelerated agent orchestrator written in Rust + CUDA. The Rust host manages command dispatch and monitoring; CUDA kernels run persistently on the GPU to process commands at warp speed. A SmartCRDT engine handles conflict-free state sync across distributed agents.

**Target**: 10,000+ concurrent agents, <10ms latency, 400K ops/s.

---

## Build Commands

```bash
# CPU-only build (no GPU needed)
cargo build

# Full GPU build (requires nvcc + CUDA 11.0+ in PATH)
cargo build --features cuda

# With NVML GPU metrics
cargo build --features cuda,gpu-metrics

# Release
cargo build --release --features cuda
```

**Run tests:**
```bash
cargo test                             # CPU tests only
cargo test --features cuda             # GPU tests (requires NVIDIA GPU)
cargo test -- --nocapture              # verbose
cargo test --test integration_test     # specific suite
```

**Run benchmarks:**
```bash
cargo run --release -- benchmark       # P99 cell-edit latency
ncu --set full ./target/release/cudaclaw    # NSight Compute profile
nsys profile --stats=true ./target/release/cudaclaw  # NSight Systems
```

---

## Critical Known Bugs (Fix These First)

These are pre-existing issues in `kernels/crdt_engine.cuh` that block GPU compilation:

### Bug 1 — Duplicate kernel definitions (~lines 1400-1638)
Four kernels are defined twice:
- `crdt_warp_process_commands_kernel`
- `crdt_warp_batch_update_kernel`
- `crdt_warp_recalculate_kernel`
- `crdt_warp_persistent_worker_kernel`

The block at lines 1400-1638 is identical to lines 1012-1230. **Delete the duplicate block.**

### Bug 2 — Statistics kernel scalar-as-array (~line 1309)
`crdt_collect_statistics_kernel` uses `block_max[tid + stride]` where `block_max` is a `__shared__ double` (scalar, not array). Fix: declare `__shared__ double block_max_arr[256]` and use that for the reduction.

### Bug 3 — lock_free_queue.cuh stale field references
`pop_command()` references `queue->commands[index]` and `queue->commands_popped` — these fields don't exist. The correct field is `queue->buffer[index]`. This file is not actively used; either update the field names or delete the file.

### Bug 4 — VolatileDispatcher disabled (src/main.rs ~line 790)
The `VolatileDispatcher` and `RoundTripBenchmark` are commented out because `cust` 0.3 requires `UnifiedBuffer` to be wrapped in `Arc<Mutex<>>`. Re-enable after wrapping.

### Bug 5 — smartcrdt_trigger_recalc disabled
The formula recalculation trigger is intentionally disabled pending a real formula engine. Leave this until the formula engine is validated end-to-end.

---

## Project Structure

```
cudaclaw/
├── Cargo.toml                  # Features: cuda, gpu-metrics
├── build.rs                    # Compiles CUDA kernels to PTX at build time
├── kernels/                    # CUDA C++ — the GPU side
│   ├── crdt_engine.cuh         # PRIMARY: 3,366-line SmartCRDT engine (4 sections)
│   ├── executor.cu             # Persistent worker kernel entry point
│   ├── shared_types.h          # Structs shared between Rust and CUDA (Command, CommandQueue)
│   ├── smartcrdt.cuh           # RGA CRDT with atomic ops
│   ├── smart_crdt.cuh          # Helper functions
│   └── lock_free_queue.cuh     # Device-side queue (stale — see Bug 3)
├── src/                        # Rust — the CPU host side
│   ├── main.rs                 # CLI entry point + all subcommands (~2186 lines)
│   ├── cuda_claw.rs            # CudaClawExecutor: kernel launch, mem alloc
│   ├── cuda_claw/ptx.rs        # PTX module loading via cust
│   ├── dispatcher.rs           # GpuDispatcher: priority batch dispatch
│   ├── bridge.rs               # UnifiedMemoryBridge: zero-copy CPU-GPU memory
│   ├── monitor.rs              # SystemMonitor: health, watchdog, stats
│   ├── lock_free_queue.rs      # Host-side lock-free SPSC ring buffer
│   ├── volatile_dispatcher.rs  # VolatileDispatcher (currently disabled)
│   ├── alignment.rs            # Runtime Rust/CUDA struct layout verification
│   ├── agent.rs                # AgentDispatcher: SuperInstance agent management
│   ├── gpu_metrics.rs          # NVML telemetry with graceful fallback
│   ├── ramify/                 # NVRTC JIT kernel compiler
│   │   ├── nvrtc_compiler.rs
│   │   ├── ptx_branching.rs
│   │   ├── resource_exhaustion.rs
│   │   └── shared_memory_bridge.rs
│   ├── constraint_theory/      # Constraint-theory DNA + geometric twin system
│   ├── gpu_cell_agent/         # GPU cell agent + muscle fiber
│   ├── ml_feedback/            # DNA mutation loop (execution log → mutator)
│   ├── installer/              # LLM-driven hardware profiler and role assigner
│   ├── spreadsheet_bridge.rs   # Spreadsheet cells → GPU agent mapping
│   ├── dna.rs                  # RamifiedRole DNA manager
│   └── runtime.rs              # DNA-driven NVRTC runtime
└── tests/
    ├── alignment_test.rs       # Verifies Rust/CUDA struct sizes match
    ├── integration_test.rs     # End-to-end command dispatch
    ├── latency_test.rs         # RTT and phase-decomposition benchmarks
    └── p99_cell_edit_test.rs   # 1M cell edit P99 latency
```

---

## Key Data Structures

All structs shared between Rust and CUDA live in `kernels/shared_types.h`. The Rust mirrors use `#[repr(C, packed(4))]`. Size mismatches will cause silent data corruption — always verify with `static_assert` (CUDA) and `alignment_test.rs` (Rust).

| Struct | Size | Purpose |
|--------|------|---------|
| `Command` | 48B | A single queue entry (cmd_type, id, timestamp, data_a/b, result) |
| `CommandQueue` | 49,192B | 1024-slot ring buffer in Unified Memory |
| `CRDTCell` | 32B | Cell state: value (f64) + Lamport timestamp + node_id + state enum |
| `FormulaCell` | 128B | Formula with up to 6 dependency cell indices + operands |
| `ActiveWorkingSet` | ~37KB | Shared memory L1 cache per block (max 1024 cells) |

---

## crdt_engine.cuh Map

The file has 4 sections. Know where things are before editing:

| Lines | Section | Key Contents |
|-------|---------|--------------|
| 1–1933 | Core Engine | CRDTCell, atomic ops, warp primitives, 4 primary kernels |
| 1935–2297 | Section 1: Warp-Aggregated Merge | PendingUpdate, bitonic sort, `crdt_warp_merge_kernel` |
| 2298–2909 | Section 2: Dependency-Graph Parallelizer | FormulaOp (12 ops), topological sort, `crdt_parallel_recalc_with_deps_kernel` |
| 2908–3366 | Section 3: Shared Memory Working Set | CacheStrategy, ActiveWorkingSet, `crdt_smart_recalc_kernel` |

---

## Development Conventions

### Memory / Alignment
- Shared structs: `#[repr(C, packed(4))]` in Rust, `#pragma pack(push, 4)` in CUDA
- GPU cache-critical structs: `__align__(32)`
- Always add `static_assert(sizeof(Foo) == N, ...)` for any new shared struct
- Run `alignment_test.rs` after any struct change

### Warp Patterns
- Broadcast from lane 0: `__shfl_sync(WARP_MASK, val, 0)`
- Count successes: `__ballot_sync(WARP_MASK, cond)` + `__popc()`
- Intra-warp sync: `__syncwarp()`
- Require `__CUDA_ARCH__ >= 700` guards for Pascal-specific intrinsics

### Kernel Naming
- Kernels: `crdt_<descriptor>_kernel`
- Device functions: `crdt_<action>` or `warp_<action>`
- Shared memory helpers: `load_*`, `store_*`, `recalc_*`

### Thread Layouts
- Persistent kernel: `<<<1, 256>>>` — thread 0 polls queue, warps 1-7 execute
- Warp kernels: `<<<num_warps, 32>>>`
- Max shared memory budget: ~37KB per block (hardware limit 48KB)

### Hard Limits
- Max formula deps per cell: 6 (`FormulaCell.deps[6]`)
- Max working set cells: 1024 (`AWS_MAX_CELLS`)
- Max DepGraph cells per block: 256 (shared memory constraint)
- CommandQueue capacity: 1024 slots

---

## Adding New Formula Operations

1. Add enum value in `FormulaOp` (`crdt_engine.cuh` ~line 2320)
2. Write evaluator: `__device__ __forceinline__ double eval_foo(const double* ops, uint32_t n)`
3. Add case in `evaluate_formula()` switch (~line 2500)
4. Verify `FormulaCell` `static_assert` still passes

## Adding New Cache Strategies

1. Add enum value in `CacheStrategy` (~line 2936)
2. Add case in `load_working_set()` switch (~line 3001)
3. Ensure total loaded cells ≤ `AWS_MAX_CELLS` (1024)

---

## MVP Checklist

- [ ] Fix Bug 1: Remove duplicate kernel block in `crdt_engine.cuh` (~lines 1400-1638)
- [ ] Fix Bug 2: Fix `block_max` scalar-as-array in statistics kernel (~line 1309)
- [ ] Fix Bug 3: Update or remove stale field refs in `lock_free_queue.cuh`
- [ ] Fix Bug 4: Re-enable `VolatileDispatcher` with `Arc<Mutex<UnifiedBuffer>>`
- [ ] Validate: `cargo build --features cuda` completes on real GPU hardware
- [ ] Validate: `cargo test --features cuda` passes all suites
- [ ] Re-enable `smartcrdt_trigger_recalc` and validate formula engine end-to-end
- [ ] Run 10K agent throughput test, confirm 400K ops/s target
- [ ] Profile with NSight Compute; identify any bottlenecks

---

## Environment Variables

```bash
CUDA_PATH=/usr/local/cuda          # Override CUDA toolkit location
# Linux
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

For `.env`-based config (used by installer LLM calls):
```
OPENAI_API_KEY=sk-...
```
