# CudaClaw Developer Notes

Build instructions, known issues, and development guidance.

---

## Prerequisites

- Rust (latest stable)
- CUDA Toolkit (11.0+ for `__CUDA_ARCH__ >= 700` features)
- NVIDIA GPU with compute capability 7.0+ (Pascal or newer recommended)
- `nvcc` in PATH for CUDA kernel compilation

## Build & Run

```bash
# Build (build.rs compiles CUDA kernels automatically)
cargo build

# Run
cargo run

# Manual kernel compilation
compile_cuda.bat

# Run tests
cargo test
```

## Project Structure

```
cudaclaw/
├── Cargo.toml              # Dependencies: cust 0.3, tokio 1.42, serde 1.0
├── build.rs                # CUDA kernel compilation at build time
├── src/
│   ├── main.rs             # Entry point
│   ├── cuda_claw.rs        # GPU orchestrator (kernel launch, mem alloc)
│   ├── dispatcher.rs       # Command dispatch with priority ordering
│   ├── bridge.rs           # Rust/CUDA unified memory bridge
│   ├── agent.rs            # Agent management
│   ├── alignment.rs        # Runtime memory layout verification
│   ├── lock_free_queue.rs # Rust-side lock-free SPSC queue
│   ├── monitor.rs          # Kernel health monitoring
│   ├── volatile_dispatcher.rs  # Volatile write dispatcher
│   └── cuda_claw/
│       └── ptx.rs          # PTX module loading via cust
├── kernels/
│   ├── main.cu             # Kernel entry points
│   ├── shared_types.h      # Shared Rust/CUDA type definitions (Command, CommandQueue)
│   ├── crdt_engine.cuh     # SmartCRDT engine (3,366 lines) - PRIMARY FILE
│   ├── executor.cu         # Persistent worker kernel
│   ├── smartcrdt.cuh       # RGA CRDT with atomic operations
│   ├── smart_crdt.cuh      # Smart CRDT helper functions
│   └── lock_free_queue.cuh # Lock-free queue device functions
└── tests/
    ├── alignment_test.rs   # Memory alignment verification
    ├── integration_test.rs # End-to-end integration tests
    └── latency_test.rs     # Latency benchmarking
```

---

## Key Data Structures

| Structure | Size | File | Purpose |
|-----------|------|------|---------|
| `Command` | 48B | shared_types.h | Queue command (packed) |
| `CommandQueue` | 49,192B | shared_types.h | 1024-slot ring buffer |
| `CRDTCell` | 32B | crdt_engine.cuh | Cell value + metadata |
| `CRDTState` | ~200B | crdt_engine.cuh | Grid state + stats |
| `WarpCommand` | 16B | crdt_engine.cuh | Broadcast command |
| `PendingUpdate` | 32B | crdt_engine.cuh | Per-lane update request |
| `FormulaCell` | 128B | crdt_engine.cuh | Formula with 6 deps |
| `ActiveWorkingSet` | ~37KB | crdt_engine.cuh | Shared memory cache |

## crdt_engine.cuh Sections

The file is organized into 4 major sections:

1. **Lines 1-1933: Core Engine**
   - `CRDTCell` (32B), `CRDTState`, `WarpCommand`, `CellState`
   - Atomic operations: `atomic_cas_cell_64`, `compare_timestamps`
   - Device functions: `crdt_read_cell`, `crdt_write_cell`, `crdt_delete_cell`, `crdt_merge_conflict`
   - Warp primitives: `warp_broadcast_command`, `warp_process_command`, `warp_parallel_update_32`
   - Kernels: `crdt_warp_process_commands_kernel`, `crdt_warp_batch_update_kernel`, `crdt_warp_recalculate_kernel`, `crdt_warp_persistent_worker_kernel`

2. **Lines 1935-2297: Section 1 - Warp-Aggregated Merge**
   - `PendingUpdate` struct, bitonic sort, deduplication
   - `warp_aggregate_updates_shmem()`, `warp_resolve_conflicts()`
   - `crdt_warp_merge_kernel`

3. **Lines 2298-2909: Section 2 - Dependency-Graph Parallelizer**
   - `FormulaOp` enum (12 ops), `FormulaCell` (128B), `DepGraph`
   - 12 evaluators: `eval_add/sub/mul/div/sum/min/max/if/count/avg/abs/power`
   - `prefix_sum_scan()` (Kogge-Stone), `compact_array()`
   - `assign_topological_levels()`, `evaluate_level()`
   - `crdt_parallel_recalc_with_deps_kernel`

4. **Lines 2908-3366: Section 3 - Shared Memory Working Set**
   - `CacheStrategy` enum, `ActiveWorkingSet`
   - `load_working_set()`, `store_working_set()`, `recalc_in_shared_mem()`
   - `working_set_recalc_kernel`, `crdt_smart_recalc_kernel` (unified)

---

## Known Issues

### Pre-existing (not yet fixed)

1. **Duplicate kernel definitions** (~lines 1400-1638)
   Four kernels are defined twice: `crdt_warp_process_commands_kernel`, `crdt_warp_batch_update_kernel`, `crdt_warp_recalculate_kernel`, `crdt_warp_persistent_worker_kernel`. The duplicates at lines 1400-1638 are identical to the originals at lines 1012-1230. This causes compilation errors on strict compilers. Remove the duplicates.

2. **Statistics kernel scalar-as-array bug** (~line 1309)
   `crdt_collect_statistics_kernel` uses `block_max[tid + stride]` where `block_max` is a `__shared__ double` scalar, not an array. This will cause compilation errors. Fix: use separate `__shared__ double block_max_arr[]` for reduction.

3. **lock_free_queue.cuh references nonexistent fields**
   `pop_command()` references `queue->commands[index]` and `queue->commands_popped` which don't exist in `CommandQueue` (the field is `queue->buffer[index]` and there's no `commands_popped`). This file appears to be from an older version and is not actively used.

### Fixed in current session

4. **FormulaCell static_assert** - Was claiming 32 bytes, fixed to 128 bytes (correct with `__align__(32)`)
5. **compact_array nullptr crash** - Removed dead code block that called `compact_array(nullptr, ...)`
6. **assign_topological_levels bounds** - Added guard for `graph->num_cells > 256` (shared memory limit)

---

## Development Conventions

### Memory Alignment
- All Rust/CUDA shared structs use `#[repr(C, packed(4))]` or `#pragma pack(push, 4)`
- `static_assert` on all CUDA struct sizes
- `__align__(32)` on cache-line-critical GPU structures
- Runtime verification via `alignment.rs`

### Warp-Level Patterns
- Use `__shfl_sync(WARP_MASK, val, 0)` for lane 0 broadcasts (Pascal+)
- Use `__ballot_sync(WARP_MASK, success)` + `__popc()` for success counting
- Use `__syncwarp()` for intra-warp synchronization
- Provide shared memory fallbacks for pre-Pascal (`#if __CUDA_ARCH__ >= 700`)

### Kernel Naming
- Kernels: `crdt_<descriptive>_kernel`
- Device functions: `crdt_<action>` or `warp_<action>`
- Shared memory functions: `load_*`, `store_*`, `recalc_*`

### Thread Organization
- Warp-level kernels: `<<<num_warps, 32>>>` (one warp per command/group)
- Block kernels: `<<<blocks, threads, shared_mem>>>`
- Persistent kernel: `<<<1, 256>>>` (thread 0 manages queue)

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Command latency | < 5 us | Dispatch to execution |
| Warp cell updates | 32/cycle | One per lane |
| Shared memory read | ~20 cycles | vs ~400 VRAM |
| Queue poll overhead | ~2-5 ns | Atomic relaxed |
| Throughput | 100K-400K ops/s | Persistent kernel |

---

## Testing

```bash
# All tests
cargo test

# Specific test
cargo test --test integration_test

# Run with output
cargo test -- --nocapture
```

Test files:
- `tests/alignment_test.rs` - Verifies Rust/CUDA memory layout matches
- `tests/integration_test.rs` - End-to-end command submission
- `tests/latency_test.rs` - Measures dispatch-to-execution latency

---

## Adding New Formula Operations

To add a new formula op (e.g., `FOP_MODULO`):

1. Add enum value in `FormulaOp` (crdt_engine.cuh, ~line 2320)
2. Add evaluator function: `__device__ __forceinline__ double eval_modulo(const double* ops, uint32_t n)`
3. Add case in `evaluate_formula()` switch (~line 2500)
4. If operand count changes, verify `FormulaCell` size and static_assert

## Adding New Cache Strategies

To add a new cache strategy (e.g., `CACHE_COLUMN`):

1. Add enum value in `CacheStrategy` (crdt_engine.cuh, ~line 2936)
2. Add case in `load_working_set()` switch (~line 3001)
3. Ensure total loaded cells stays within `AWS_MAX_CELLS` (1024)

---

## Important Constraints

- **Max formula dependencies**: 6 per cell (hardcoded in `FormulaCell.deps[6]`)
- **Max working set cells**: 1024 per block (`AWS_MAX_CELLS`)
- **Shared memory budget**: ~37KB per block (within 48KB hardware limit)
- **DepGraph max cells per block**: 256 (`WARP_SIZE * 8`, shared memory constraint)
- **CommandQueue capacity**: 1024 commands
- **GPU compute capability**: 7.0+ recommended (Pascal+) for `__nanosleep` and `__shfl_sync`
