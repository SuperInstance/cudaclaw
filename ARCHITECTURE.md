# CudaClaw Architecture

GPU-accelerated SmartCRDT orchestrator using persistent CUDA kernels with warp-level parallelism.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RUST HOST (CPU)                            │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  dispatcher  │  │    bridge    │  │        monitor             │ │
│  │  (submit)    │  │  (unified   │  │  (health, watchdog)       │ │
│  │              │  │   memory)   │  │                           │ │
│  └──────┬──────┘  └──────┬───────┘  └─────────────┬─────────────┘ │
│         │                │                       │                 │
│         │    ┌───────────┴───────────┐           │                 │
│         │    │    CommandQueue       │◄──────────┘                 │
│         └───►│    (Unified Memory)   │                             │
│              │    49,192 bytes       │                             │
│              │    1024-slot ring     │                             │
│              └───────────┬───────────┘                             │
└──────────────────────────┼──────────────────────────────────────────┘
                           │ PCIe (Unified Memory)
┌──────────────────────────┼──────────────────────────────────────────┐
│                         CUDA DEVICE (GPU)                           │
│                          │                                         │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │              persistent_worker kernel                         │   │
│  │              (1 block, 256 threads)                           │   │
│  │              Thread 0: queue poll + dispatch                  │   │
│  │              Threads 1-255: available for parallel work       │   │
│  └───────────────────────┬─────────────────────────────────────┘   │
│                          │                                         │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │              crdt_engine.cuh (3,366 lines)                    │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  CORE: CRDTCell (32B), CRDTState, WarpCommand           │ │   │
│  │  │  - atomicCAS spin loops with exponential backoff        │ │   │
│  │  │  - Lamport timestamp + node_id conflict resolution      │ │   │
│  │  │  - __shfl_sync command broadcast to 32 warp lanes       │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  SECTION 1: Warp-Aggregated Merge                       │ │   │
│  │  │  - PendingUpdate (32B) per lane                         │ │   │
│  │  │  - Bitonic sort deduplication by cell_idx               │ │   │
│  │  │  - 1 CAS per unique target (vs N without aggregation)   │ │   │
│  │  │  - crdt_warp_merge_kernel <<<num_warps, 32>>>          │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  SECTION 2: Dependency-Graph Parallelizer               │ │   │
│  │  │  - FormulaCell (128B): op, deps[6], operands, result   │ │   │
│  │  │  - DepGraph: in_degree[], level[]                       │ │   │
│  │  │  - 12 formula ops: ADD,SUB,MUL,DIV,SUM,MIN,MAX,       │ │   │
│  │  │    IF,COUNT,AVG,ABS,POWER                               │ │   │
│  │  │  - Kogge-Stone prefix sum scan                          │ │   │
│  │  │  - Topological level assignment (data-race-free)        │ │   │
│  │  │  - crdt_parallel_recalc_with_deps_kernel               │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │  SECTION 3: Shared Memory Working Set                   │ │   │
│  │  │  - ActiveWorkingSet: ~37KB shared memory               │ │   │
│  │  │  - 1024 CRDTCells cached in L1 (~20 cycle access)      │ │   │
│  │  │  - Cache strategies: DIRTY_ONLY, DIRTY_AND_DEPS,       │ │   │
│  │  │    FULL_ROW                                             │ │   │
│  │  │  - recalc_in_shared_mem: register-speed evaluation      │ │   │
│  │  │  - crdt_smart_recalc_kernel: unified entry point       │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   smartcrdt.cuh  │  │ lock_free_queue  │  │  shared_types.h  │   │
│  │   (RGA CRDT)     │  │   (device fns)   │  │  (Rust/CUDA)     │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Layout

### CommandQueue (49,192 bytes, Unified Memory)

```
Offset     Field                 Size        Description
──────     ─────                 ────        ────────────
0          buffer[1024]          48,992      Ring buffer of Command (48B each)
48,992     status                4           Queue status
48,996     head                  4           Write index (Rust volatile write)
49,000     tail                  4           Read index (GPU volatile write)
49,004     is_running            1           Kernel running flag
49,008     commands_sent         8           Total commands sent (stat)
49,016     commands_processed    8           Total commands processed (stat)
```

### Command (48 bytes, `#pragma pack(push, 4)`)

```
Offset  Field           Size
──────  ─────           ────
0       cmd_type        4      NOOP(0), EDIT_CELL(1), SYNC_CRDT(2), SHUTDOWN(3)
4       id              4      Command/cell identifier
8       timestamp       8      Lamport timestamp
16      data_a          4      Primary data value
20      data_b          4      Secondary data value
24      result          4      Result value
28      batch_data      8      Batch data pointer
36      batch_count     4      Batch count
40      _padding        4      Alignment padding
44      result_code     4      Result code
```

### CRDTCell (32 bytes, `__align__(32)`)

```
Offset  Field           Size
──────  ─────           ────
0       value           8      Primary cell value (double)
8       timestamp       8      Lamport timestamp
16      node_id         4      Origin node
20      state           4      CellState enum (ACTIVE/DELETED/CONFLICT/MERGED/PENDING/LOCKED)
24      padding[3]      12     Alignment to 32 bytes
```

### PendingUpdate (32 bytes, `__align__(32)`)

```
Offset  Field           Size
──────  ─────           ────
0       cell_idx        4      1D flat index into CRDT grid
4       new_value       8      Proposed value (double)
12      timestamp       8      Lamport timestamp
20      node_id         4      Origin node
24      valid           1      1 if lane has pending update
25      _pad[3]         3      Alignment padding
```

### FormulaCell (128 bytes, `__align__(32)`)

```
Offset  Field           Size
──────  ─────           ────
0       cell_idx        4      Flat index in grid
4       op              1      FormulaOp enum
8       num_deps        4      Number of dependencies (0-6)
12      deps[6]         24     Dependency cell indices
36      operands[6]     48     Cached operand values
84      result          8      Computed result
92      timestamp       8      Recalculation timestamp
100     node_id         4      Node performing recalc
104     dirty           1      Needs recalculation
105     computing       1      Currently being computed
108     _pad[2]         2      Padding to 128
```

### ActiveWorkingSet (shared memory, ~37KB)

```
Offset              Field                   Size
──────              ─────                   ────
0                   cells[1024]            32,768    CRDTCell copies (32KB)
32768               cell_indices[1024]     4,096     Local -> global index map
36864               num_cells              4         Current loaded cell count
36868               num_dirty              4         Originally dirty count
36872               dirty_flags[1024]      1,024     Which cells modified
37896               formula_flags[1024]    1,024     Which are formula cells
38920               _pad[2]                2         Alignment padding
```

---

## Data Flow

### Command Submission (Rust -> GPU)

```
Rust                              GPU
────                              ───
1. Write Command to              4. __threadfence_system()
   queue->buffer[head]              (PCIe visibility)
2. queue->head = new_head       5. Read queue->head
   (volatile write)              6. If head != tail:
3. __threadfence_system()          a. Read queue->buffer[tail]
   (optional but safe)              b. Process command
                                    c. queue->tail = new_tail
                                    d. __threadfence_system()
                                    e. __nanosleep(100) if empty
```

### Smart Recalculation Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Phase 1:        │     │  Phase 2:        │     │  Phase 3:        │
│  Warp-Aggregate  │────►│  Shared Memory   │────►│  Dependency-     │
│  Merge           │     │  Working Set     │     │  Graph Recalc    │
│                  │     │                  │     │                  │
│  • Deduplicate   │     │  • Load dirty    │     │  • Topo sort      │
│    by cell_idx    │     │    cells to L1   │     │  • Level-by-level│
│  • Bitonic sort   │     │  • 3 strategies  │     │  • Parallel eval  │
│  • Single CAS     │     │  • ~20 cycle     │     │  • Write back     │
│    per unique     │     │    access speed  │     │    with atomicCAS │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## Kernel Launch Configurations

| Kernel | Launch | Shared Mem | Description |
|--------|--------|------------|-------------|
| `persistent_worker` | `<<<1, 256>>>` | 0 | Queue polling + command dispatch |
| `crdt_warp_merge_kernel` | `<<<num_warps, 32>>>` | 32*32=1024B | Warp-aggregated conflict resolution |
| `crdt_parallel_recalc_with_deps_kernel` | `<<<blocks, threads>>>` | 2*num_cells | Topological formula recalculation |
| `working_set_recalc_kernel` | `<<<blocks, threads, AWS_SIZE>>>` | ~37KB | L1-cached working set recalc |
| `crdt_smart_recalc_kernel` | `<<<blocks, threads, AWS_SIZE>>>` | ~37KB | Unified pipeline (all 3 subsystems) |
| `crdt_warp_batch_update_kernel` | `<<<num_warps, 32>>>` | 0 | 32 parallel cell updates per warp |
| `crdt_warp_recalculate_kernel` | `<<<num_warps, 32>>>` | 0 | 32 parallel cell recalculations |

---

## Warp Primitives Reference

| Primitive | Purpose | Latency |
|-----------|---------|---------|
| `__shfl_sync(mask, val, srcLane)` | Broadcast value across warp lanes | ~4 cycles |
| `__ballot_sync(mask, predicate)` | Warp vote on predicate (1 bit per lane) | ~4 cycles |
| `__popc(ballot)` | Count set bits in ballot result | ~1 cycle |
| `__syncwarp()` | Synchronize all lanes in warp | ~4-8 cycles |
| `__nanosleep(ns)` | Sleep nanoseconds (Pascal+) | Variable |
| `__threadfence_system()` | PCIe bus memory fence | ~100-200 cycles |

---

## Conflict Resolution

All concurrent updates use **last-write-wins** with two-level priority:

1. **Timestamp comparison**: Higher Lamport timestamp wins
2. **Node ID tiebreaker**: Higher node_id wins when timestamps are equal

```
compare_timestamps(ts1, node1, ts2, node2):
    if ts1 > ts2  -> (ts1, node1) wins
    if ts1 < ts2  -> (ts2, node2) wins
    if ts1 == ts2 -> node1 > node2 ? (ts1,node1) : (ts2,node2)
```

---

## Cache Strategies (Shared Memory Working Set)

| Strategy | Cells Loaded | Use Case |
|----------|-------------|----------|
| `CACHE_DIRTY_ONLY` | Only flagged dirty | Minimal memory, few deps |
| `CACHE_DIRTY_AND_DEPS` | Dirty + formula deps | Typical recalculation |
| `CACHE_FULL_ROW` | Entire rows with dirty | Row-heavy workloads |

---

## CRDT Cell States

```
CELL_ACTIVE(0)   -> Normal, visible cell
CELL_DELETED(1)  -> Tombstoned (logical delete)
CELL_CONFLICT(2)  -> Concurrent updates detected
CELL_MERGED(3)   -> Conflict resolved
CELL_PENDING(4)  -> Update awaiting confirmation
CELL_LOCKED(5)   -> Cell locked for in-progress update
```

---

## File Dependencies

```
shared_types.h
  ├── executor.cu          (CommandQueue, Command)
  ├── crdt_engine.cuh      (CRDTState, CommandQueue)
  ├── lock_free_queue.cuh  (CommandQueue)
  └── smartcrdt.cuh        (CommandQueue)

crdt_engine.cuh (self-contained except for shared_types.h)
  └── CRDTCell, CRDTState, WarpCommand, PendingUpdate, FormulaCell, DepGraph, ActiveWorkingSet
```
