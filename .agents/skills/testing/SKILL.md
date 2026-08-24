# Testing Cudaclaw

## Build & Test Without CUDA

The `cust` crate is optional behind the `cuda` Cargo feature. All Rust code compiles and tests pass without a CUDA toolkit:

```bash
# Full test suite (280 tests, ~30s)
cargo test

# Specific test suites
cargo test p99_cell_edit       # 1M-edit P99 benchmark (4 tests)
cargo test --test latency      # RTT latency suite (8 tests)
cargo test --test latency_test # Polling/coalescing tests (8 tests, 1 ignored)
cargo test --test alignment_test # Struct layout verification (5 tests)
cargo test dispatcher          # Lock-free dispatcher (17 tests)
cargo test lock_free_queue     # Queue operations (4 tests via test_queue filter)
```

## Build & Test With CUDA

Requires CUDA toolkit (nvcc) installed:

```bash
cargo check --features cuda
cargo test --features cuda --release -- --nocapture
```

## CLI Subcommands (all work without GPU)

All subcommands are routed before CUDA initialization:

```bash
cargo run -- dna --demo              # DNA schema demo
cargo run -- dna --probe             # Hardware micro-benchmarks
cargo run -- spreadsheet --demo      # Cell-to-root bridge demo
cargo run -- constraint --demo       # Constraint-Theory demo
cargo run -- agent --demo            # GPU cell agents demo
cargo run -- feedback --demo         # ML feedback loop demo
cargo run -- ramify --demo           # Ramify engine demo
cargo run -- runtime --demo          # NVRTC runtime demo
cargo run -- monitor-tree --demo     # Tree dashboard demo
cargo run -- install --role spreadsheet_engine --heuristic-only  # Installer
```

## Key Test Files

- `tests/p99_cell_edit_test.rs` — Self-contained 1M-edit benchmark, writes `latency_report.json`
- `tests/latency.rs` — RTT benchmarking with 4-phase decomposition, writes `rtt_latency_report.json`
- `tests/latency_test.rs` — Polling strategies, coalescing, warp contention
- `tests/alignment_test.rs` — Verifies Command/CommandQueueHost struct layout matches CUDA C++
- `tests/integration_test.rs` — Requires CUDA hardware (all 4 tests are `#[ignore]`)

## Integration Tests

`tests/integration_test.rs` uses mock types and all tests are `#[ignore]` because they require CUDA hardware. These are placeholder tests for real GPU validation.

## Notes

- `latency_report.json` and `rtt_latency_report.json` are runtime artifacts, not committed
- GPU metrics use simulated fallback without NVML; enable with `--features gpu-metrics`
- NVRTC compilation uses simulated fallback; enable with `--features nvrtc`
- The installer's `--heuristic-only` mode works fully offline
- `.cudaclaw/roles/` directory is created by the installer at runtime
