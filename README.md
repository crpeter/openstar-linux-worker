# OpenStar Linux worker

A headless, generic CPU/Vulkan compute worker. It registers with an OpenStar coordinator,
claims opaque work, downloads its referenced dataset, and dispatches
`openstar.lomb-scargle.v1` plus the CPU-only generic
`openstar.box-period-search.v1`. It contains no dataset discovery or science orchestration.

The wire models follow the verified `openstarserver/main` contract and the CPU
kernel follows the Float32 scalar semantics of the validator in `OpenStar/main`.
Registration supplies a persistent node UUID and advertises the backend actually selected.
Periodic-box work phase-bins each assigned frequency and scores circular low-value
windows with deterministic Float32 tie-breaking; it makes no scientific classification.
Vulkan is optional; normal installations need only a Vulkan loader and driver because
the GLSL compute shader is compiled to SPIR-V by `shaderc` at build time and embedded.

## Build and test

```sh
cargo build --release
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

## Run

```sh
cp openstar-worker.env.example /etc/openstar-worker.env
set -a; . /etc/openstar-worker.env; set +a
./target/release/openstar-linux-worker
```

The coordinator URL must end in `/` so relative `/v1/...` paths remain beneath its
path prefix. CLI flags correspond to the environment keys below; see `--help`.

| Variable | Default | Meaning |
|---|---:|---|
| `OPENSTAR_COORDINATOR_URL` | required | Coordinator base URL |
| `OPENSTAR_NODE_ID` | generated | Explicit stable node UUID |
| `OPENSTAR_STATE_DIR` | `/var/lib/openstar-worker` | Stores generated node identity |
| `OPENSTAR_WORK_CONCURRENCY` | `1` | Simultaneously claimed units |
| `OPENSTAR_WORK_BATCH_SIZE` | `1` | Units claimed together (1–128); a batch shares one dataset download |
| `OPENSTAR_CPU_THREADS` | host parallelism | Rayon CPU threads |
| `OPENSTAR_COMPUTE_BACKEND` | `auto` | `auto`, `cpu`, or `vulkan` |
| `OPENSTAR_POLL_INTERVAL_MS` | `2000` | Delay after no work |
| `OPENSTAR_MAX_BACKOFF_MS` | `30000` | Retry delay ceiling |
| `OPENSTAR_REQUEST_TIMEOUT_SECS` | `30` | Per-request timeout |
| `OPENSTAR_LOG` | `info` | tracing filter |

SIGINT/SIGTERM stops claiming; active units finish result submission before exit.
`auto` selects a discrete or integrated Vulkan GPU when possible and otherwise logs
a warning and falls back to CPU. `vulkan` makes initialization failure fatal. Vulkan
devices must expose a queue with both `GRAPHICS` and `COMPUTE`; a compute-only queue
is deliberately never selected (including when it appears to be dedicated), to avoid
the known AMD Liverpool dedicated-compute-ring hang. GPU frequency powers are copied
back and the host applies the CPU kernel's lowest-index exact-tie rule.
The shader follows the CPU kernel's Float32 formulas and accumulation order, but
GPU implementations of `sin`, `cos`, and `atan` can differ slightly from the host
math library. Host-side selection is deterministic for the powers returned by the
GPU; CPU/Vulkan power values are therefore compared with a numerical tolerance.
For wire compatibility, `cpuDurationSeconds` remains present for CPU results only;
Vulkan results use the existing backend-neutral total workload duration fields and
do not label Vulkan execution time as CPU (or Metal) time.
The service's `StateDirectory` gives its unprivileged user write access to the
identity directory. Back up `node-id` to preserve coordinator identity when moving
the worker to a new host, or configure `OPENSTAR_NODE_ID` explicitly.

Tests use only local mock HTTP servers; they never connect to or claim work from a
live coordinator.
