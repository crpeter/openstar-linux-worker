# OpenStar Linux worker

A headless, generic CPU compute worker. It registers with an OpenStar coordinator,
claims opaque work, downloads its referenced dataset, and currently dispatches only
`openstar.lomb-scargle.v1`. It contains no dataset discovery or science orchestration.

The wire models follow the verified `openstarserver/main` contract and the CPU
kernel follows the Float32 scalar semantics of the validator in `OpenStar/main`.
Registration supplies a persistent node UUID and advertises only the CPU backend.

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
| `OPENSTAR_CPU_THREADS` | host parallelism | Rayon CPU threads |
| `OPENSTAR_POLL_INTERVAL_MS` | `2000` | Delay after no work |
| `OPENSTAR_MAX_BACKOFF_MS` | `30000` | Retry delay ceiling |
| `OPENSTAR_REQUEST_TIMEOUT_SECS` | `30` | Per-request timeout |
| `OPENSTAR_LOG` | `info` | tracing filter |

SIGINT/SIGTERM stops claiming; active units finish result submission before exit.
The worker advertises CPU threads and the one workload only—no GPU capability.
The service's `StateDirectory` gives its unprivileged user write access to the
identity directory. Back up `node-id` to preserve coordinator identity when moving
the worker to a new host, or configure `OPENSTAR_NODE_ID` explicitly.

Tests use only local mock HTTP servers; they never connect to or claim work from a
live coordinator.
