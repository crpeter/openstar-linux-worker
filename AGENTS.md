# Contributor guidance

This repository is a generic compute worker. Keep astronomy, dataset discovery,
target selection, scheduling, interpretation, and persisted science state in the
coordinator. Do not add workload protocol fields without confirming them in the
current server. This worker is CPU-only until a separate acceleration change.

Before committing, run `cargo fmt --check`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-targets`, and `cargo build --release`.
