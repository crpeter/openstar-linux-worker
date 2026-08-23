FROM rust:1.88-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin openstar \
 && install -d -o openstar -g openstar /var/lib/openstar-worker \
 && apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/openstar-linux-worker /usr/local/bin/
USER openstar
VOLUME ["/var/lib/openstar-worker"]
ENTRYPOINT ["/usr/local/bin/openstar-linux-worker"]
