# --- Stage 1: Build Rust CLOB extension ---
FROM python:3.12-slim AS rust-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev git && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

# Clone the Polymarket Rust SDK
RUN git clone --depth 1 https://github.com/Polymarket/rs-clob-client /tmp/rs-clob-client

# Copy our PyO3 wrapper crate
COPY rs-clob-python/ /app/rs-clob-python/

# Build the wheel
WORKDIR /app/rs-clob-python
RUN maturin build --release --out /app/wheels

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the Rust CLOB extension (optional — falls back to Python if missing)
COPY --from=rust-builder /app/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl 2>/dev/null || echo "Rust extension unavailable, using Python fallback"
RUN rm -rf /tmp/wheels

COPY src/ src/
COPY config/ config/

RUN useradd --create-home --shell /bin/bash mvhe
USER mvhe

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--paper", "--strategy", "singularity"]
