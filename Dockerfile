FROM python:3.11-slim

WORKDIR /app

# System deps for native extensions (numpy, tokenizers, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install Nix (single-user mode for containers)
RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | \
    sh -s -- install linux --no-confirm --init none
ENV PATH="/nix/var/nix/profiles/default/bin:${PATH}"

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy shell.nix and pre-populate the nix-shell environment
COPY shell.nix ./
RUN nix-shell --run "echo 'nix-shell deps cached'"

# Copy everything needed for install
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install the package with all extras + aiohttp for the web demo
RUN uv pip install --system -e ".[wasm,eval]" aiohttp

# Copy runtime assets
COPY web/ web/
COPY examples/ examples/
# Download CPython WASI binary for the eval sandbox
RUN mkdir -p .wasm && \
    curl -fsSL -o .wasm/python-3.12.0.wasm \
    "https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python/3.12.0%2B20231211-040d5a6/python-3.12.0.wasm"

# Default: run the web demo on port 8765
EXPOSE 8765
ENV PORT=8765

CMD ["python", "web/server.py"]
