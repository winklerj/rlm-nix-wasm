FROM python:3.11-slim

WORKDIR /app

# System deps for native extensions (numpy, tokenizers, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy everything needed for install
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install the package with all extras + aiohttp for the web demo
RUN uv pip install --system -e ".[wasm,eval]" aiohttp

# Copy runtime assets
COPY web/ web/
COPY examples/ examples/
COPY .wasm/ .wasm/

# Default: run the web demo on port 8765
EXPOSE 8765
ENV PORT=8765

CMD ["python", "web/server.py"]
