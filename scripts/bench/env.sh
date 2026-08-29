# Shared environment for OOLONG benchmark runs against the self-hosted
# llama-server. Reads OPENAI_API_KEY from .env; everything else is explicit
# here so a reboot can't lose run configuration (scripts used to live in /tmp).
set -eu
cd /var/lib/microvms/rlm-secure
export PATH=/nix/var/nix/profiles/default/bin:/run/current-system/sw/bin:$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH
export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2-)
export OPENAI_BASE_URL=http://192.168.1.247:11434/v1
export RLM_MAX_PARALLEL_JOBS=3
export RLM_REASONING_STRENGTH=low
export RLM_TEMPERATURE=0.0
export RLM_MAX_OUTPUT_TOKENS=16384
export RLM_MODEL="${RLM_MODEL:-openai//models/Muse-Glimmer-30B-UD-Q5_K_M.gguf}"
