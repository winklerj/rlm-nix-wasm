# Shared environment for OOLONG benchmark runs against the self-hosted
# llama-server. Reads OPENAI_API_KEY from .env; everything else is explicit
# here so a reboot can't lose run configuration (scripts used to live in /tmp).
set -eu
cd /var/lib/microvms/rlm-secure
export PATH=/nix/var/nix/profiles/default/bin:/run/current-system/sw/bin:$HOME/.nix-profile/bin:$HOME/.local/bin:$PATH
export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2-)
# Resolve the mDNS name here, outside nix-shell: the nix glibc cannot load
# the host's mdns NSS module, so .local names do not resolve inside it.
LLM_HOST=$(getent hosts star-destroyer.local | awk '{print $1}' | head -1)
[ -n "$LLM_HOST" ] || { echo "cannot resolve star-destroyer.local" >&2; exit 1; }
export OPENAI_BASE_URL=http://$LLM_HOST:11434/v1
export RLM_MAX_PARALLEL_JOBS=3
export RLM_REASONING_STRENGTH=low
export RLM_TEMPERATURE=0.0
export RLM_MAX_OUTPUT_TOKENS=16384
export RLM_MODEL="${RLM_MODEL:-openai//models/Muse-Glimmer-30B-UD-Q5_K_M.gguf}"
