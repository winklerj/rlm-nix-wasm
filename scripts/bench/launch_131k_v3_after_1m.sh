#!/bin/bash
# Waits for the 1M run to complete 50 tasks, then launches the 131K v3
# rerun (new-harness comparison) in its own tmux session.
trap '' HUP
d=/var/lib/microvms/rlm-secure
f=$d/results/oolong-trec-coarse-1048576-maxdepth1.jsonl
while [ "$(grep -c . "$f" 2>/dev/null || echo 0)" -lt 50 ]; do sleep 60; done
sleep 60
tmux kill-session -t oolong 2>/dev/null
echo "$(date -u) 1M complete; launching 131K v3" >> /tmp/oolong_131072_v3.log
tmux new-session -d -s oolong131v3 "$d/scripts/bench/run_trec.sh 131072 -v3 2>&1 | tee -a /tmp/oolong_131072_v3.log"
