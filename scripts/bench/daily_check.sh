#!/bin/bash
# Cron: daily progress summary of all result files to /tmp/oolong_daily_report.txt
cd /var/lib/microvms/rlm-secure
{
date -u
python3 - <<'PY'
import glob, json
for f in sorted(glob.glob("results/oolong-trec-coarse-*maxdepth1*.jsonl")):
    rs = [json.loads(l) for l in open(f) if l.strip()]
    if not rs: continue
    avg = sum(r["score"] for r in rs) / len(rs)
    print(f"{f}: {len(rs)}/50 tasks, avg {avg:.3f}")
PY
tmux list-sessions 2>&1
} > /tmp/oolong_daily_report.txt 2>&1
