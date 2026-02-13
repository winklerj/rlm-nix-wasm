"""Nix expression templates for compiling DSL operations to derivations."""

from __future__ import annotations

GREP_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-grep-{hash}" {{
  input = {input_path};
  pattern = "{pattern}";
}} ''
  grep -E "$pattern" "$input" > $out || true
''
'''

SLICE_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-slice-{hash}" {{
  input = {input_path};
}} ''
  tail -c +{start} "$input" | head -c {length} > $out
''
'''

COUNT_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-count-{hash}" {{
  input = {input_path};
}} ''
  wc -{mode_flag} < "$input" | tr -d ' ' > $out
''
'''

CHUNK_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-chunk-{hash}" {{
  input = {input_path};
}} ''
  mkdir -p $out
  total=$(wc -l < "$input")
  chunk_size=$(( (total + {n} - 1) / {n} ))
  split -l "$chunk_size" -d --additional-suffix=.txt "$input" "$out/chunk_"
''
'''

SPLIT_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-split-{hash}" {{
  input = {input_path};
  delimiter = "{delimiter}";
}} ''
  mkdir -p $out
  csplit -f "$out/part_" -z "$input" "/$delimiter/" '{{*}}' 2>/dev/null || cp "$input" "$out/part_00"
''
'''

COMBINE_CONCAT_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-combine-{hash}" {{
  inputs = [{input_paths}];
}} ''
  for f in $inputs; do
    cat "$f" >> $out
    echo >> $out
  done
''
'''

COMBINE_SUM_TEMPLATE = '''{{ pkgs ? import <nixpkgs> {{}} }}:

pkgs.runCommand "rlm-combine-sum-{hash}" {{
  inputs = [{input_paths}];
}} ''
  total=0
  for f in $inputs; do
    val=$(cat "$f" | tr -d '[:space:]')
    if [ -n "$val" ] && echo "$val" | grep -qE '^[0-9]+$'; then
      total=$((total + val))
    fi
  done
  echo "$total" > $out
''
'''
