#!/bin/sh
# Fetch the inference schemas and the expected outputs of a probe's documents into
# artifacts/probes/data (git-ignored). Expected outputs are read only by tools/score_probe.py.
# Usage: tools/probe_data.sh <packet> <id> [<id> ...]
set -eu
packet=$1; shift
data=artifacts/probes/data
mkdir -p "$data"
modal volume get --force docextract-eval "packets/$packet/evaluation/index.json" "$data/index-$packet.json" >/dev/null
for id in "$@"; do
  [ -f "$data/$id.schema.json" ] || modal volume get docextract-work "packets/$packet/inference/$id.schema.json" "$data/$id.schema.json" >/dev/null
  test=$(python3 -c "import json,sys; print(next(c['test_json'] for c in json.load(open(sys.argv[1]))['cases'] if c['id']==sys.argv[2]))" "$data/index-$packet.json" "$id")
  [ -f "$data/$id.test.json" ] || modal volume get docextract-eval "packets/$packet/evaluation/$test" "$data/$id.test.json" >/dev/null
done
ls "$data" | wc -l
