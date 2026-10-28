#!/usr/bin/env bash
# MatEvolve smoke test: the README's command sequence, run with the DUMMY model (configs/smoke_run.yaml).
# No language model, no DFT, no credits.  Every command below is typed out exactly as a user would type it;
# the only differences from a real run are the configuration file and the two stand-ins
# (cluster_tools.py dummy-votes for the skill's worker votes, scripts/dummy_dft.py for the DFT tool).
#
#   bash scripts/smoke_test.sh                 # 3 sample structures, 2 miniature generations, ~20 s -> runs/smoke/
#   CIFS=my_cifs bash scripts/smoke_test.sh    # your own (few) CIF files as seeds
#   GENERATIONS=1 bash scripts/smoke_test.sh
#   RUN=/tmp/smoke CFG=/tmp/smoke.yaml bash scripts/smoke_test.sh   # another run folder (CFG: a copy of smoke_run.yaml with its paths)
#   rm -rf runs/smoke                          # start over
#
# Nothing in runs/smoke/ is a scientific result (see runs/smoke/DUMMY_LLM.txt).

set -euo pipefail
cd "$(dirname "$0")/.."                        # repository root

CFG=${CFG:-configs/smoke_run.yaml}
RUN=${RUN:-runs/smoke}
DB=$RUN/database
TOOL=skills/matevolve-cluster/scripts/cluster_tools.py
CIFS=${CIFS:-tests/data/sample_db}
GENERATIONS=${GENERATIONS:-2}

run() { echo; echo "\$ $*"; "$@"; }             # echo each command, then run it

# -- the motif-cluster skill hand-over, without workers (dummy votes) -----------------------------------------------
#    $1 = motif source (database folder or gen_XX/offspring.json)   $2 = state folder   $3.. = apply target
cluster_dummy() {
  local source=$1 state=$2; shift 2
  local dict=()
  if [ -f "$DB/motif_groups/clusters.json" ]; then
    dict=(--dictionary "$DB/motif_groups/clusters.json")           # frozen vocabulary: assignment only
    [ -d "$source" ] && dict+=(--only-unassigned)
  fi
  run python "$TOOL" init        --motifs "$source" --state "$state" --mode full "${dict[@]}"
  run python "$TOOL" passes      --state "$state" --round 0
  run python "$TOOL" dummy-votes --state "$state" --round 0        # <- the workers' votes, faked from the vocabulary tags
  run python "$TOOL" tally       --state "$state" --round 0
  run python "$TOOL" finalize    --state "$state" --round 0
  run python "$TOOL" apply       --state "$state" "$@"
}

# -- 1.2.2  a new database from CIF files ------------------------------------------------------------------------------
if [ ! -f "$DB/data.csv" ]; then
  run python scripts/run_matevolve.py template --config $CFG --cifs "$CIFS" --out $DB
  run python scripts/dummy_dft.py --config $CFG --run $RUN --seeds        # [you]  fill the properties -> placeholders
fi

# -- 3.3  fresh scenario: check, init, build the motif-group vocabulary --------------------------------------------
run python scripts/run_matevolve.py check --config $CFG --run $RUN --llm
run python scripts/run_matevolve.py init  --config $CFG --run $RUN                              # step 0
[ -f "$DB/motif_groups/clusters.json" ] || cluster_dummy $DB $RUN/cluster_init --database $DB    # [skill] cluster all motifs

# -- 3.2  the generation loop ----------------------------------------------------------------------------------------
for g in $(seq 1 "$GENERATIONS"); do
  gen=$(printf "gen_%02d" "$g")
  run python scripts/run_matevolve.py generate --config $CFG --run $RUN                         # step 1
  cluster_dummy $RUN/$gen/offspring.json $RUN/$gen/cluster_offspring --offspring $RUN/$gen/offspring.json   # [skill] assign
  run python scripts/run_matevolve.py rank     --config $CFG --run $RUN                         # step 2
  run python scripts/dummy_dft.py              --config $CFG --run $RUN                         # [DFT]  placeholders
  run python scripts/run_matevolve.py update   --config $CFG --run $RUN                         # step 3
  cluster_dummy $DB $RUN/cluster_db_$(printf "g%02d" "$g") --database $DB                       # [skill] assign the new motifs
done

run python scripts/run_matevolve.py score --config $CFG --run $RUN

echo
echo "smoke test passed. Output tree: $RUN/ (layout: docs/FOLDER_LAYOUT.md)."
echo "Every file in it is a placeholder (DUMMY_LLM.txt). Real runs: configs/magnet_run.yaml (API) or configs/agent_run.yaml (skills)."
