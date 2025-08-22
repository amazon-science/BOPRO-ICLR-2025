#!/bin/bash -e

CAND_PATH="data/semantle"
TARGET="computer"
SEED=42
N_WARMSTART=20
WARMSTART_STRATEGY="random"
GEN_MODEL="llama-3-70b-instruct-bedrock"
REPR_MODEL="t5-base"
ACQUISITION_FN="qLogEI"
OPT_HULL_MARGIN="1.5"
OPT_BATCH_SIZE="2"
LOW_DIM_STRATEGY="random"
LOW_DIM=10
# opt_square_hull
# llm_enable_cot
RUN_ID=""
WILDCARD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Script arguments
        --cand_path) CAND_PATH="$2"; shift ;;
        --target) TARGET="$2"; shift ;;
        --n_warmstart) N_WARMSTART="$2"; shift ;;
        --warmstart_strategy) WARMSTART_STRATEGY="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --gen_model) GEN_MODEL="$2"; shift ;;
        --repr_model) REPR_MODEL="$2"; shift ;;
        --acquisition_fn) ACQUISITION_FN="$2"; shift ;;
        --opt_hull_margin) OPT_HULL_MARGIN="$2"; shift ;;
        --opt_batch_size) OPT_BATCH_SIZE="$2"; shift ;;
        --low_dim_strategy) LOW_DIM_STRATEGY="$2"; shift ;;
        --low_dim) LOW_DIM="$2"; shift ;;
        --run_id) RUN_ID="$2"; shift ;;
        --wildcard) WILDCARD="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Activate env
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bogen
export PYTHONPATH=$(pwd):$PYTHONPATH;
# Echo all script arguments in one line
echo "EXPERIMENT ARGS: cand_path=$CAND_PATH, target=$TARGET, n_warmstart=$N_WARMSTART, seed=$SEED, \
gen_model=$GEN_MODEL, repr_model=$REPR_MODEL, acquisition_fn=$ACQUISITION_FN, \
warmstart_strategy=$WARMSTART_STRATEGY, opt_hull_margin=$OPT_HULL_MARGIN, opt_batch_size=$OPT_BATCH_SIZE, \
low_dim_strategy=$LOW_DIM_STRATEGY, low_dim=$LOW_DIM, wildcard=$WILDCARD
"

# Run experiment
python src/semantle_bo.py \
  --seed=$SEED \
  --n_warmstart=$N_WARMSTART \
  --warmstart_strategy="$WARMSTART_STRATEGY" \
  --candidates_fname="$CAND_PATH/$TARGET.csv" \
  --target="$TARGET" \
  --gen_model="$GEN_MODEL" \
  --repr_model="$REPR_MODEL" \
  --acquisition_fn="$ACQUISITION_FN" \
  --opt_hull_margin=$OPT_HULL_MARGIN \
  --opt_batch_size=$OPT_BATCH_SIZE \
  --low_dim_strategy="$LOW_DIM_STRATEGY" \
  --low_dim=$LOW_DIM \
  ${RUN_ID:+--run_id="$RUN_ID"}${WILDCARD:+ $WILDCARD}
