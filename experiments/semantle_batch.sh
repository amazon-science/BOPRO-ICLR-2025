#!/bin/bash -e

# Iterable parameters
TARGETS="computer"
SEEDS="42"
N_WARMSTARTS="20"
WARMSTART_STRATEGIES="random"
GEN_MODELS="llama-3-70b-instruct-bedrock"
REPR_MODELS="t5-base"
ACQUISITION_FNS="qLogEI"
OPT_HULL_MARGINS="1.5"
OPT_BATCH_SIZES="2"
LOW_DIM_STRATEGIES="pca"
LOW_DIMS="10"
WILDCARDS="none"
# Non-iterable parameters
CAND_PATH="data/semantle"
CONST_WILDCARD="none"
# Runner parameters
N_ASYNC="1"
TMUX_SESSION="experiments"
LOG_FILE="experiments/log.csv"
PER_RUN_LOG_DIR="experiments/logs"
SLEEP=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Script arguments
        --cand_path) CAND_PATH="$2"; shift ;;
        --targets) TARGETS="$2"; shift ;;
        --n_warmstarts) N_WARMSTARTS="$2"; shift ;;
        --warmstart_strategies) WARMSTART_STRATEGIES="$2"; shift ;;
        --seeds) SEEDS="$2"; shift ;;
        --gen_models) GEN_MODELS="$2"; shift ;;
        --repr_models) REPR_MODELS="$2"; shift ;;
        --acquisition_fns) ACQUISITION_FNS="$2"; shift ;;
        --opt_hull_margins) OPT_HULL_MARGINS="$2"; shift ;;
        --opt_batch_sizes) OPT_BATCH_SIZES="$2"; shift ;;
        --low_dim_strategies) LOW_DIM_STRATEGIES="$2"; shift ;;
        --low_dims) LOW_DIMS="$2"; shift ;;
        --wildcard) CONST_WILDCARD="$2"; shift ;;
        --wildcards) WILDCARDS="$2"; shift ;;
        --n_async) N_ASYNC="$2"; shift ;;
        --tmux_session) TMUX_SESSION="$2"; shift ;;
        --log_file) LOG_FILE="$2"; shift ;;
        --per_run_log_dir) PER_RUN_LOG_DIR="$2"; shift ;;
        --sleep) SLEEP="$2"; shift ;;
        *) echo "Invalid option: $1" >&2; exit 1 ;;
    esac
    shift
done

# Log the command execution to the log file
log_command() {
    local window_name="$1"
    local cmd="$2"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "$timestamp,$window_name,$cmd" >> "$LOG_FILE"
}

# Get timestamp for tmux window name
get_current_timestamp() {
  date +%s%3N
}

# Get the number of active tmux windows in the "$TMUX_SESSION" session minus 1 (for the default window that opens)
get_active_tmux_windows() {
    echo $(( $(tmux list-windows -t "$TMUX_SESSION" | wc -l) - 1 ))
}

# Wait until the number of active tmux windows is below the limit
wait_for_slot() {
    local DISPLAYED_WAIT_MSG=false
    while [ "$(get_active_tmux_windows)" -ge "$N_ASYNC" ]; do
        if [ "$DISPLAYED_WAIT_MSG" = false ]; then
            printf "Waiting for slots...\n"
            DISPLAYED_WAIT_MSG=true
        fi
        sleep "$SLEEP"
    done
}

# Ensure log file exists and add header if it's a new file
if [ ! -f "$LOG_FILE" ]; then
    echo "timestamp,window_name,command" > "$LOG_FILE"
fi

# Ensure per-run log directory exists
mkdir -p "$PER_RUN_LOG_DIR"

# Check if "$TMUX_SESSION" session exists and create if not
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$TMUX_SESSION"
fi

# Start experiments
for TARGET in $TARGETS; do
    for SEED in $SEEDS; do
        for WARMSTART_STRATEGY in $WARMSTART_STRATEGIES; do
            for N_WARMSTART in $N_WARMSTARTS; do
                for GEN_MODEL in $GEN_MODELS; do
                    for REPR_MODEL in $REPR_MODELS; do
                        for ACQUISITION_FN in $ACQUISITION_FNS; do
                            for OPT_HULL_MARGIN in $OPT_HULL_MARGINS; do
                                for OPT_BATCH_SIZE in $OPT_BATCH_SIZES; do
                                    for LOW_DIM_STRATEGY in $LOW_DIM_STRATEGIES; do
                                        for LOW_DIM in $LOW_DIMS; do
                                            for WILDCARD in $WILDCARDS; do
                                                if [ "$WILDCARD" == "none" ]; then
                                                    WILDCARD=""
                                                fi
                                                if [ "$CONST_WILDCARD" != "none" ]; then
                                                    # Strip leading and trailing whitespaces
                                                    WILDCARD=$(echo "$WILDCARD $CONST_WILDCARD" | xargs)
                                                fi
                                                run_cmd() {
                                                    CMD="./experiments/semantle.sh \
                                                      --run_id "$1" \
                                                      --cand_path $CAND_PATH \
                                                      --target $TARGET \
                                                      --n_warmstart $N_WARMSTART \
                                                      --warmstart_strategy $WARMSTART_STRATEGY \
                                                      --seed $SEED \
                                                      --gen_model $GEN_MODEL \
                                                      --repr_model $REPR_MODEL \
                                                      --acquisition_fn $ACQUISITION_FN \
                                                      --opt_hull_margin $OPT_HULL_MARGIN \
                                                      --opt_batch_size $OPT_BATCH_SIZE \
                                                      --low_dim_strategy $LOW_DIM_STRATEGY \
                                                      --low_dim $LOW_DIM${WILDCARD:+ --wildcard \"$WILDCARD\"}"
                                                    echo $CMD
                                                }
                                                if [ $N_ASYNC -gt 0 ]; then
                                                    wait_for_slot
                                                    WINDOW_NAME="$(get_current_timestamp)"
                                                    CMD=$(run_cmd "$WINDOW_NAME")
                                                    tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME" "$CMD 2>&1 | tee -a $PER_RUN_LOG_DIR/$WINDOW_NAME.log"
                                                else
                                                    WINDOW_NAME="$(get_current_timestamp)"
                                                    CMD=$(run_cmd "$WINDOW_NAME")
                                                    eval "$CMD"
                                                fi
                                                log_command "$WINDOW_NAME" "$CMD"
                                                echo "$CMD"
                                                echo "Submitted: $WINDOW_NAME"
                                                echo ""
                                                sleep 1
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
