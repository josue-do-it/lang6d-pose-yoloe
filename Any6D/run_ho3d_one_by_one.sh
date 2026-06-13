#!/bin/bash
# Run run_ho3d_query.py one object at a time (fresh process per object to avoid CUDA segfault)
# At the end, combine all results into a single mean Excel

OBJECTS=("MPM10" "MPM11" "MPM12" "MPM13" "MPM14" "AP10" "AP11" "AP12" "AP13" "AP14" "SB11" "SB13" "SM1")
TOTAL=${#OBJECTS[@]}

ANCHOR="/workspace/anchor_results/dexycb_reference_view_ours"
HO3D_ROOT="/dataset/ho3d/HO3D_data"
YCB_MODELS="/dataset/ho3d"
SAVE_DIR="/workspace/results/ho3d_results/any6d/run_full"
LOG_DIR="/workspace/results/ho3d_results/any6d/logs"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

for idx in $(seq 0 $((TOTAL-1))); do
    OBJ=${OBJECTS[$idx]}
    XLSX="$SAVE_DIR/${OBJ}_metrics_results.xlsx"

    if [ -f "$XLSX" ]; then
        echo "[$idx/$((TOTAL-1))] $OBJ — already done, skipping"
        continue
    fi

    echo "[$idx/$((TOTAL-1))] Running $OBJ ..."
    python run_ho3d_query.py \
        --anchor_path "$ANCHOR" \
        --hot3d_data_root "$HO3D_ROOT" \
        --ycb_model_path "$YCB_MODELS" \
        --start_idx "$idx" \
        --end_idx "$((idx+1))" \
        --save_dir "$SAVE_DIR" \
        > "$LOG_DIR/${OBJ}.log" 2>&1

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "[$idx] $OBJ — FAILED (exit $EXIT), check $LOG_DIR/${OBJ}.log"
    else
        echo "[$idx] $OBJ — done"
    fi
done

echo ""
echo "All objects done. Computing final combined mean..."
python combine_ho3d_results.py --results_dir "$SAVE_DIR"
echo "Done. Results in $SAVE_DIR"
