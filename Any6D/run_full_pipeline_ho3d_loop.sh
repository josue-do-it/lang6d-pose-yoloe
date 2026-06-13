#!/bin/bash
# Full pipeline: LLM keyword extractor → YOLOE → Any6D → BOP metrics
# Runs one process per object to avoid CUDA context corruption.
# At the end: builds final summary + generates all plots.

OBJECTS=("MPM10" "MPM11" "MPM12" "MPM13" "MPM14" "AP10" "AP11" "AP12" "AP13" "AP14" "SB11" "SB13" "SM1")
TOTAL=${#OBJECTS[@]}

ANCHOR="/workspace/anchor_results/dexycb_reference_view_ours"
HO3D_ROOT="/dataset/ho3d/HO3D_data"
YCB_MODELS="/dataset/ho3d"
SAVE_DIR="/workspace/results/ho3d_pipeline/run_full"
LOG_DIR="/workspace/results/ho3d_pipeline/logs"
LLM_MODEL="mistral:latest"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "======================================"
echo "HO3D Full Pipeline: LLM + YOLOE + Any6D"
echo "LLM: $LLM_MODEL"
echo "Save: $SAVE_DIR"
echo "======================================"

for idx in $(seq 0 $((TOTAL-1))); do
    OBJ=${OBJECTS[$idx]}
    XLSX="$SAVE_DIR/${OBJ}_metrics_results.xlsx"

    if [ -f "$XLSX" ]; then
        echo "[$idx/$((TOTAL-1))] $OBJ — already done, skipping"
        continue
    fi

    echo "[$idx/$((TOTAL-1))] Running $OBJ ..."
    python run_full_pipeline_ho3d.py \
        --anchor_path "$ANCHOR" \
        --hot3d_data_root "$HO3D_ROOT" \
        --ycb_model_path "$YCB_MODELS" \
        --start_idx "$idx" \
        --end_idx "$((idx+1))" \
        --save_dir "$SAVE_DIR" \
        --llm_model "$LLM_MODEL" \
        > "$LOG_DIR/${OBJ}.log" 2>&1

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "[$idx] $OBJ — FAILED (exit $EXIT), check $LOG_DIR/${OBJ}.log"
    else
        echo "[$idx] $OBJ — done"
    fi
done

echo ""
echo "All objects done. Building final summary..."
python make_final_summary_ho3d.py --results_dir "$SAVE_DIR"

echo "Generating HO3D plots..."
python make_pipeline_plots.py \
    --ho3d_dir "$SAVE_DIR" \
    --out_dir "/workspace/results/pipeline_plots"

echo ""
echo "Done. Results:"
echo "  Pose + detection: $SAVE_DIR"
echo "  Plots:            /workspace/results/pipeline_plots/ho3d/"
