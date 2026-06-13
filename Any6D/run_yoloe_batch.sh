#!/bin/bash
# Lance run_yoloe_ho3d_query.py objet par objet
# Reprend automatiquement après chaque kill

CMD="PYOPENGL_PLATFORM=osmesa python run_yoloe_ho3d_query.py \
  --anchor_path anchor_results/dexycb_reference_view_ours \
  --hot3d_data_root dataset/ho3d/HO3D_data \
  --ycb_model_path dataset/ho3d/YCB_Video_Models"

TOTAL=13

for idx in $(seq 0 $((TOTAL-1))); do
    # Vérifier si déjà traité
    OBJETS=("MPM10" "MPM11" "MPM12" "MPM13" "MPM14" "AP10" "AP11" "AP12" "AP13" "AP14" "SB11" "SB13" "SM1")
    OBJ=${OBJETS[$idx]}
    
    # Chercher si xlsx existe déjà
    DONE=$(find results/ -name "${OBJ}_metrics_results.xlsx" 2>/dev/null | wc -l)
    if [ "$DONE" -gt "0" ]; then
        echo "[$idx] $OBJ — déjà traité, skip"
        continue
    fi
    
    echo "[$idx] Lancement $OBJ ..."
    eval "$CMD --start_idx $idx" &
    PID=$!
    
    # Attendre max 15 minutes par objet
    TIMEOUT=900
    ELAPSED=0
    while kill -0 $PID 2>/dev/null; do
        sleep 10
        ELAPSED=$((ELAPSED+10))
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo "[$idx] $OBJ — timeout, kill manuel"
            kill $PID 2>/dev/null
            break
        fi
    done
    
    echo "[$idx] $OBJ — terminé"
    sleep 5
done

echo "Tous les objets traités"
