#!/bin/bash
# Run inside Docker: docker exec any6d_active bash /workspace/run_replot.sh
# Generates publication-quality report figures from real-world iPhone LiDAR test.

python /workspace/visualization/replot_infer_pose.py \
    --json    /workspace/results/infer_pose/0000001_pose.json \
    --image   /workspace/test_input/0000001.jpg \
    --ply     /workspace/test_input/0000001.ply \
    --K       /workspace/test_input/K.txt \
    --out_dir /workspace/results/infer_pose/report_plots

echo ""
echo "Figures:"
ls -lh /workspace/results/infer_pose/report_plots/
