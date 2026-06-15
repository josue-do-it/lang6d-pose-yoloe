"""
Results I/O: save pose metrics to JSON and Excel.

Output format rationale
-----------------------
JSON is the canonical format for downstream processing and version control.
XLSX is generated alongside it because researchers often want to inspect
per-frame numbers in a spreadsheet without writing parsing code.

The MEAN row at the bottom of each XLSX sheet makes it easy to copy
aggregate numbers directly into a paper table without further computation.
"""
import json
import os
import numpy as np
import pandas as pd

from .metrics_utils import nanmean


def save_json(path: str, data, indent: int = 2) -> None:
    """
    Serialise data to a JSON file, creating parent directories if needed.

    Args:
        path:   Absolute or relative file path (e.g. "results/obj_01.json").
        data:   Any JSON-serialisable Python object.
        indent: Number of spaces for pretty-printing (default 2).
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def build_summary_row(
    obj_id: int,
    obj_name: str,
    metric_lists: dict,
    instruction: str,
    keyword: str,
    symmetric: bool = False,
) -> dict:
    """
    Aggregate per-frame metric lists into a single summary dictionary.

    The "used" ADD score is ADD-S for symmetric objects and ADD for asymmetric
    ones — this matches the BOP evaluation convention where symmetric objects
    are always evaluated with ADD-S regardless of which metric was computed.

    Args:
        obj_id:        Integer object identifier (matches dataset models_info).
        obj_name:      Human-readable object name (e.g. "mustard_bottle").
        metric_lists:  Dict of lists, keyed by metric name.
                       Expected keys: ADD, ADD-S, AR, R_error, T_error.
                       Optional keys: yoloe_det.
        instruction:   The natural language instruction used for this object.
        keyword:       The YOLOE keyword extracted by the LLM.
        symmetric:     Whether the object is rotationally symmetric.
                       Affects which ADD variant is labelled "used".

    Returns:
        Flat dict suitable for appending to a results table or writing to JSON.
    """
    metric_sym = 'ADD-S' if symmetric else 'ADD'
    add_score  = nanmean(metric_lists[metric_sym])
    r_vals = [v for v in metric_lists['R_error'] if not np.isnan(v)]
    return {
        'obj_id':         obj_id,
        'obj_name':       obj_name,
        'instruction':    instruction,
        'keyword':        keyword,
        'symmetric':      symmetric,
        'n_frames':       len(metric_lists['ADD']),
        'yoloe_det_rate': nanmean(metric_lists.get('yoloe_det', [])),
        'ADD':            nanmean(metric_lists['ADD']),
        'ADD-S':          nanmean(metric_lists['ADD-S']),
        f'{metric_sym} (used)': add_score,
        'AR':             nanmean(metric_lists['AR']),
        'R_error_mean':   nanmean(metric_lists['R_error']),
        'R_error_med':    float(np.median(r_vals)) if r_vals else float('nan'),
        'T_error_mean':   nanmean(metric_lists['T_error']),
    }


def save_per_object_xlsx(
    path: str,
    obj_name: str,
    frames_out: list,
    metric_lists: dict,
) -> None:
    """
    Write per-frame metrics to an Excel file with a MEAN summary row at the bottom.

    Args:
        path:         Output .xlsx file path.
        obj_name:     Object name used to populate the obj_name column.
        frames_out:   List of per-frame dicts containing at least 'im_id'.
        metric_lists: Dict of lists keyed by metric name (ADD, ADD-S, AR, etc.).
                      Lists must have the same length as frames_out.
    """
    df = pd.DataFrame({
        'im_id':     [fr['im_id'] for fr in frames_out],
        'obj_name':  obj_name,
        'ADD':       metric_lists['ADD'],
        'ADD-S':     metric_lists['ADD-S'],
        'AR':        metric_lists['AR'],
        'MSSD':      metric_lists['MSSD'],
        'MSPD':      metric_lists['MSPD'],
        'R_error':   metric_lists['R_error'],
        'T_error':   metric_lists['T_error'],
        'yoloe_det': metric_lists.get('yoloe_det', [None]*len(frames_out)),
    })
    mean_row = {
        'im_id': 'MEAN', 'obj_name': obj_name,
        'ADD':   f"{nanmean(metric_lists['ADD'])*100:.1f}%",
        'ADD-S': f"{nanmean(metric_lists['ADD-S'])*100:.1f}%",
        'AR':    f"{nanmean(metric_lists['AR'])*100:.1f}%",
        'MSSD':  f"{nanmean(metric_lists['MSSD']):.4f}",
        'MSPD':  f"{nanmean(metric_lists['MSPD']):.1f}",
        'R_error': f"{nanmean(metric_lists['R_error']):.2f}°",
        'T_error': f"{nanmean(metric_lists['T_error']):.2f}cm",
        'yoloe_det': (f"{nanmean(metric_lists['yoloe_det'])*100:.0f}%"
                      if metric_lists.get('yoloe_det') else '—'),
    }
    pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True).to_excel(
        path, index=False)


def save_result(out_dir: str, record: dict) -> None:
    """
    Save a single-image pipeline result to result.json and result.xlsx.

    The record dict must contain 'pose_4x4' as a 4×4 list or numpy array.
    All other keys (instruction, keyword, yoloe_detected, metrics, …) are
    included automatically in both output files.

    The XLSX flattens the rotation matrix into R00…R22 columns and the
    translation into tx_cm / ty_cm / tz_cm (centimetres) for easy reading.

    Args:
        out_dir: Directory where result.json and result.xlsx are written.
                 Created if it does not exist.
        record:  Dictionary containing at minimum {'pose_4x4': <4×4>}.
    """
    pose = np.array(record['pose_4x4'])
    R, t = pose[:3, :3], pose[:3, 3]

    full = dict(record)
    full['pose_4x4'] = pose.tolist()
    full.setdefault('T_pred_cm', [round(float(v) * 100, 3) for v in t])

    save_json(os.path.join(out_dir, "result.json"), full)

    # Flatten nested structures (lists, dicts, arrays) into scalar columns
    row = {k: v for k, v in full.items()
           if not isinstance(v, (list, dict, np.ndarray))}
    for i in range(3):
        for j in range(3):
            row[f'R{i}{j}'] = round(float(R[i, j]), 6)
    row['tx_cm'] = round(float(t[0]) * 100, 3)
    row['ty_cm'] = round(float(t[1]) * 100, 3)
    row['tz_cm'] = round(float(t[2]) * 100, 3)

    pd.DataFrame([row]).to_excel(os.path.join(out_dir, "result.xlsx"), index=False)
    print(f"Saved → {out_dir}/result.json  +  result.xlsx")
