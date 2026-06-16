import argparse
import json
from pathlib import Path


EVAL_FILES = {
    "amass_after_pretrain": "eval_amass_after_pretrain.json",
    "dip_test_after_amass_pretrain": "eval_dip_test_after_amass_pretrain.json",
    "totalcapture_test_after_amass_pretrain": "eval_totalcapture_test_after_amass_pretrain.json",
    "dip_test_after_dip_finetune": "eval_dip_test_after_dip_finetune.json",
    "totalcapture_test_after_dip_finetune": "eval_totalcapture_test_after_dip_finetune.json",
}

DOWN = "\u2193"
BASELINE_MARKERS = (
    "official",
    "newpl_v4",
    "newpl_v5_raw",
    "newpl_v6_raw",
    "p_pdot_pddot_strong",
    "dip_last",
)


def load_json(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows(data, table):
    value = data.get(table, []) if data else []
    return value if isinstance(value, list) else []


def by_version(table_rows):
    return {str(row.get("Version", "")): row for row in table_rows}


def compact_eval(data):
    current = by_version(rows(data, "current_frame_table"))
    nxt = by_version(rows(data, "next_frame_table"))
    dyn = by_version(rows(data, "dynamics_table"))
    control = by_version(rows(data, "control_table"))
    versions = sorted(set(current) | set(nxt) | set(dyn) | set(control))
    out = []
    for version in versions:
        cur = current.get(version, {})
        next_row = nxt.get(version, {})
        dyn_row = dyn.get(version, {})
        ctl = control.get(version, {})
        if not any(marker in version for marker in BASELINE_MARKERS):
            continue
        out.append(
            {
                "version": version,
                "current_pRB_L1_cm": as_float(cur.get(f"pRB_t L1 cm {DOWN}")),
                "current_pRB_L2_cm": as_float(cur.get(f"pRB_t L2 cm {DOWN}")),
                "current_gR1_angle_deg": as_float(cur.get(f"gR1_t angle deg {DOWN}")),
                "next_p_L1_cm": as_float(next_row.get(f"pRB_t+1 L1 cm {DOWN}")),
                "next_p_L2_cm": as_float(next_row.get(f"pRB_t+1 L2 cm {DOWN}")),
                "next_gR1_angle_deg": as_float(next_row.get(f"gR1_t+1 angle deg {DOWN}")),
                "next_pd_L2_cm_s": as_float(dyn_row.get(f"pRB_vel L2 cm/s {DOWN}")),
                "next_pdd_L2_cm_s2": as_float(dyn_row.get(f"pRB_acc L2 cm/s^2 {DOWN}")),
                "current_control_pRB_L2_cm": as_float(ctl.get(f"current control pRB L2 cm {DOWN}")),
                "next_control_pRB_L2_cm": as_float(ctl.get(f"next control pRB L2 cm {DOWN}")),
                "last_preview_control_pRB_L2_cm": as_float(ctl.get(f"last preview control pRB L2 cm {DOWN}")),
            }
        )
    return out


def best_strong_metric(train_result):
    if not train_result:
        return None
    best = train_result.get("best", {})
    epochs = train_result.get("best_epochs", {})
    return {
        "best_p_pdot_pddot_strong": best.get("best_p_pdot_pddot_strong.pt"),
        "best_epoch": epochs.get("best_p_pdot_pddot_strong.pt"),
        "weights": train_result.get("weights"),
        "p_pdot_pddot_norm_scales": train_result.get("p_pdot_pddot_norm_scales"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL v6 next p/pd/pdd strong-supervision run.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    evals = {}
    compact_tables = {}
    for key, filename in EVAL_FILES.items():
        path = root / filename
        data = load_json(path)
        evals[key] = str(path) if data else None
        compact_tables[key] = compact_eval(data)

    out = {
        "status": "ok",
        "root": str(root),
        "experiment": "newpl_v6_next_p_pdot_pddot_strong",
        "protocol": {
            "loss_preset": "p_pdot_pddot_strong",
            "supervision": "decoded next_pl/next_pldot/next_plddot pRB[15] only",
            "normalization": "train-cache RMS scales for p, pd, and pdd",
            "selection": "best_p_pdot_pddot_strong.pt by validation next_pRB_norm_composite",
            "output_contract": "pRB[15]+gR1[3] remains unchanged for downstream IK/full pipeline",
            "current_output": "output['pl'] = pRB_t[15] + gR1_t[3]",
            "next_output": "output['next_pl'] = predicted pRB_{t+1}[15] + gR1_{t+1}[3]",
            "next_derivatives": "output['next_pldot'] and output['next_plddot'] are decoded from predicted next control via spline",
            "current_frame_warning": (
                "This experiment primarily supervises and selects next-frame decoded p/pdot/pddot; "
                "it does not by itself prove current-frame p/pdot/pddot accuracy."
            ),
            "current_frame_eval": "See current_p_pdot_pddot_eval/summary_current_p_pdot_pddot.md and eval_current_p_pdot_pddot_*.json",
            "full_pipeline_11_metrics": False,
        },
        "train": {
            "amass_pretrain": best_strong_metric(load_json(root / "amass_pretrain" / "train_result.json")),
            "dip_finetune": best_strong_metric(load_json(root / "dip_finetune" / "train_result.json")),
        },
        "eval_jsons": evals,
        "same_cache_tables": compact_tables,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
