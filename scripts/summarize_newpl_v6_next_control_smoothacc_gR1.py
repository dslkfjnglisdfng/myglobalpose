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


def load_json(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_from_table(data, table_name):
    if not data:
        return []
    rows = data.get(table_name, [])
    return rows if isinstance(rows, list) else []


def compact_current_rows(data):
    rows = []
    for row in rows_from_table(data, "current_frame_table"):
        rows.append(
            {
                "dataset": row.get("Dataset"),
                "version": row.get("Version"),
                "pRB_L1_cm": float_or_none(row.get(f"pRB_t L1 cm {DOWN}")),
                "pRB_L2_cm": float_or_none(row.get(f"pRB_t L2 cm {DOWN}")),
                "gR1_angle_deg": float_or_none(row.get(f"gR1_t angle deg {DOWN}")),
                "notes": row.get("Notes"),
            }
        )
    return rows


def compact_control_rows(data):
    rows = []
    for row in rows_from_table(data, "control_table"):
        rows.append(
            {
                "dataset": row.get("Dataset"),
                "version": row.get("Version"),
                "current_control_pRB_L2_cm": float_or_none(row.get(f"current control pRB L2 cm {DOWN}")),
                "current_control_gR1_angle_deg": float_or_none(row.get(f"current control gR1 angle deg {DOWN}")),
                "next_control_pRB_L2_cm": float_or_none(row.get(f"next control pRB L2 cm {DOWN}")),
                "next_control_gR1_angle_deg": float_or_none(row.get(f"next control gR1 angle deg {DOWN}")),
                "last_preview_control_pRB_L2_cm": float_or_none(row.get(f"last preview control pRB L2 cm {DOWN}")),
                "tail4_control_pRB_L2_cm": float_or_none(row.get(f"tail4 control pRB L2 cm {DOWN}")),
                "notes": row.get("Notes"),
            }
        )
    return rows


def best_by_gR1(rows):
    candidates = [row for row in rows if row.get("gR1_angle_deg") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["gR1_angle_deg"])


def best_new_candidate(rows):
    candidates = [
        row
        for row in rows
        if row.get("gR1_angle_deg") is not None
        and str(row.get("version", "")).startswith("newpl_v6")
        and "smoothacc" in str(row.get("version", ""))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["gR1_angle_deg"])


def comparison_summary(rows):
    best_all = best_by_gR1(rows)
    best_new = best_new_candidate(rows)
    official = next((row for row in rows if str(row.get("version", "")).startswith("official")), None)
    v4 = next((row for row in rows if "newpl_v4" in str(row.get("version", ""))), None)
    return {
        "best_all_by_current_gR1": best_all,
        "best_newpl_v6_smoothacc_by_current_gR1": best_new,
        "official_baseline": official,
        "newpl_v4_baseline": v4,
        "new_candidate_delta_vs_official_gR1_deg": (
            None
            if not best_new or not official or official.get("gR1_angle_deg") is None
            else best_new["gR1_angle_deg"] - official["gR1_angle_deg"]
        ),
        "new_candidate_delta_vs_v4_gR1_deg": (
            None
            if not best_new or not v4 or v4.get("gR1_angle_deg") is None
            else best_new["gR1_angle_deg"] - v4["gR1_angle_deg"]
        ),
    }


def read_train_result(path):
    data = load_json(path)
    if not data:
        return None
    return {
        "status": data.get("status"),
        "num_train_sequences": data.get("num_train_sequences"),
        "num_val_sequences": data.get("num_val_sequences"),
        "best": data.get("best"),
        "best_epochs": data.get("best_epochs"),
        "weights": data.get("weights"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize NewPL v6 next-control smoothacc gR1 experiment.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    evals = {}
    current_tables = {}
    control_tables = {}
    comparisons = {}
    for key, filename in EVAL_FILES.items():
        data = load_json(root / filename)
        evals[key] = str(root / filename) if data else None
        current_rows = compact_current_rows(data)
        control_rows = compact_control_rows(data)
        current_tables[key] = current_rows
        control_tables[key] = control_rows
        comparisons[key] = comparison_summary(current_rows)

    out = {
        "status": "ok",
        "root": str(root),
        "experiment": "newpl_v6_next_control_smoothacc_gR1_v1",
        "protocol": {
            "input": "smoothacc aM + raw wM/RMB, then official 84D PL feature",
            "output": "current pRB[15]+gR1[3] plus v6 auxiliary next-control outputs",
            "training": "AMASS pretrain -> DIP-IMU fine-tune; TotalCapture is eval-only",
            "full_pipeline_11_metrics": False,
            "dip_trans_used": False,
            "selection": "trainer saves best_current_gR1.pt, best_next_gR1.pt, best_gravity_control.pt, plus balanced/current/next/control checkpoints",
        },
        "train": {
            "amass_pretrain": read_train_result(root / "amass_pretrain" / "train_result.json"),
            "dip_finetune": read_train_result(root / "dip_finetune" / "train_result.json"),
        },
        "eval_jsons": evals,
        "current_frame_tables": current_tables,
        "control_tables": control_tables,
        "comparisons": comparisons,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
