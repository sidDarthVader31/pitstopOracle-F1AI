"""Evaluate cautious race-level Safety Car/VSC probability baselines."""

import json

import pandas as pd

from f1ml.paths import RAW, REPORTS
from f1ml.safety_car import add_prior_circuit_safety_features, evaluate_safety_event_model


def main() -> None:
    labels = pd.read_parquet(RAW / "openf1_safety_car_labels.parquet")
    sessions = pd.read_parquet(RAW / "openf1_race_sessions.parquet")
    circuit_type = sessions[["season", "round", "circuit_type"]].drop_duplicates(
        ["season", "round"]
    )
    labels = labels.merge(circuit_type, on=["season", "round"], how="left", validate="one_to_one")
    labels["circuit_street"] = labels["circuit_type"].astype("string").str.contains(
        "street", case=False, na=pd.NA
    ).astype("Float64").astype(float)
    labels["round"] = labels["round"].astype(float)
    labels["regs_era"] = (labels["season"] >= 2026).astype(float)
    labels = add_prior_circuit_safety_features(labels, labels)

    metrics = {
        "source": "OpenF1 race_control, 2023 onward",
        "n_races": int(len(labels)),
        "sc_events": int(labels["has_sc"].sum()),
        "vsc_events": int(labels["has_vsc"].sum()),
        "sc": evaluate_safety_event_model(labels, "has_sc"),
        "vsc": evaluate_safety_event_model(labels, "has_vsc"),
        "artifacts_written": False,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "SAFETY_CAR_METRICS.json").write_text(json.dumps(metrics, indent=2))
    lines = [
        "# Safety Car / VSC probability diagnostic",
        "",
        "Source: OpenF1 historical race-control messages (2023 onward). Labels count deployed SC/VSC messages; ending messages do not count as deployment.",
        "The model is a race-level logistic regression using only prior circuit SC/VSC rates, sourced street-circuit type, round, and regulation era. It is an experimental diagnostic and is not promoted or integrated into the race simulator.",
        "",
        f"- Races labeled: {metrics['n_races']}",
        f"- SC events: {metrics['sc_events']}; VSC events: {metrics['vsc_events']}",
        "",
    ]
    for name, result in (("Safety Car", metrics["sc"]), ("Virtual Safety Car", metrics["vsc"])):
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Walk-forward races: {result.get('n_races', 0)}",
                f"- Positive rate: {result.get('positive_rate', float('nan')):.1%}",
                f"- Model log loss / Brier: {result.get('log_loss', float('nan')):.3f} / {result.get('brier', float('nan')):.3f}",
                f"- Expanding prevalence baseline log loss / Brier: {result.get('baseline_log_loss', float('nan')):.3f} / {result.get('baseline_brier', float('nan')):.3f}",
                f"- PR-AUC / ROC-AUC: {result.get('pr_auc', float('nan')):.3f} / {result.get('roc_auc', float('nan')):.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "No model artifact is written. Keep SC/VSC probabilities out of the production simulator unless the walk-forward scores and calibration justify their use.",
            "",
        ]
    )
    (REPORTS / "SAFETY_CAR_EVAL.md").write_text("\n".join(lines))
    print(REPORTS / "SAFETY_CAR_EVAL.md")


if __name__ == "__main__":
    main()
