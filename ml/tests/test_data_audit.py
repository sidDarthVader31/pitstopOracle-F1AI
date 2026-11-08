from __future__ import annotations

import pandas as pd

from f1ml.data_audit import audit_dataset, render_audit_markdown


def test_data_audit_reports_grain_status_and_source_timing(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "round": [1, 1, 1],
            "driver_id": ["a", "b", "c"],
            "status": ["Finished", "Lapped", "Retired"],
        }
    ).to_parquet(tmp_path / "raw" / "results.parquet", index=False)
    pd.DataFrame(
        {
            "season": [2025],
            "round": [1],
            "lat": [1.0],
            "lng": [2.0],
        }
    ).to_parquet(tmp_path / "raw" / "races.parquet", index=False)
    pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "round": [1, 1, 1],
            "driver_id": ["a", "b", "c"],
        }
    ).to_parquet(tmp_path / "processed" / "driver_race.parquet", index=False)

    audit = audit_dataset(tmp_path)
    markdown = render_audit_markdown(audit)

    assert audit["processed_duplicate_driver_race_keys"] == 0
    assert audit["classified_rows"] == 2
    assert audit["nonclassified_rows"] == 1
    assert "observed weather" in markdown
    assert "no corner map" in markdown
