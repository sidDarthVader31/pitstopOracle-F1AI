"""Write the local dataset quality report to reports/DATA_AUDIT.md."""

from f1ml.data_audit import audit_dataset, render_audit_markdown
from f1ml.paths import DATA, REPORTS


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "DATA_AUDIT.md"
    output.write_text(render_audit_markdown(audit_dataset(DATA)))
    print(output)


if __name__ == "__main__":
    main()
