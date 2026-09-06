"""Extracts a single Flash Report PDF; invoked per-file by run_extraction.sh
under an OS-level `timeout` so one malformed PDF (pypdf hang) cannot block
the batch. Writes the same per-report JSON/text artifacts as
extract_reports.process_report(), plus this report's summary dict as its own
small JSON file (merged into extraction_summary.json by the shell driver).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_reports as er


def main() -> None:
    pdf_path = Path(sys.argv[1])
    try:
        summary = er.process_report(pdf_path)
    except Exception as exc:  # noqa: BLE001
        summary = {
            "report_id": pdf_path.stem,
            "source_file": pdf_path.name,
            "sha256": er.sha256_of(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "error": repr(exc),
        }
        print(f"  FAILED: {exc!r}")
    else:
        print(f"  pages={summary['num_pages']} chars={summary['extracted_chars']} "
              f"bracket_ids={summary['total_bracket_ids_in_report']} "
              f"completed={summary['completed_records_parsed']} "
              f"ongoing={summary['ongoing_records_parsed']}")

    out = er.EXTRACTED_DIR / f"{pdf_path.stem}_summary_partial.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
