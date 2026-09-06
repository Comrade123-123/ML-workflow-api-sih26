"""Drives extract_one_report.py over every raw PDF using subprocess.run's
own timeout + kill, which is the well-tested cross-platform mechanism for
this (unlike the git-bash `timeout` command and PowerShell's
Process.WaitForExit(ms), both of which were tried first in this session and
failed to reliably terminate a stuck child on this Windows setup).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_reports as er

TIMEOUT_SECS = 600  # generous: confirmed some 30MB+ reports legitimately take ~4-5 min


def main() -> None:
    er.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    for f in er.EXTRACTED_DIR.glob("*_summary_partial.json"):
        f.unlink()

    reports = sorted(er.RAW_DIR.glob("*.pdf"))
    for pdf_path in reports:
        print(f"Processing {pdf_path.name} ...", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent / "extract_one_report.py"), str(pdf_path)],
                timeout=TIMEOUT_SECS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  FAILED: timed out after {TIMEOUT_SECS}s (pypdf hang)", flush=True)
            summary = {
                "report_id": pdf_path.stem,
                "source_file": pdf_path.name,
                "sha256": er.sha256_of(pdf_path),
                "size_bytes": pdf_path.stat().st_size,
                "error": f"TimeoutError: extraction exceeded {TIMEOUT_SECS}s",
            }
            (er.EXTRACTED_DIR / f"{pdf_path.stem}_summary_partial.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

    summaries = []
    for f in sorted(er.EXTRACTED_DIR.glob("*_summary_partial.json")):
        summaries.append(json.loads(f.read_text(encoding="utf-8")))
        f.unlink()
    (er.EXTRACTED_DIR / "extraction_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote extraction_summary.json for {len(summaries)} reports.")


if __name__ == "__main__":
    main()
