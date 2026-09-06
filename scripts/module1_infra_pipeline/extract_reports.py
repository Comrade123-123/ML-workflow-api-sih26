"""Extraction pipeline for MoSPI/IPMD/OCMS Flash Report PDFs.

Reads each downloaded Flash Report PDF (data/module1_infra_source/raw_reports/),
extracts text via pypdf, and parses two distinct tables using the project's
bracketed identifier (e.g. "[N16000289]") as the primary anchor -- this is
far more reliable than positional/line-based regexes given how pypdf's text
extraction wraps project names across lines inconsistently.

Produces two record types per report:
  - completed: from "Month wise List of Completed Projects..." sections
  - ongoing:   from "Annexure XVIII: Details of Ongoing Projects"

This is investigation/acquisition tooling, not production API code. Nothing
here is wired into the live API yet -- that only happens after Stage 5+
validation per the approved plan.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "raw_reports"
EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "extracted"

ID_PATTERN = re.compile(r"\[([A-Z]?\d{6,9})\]")
MONEY = r"[\d,]+\.\d{2}"
DATE = r"\d{1,2}/\d{4}"

# Highest genuine month-wise completed-project count confirmed across the
# whole successfully-parsed 2020-2024 OCMS corpus was 183 (Oct 2023 edition).
# Some pre-2021 report editions omit the "Month wise List of Deleted
# Projects" boundary this parser anchors on, causing the fallback "next
# Annexure" search to land hundreds of thousands of characters later and
# swallow unrelated tables -- confirmed for FR_2020_05 (1695 spurious
# records). Rather than silently ship a report's misaligned/over-scoped
# parse into the dataset, treat any count above this generous ceiling as a
# section-boundary detection failure for that report's edition and discard.
MAX_PLAUSIBLE_COMPLETED_RECORDS_PER_REPORT = 300


@dataclass
class ReportMeta:
    report_id: str
    source_file: str
    sha256: str
    size_bytes: int
    num_pages: int
    extracted_chars: int
    extraction_method: str


@dataclass
class CompletedProjectRecord:
    report_id: str
    project_id: str
    project_name: str
    sector: Optional[str]
    original_cost_cr: Optional[float]
    original_commissioning_date: Optional[str]
    cumulative_expenditure_cr: Optional[float]


@dataclass
class OngoingProjectRecord:
    report_id: str
    project_id: str
    project_name: str
    state: Optional[str]
    sector: Optional[str]
    approval_date: Optional[str]
    commissioning_original: Optional[str]
    commissioning_revised: Optional[str]
    commissioning_anticipated: Optional[str]
    cost_original_cr: Optional[float]
    cost_revised_cr: Optional[float]
    cost_anticipated_cr: Optional[float]
    cumulative_expenditure_cr: Optional[float]
    cost_overrun_cr: Optional[float]
    time_overrun_months: Optional[str]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def extract_text(pdf_path: Path) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    return text, len(reader.pages)


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


SECTOR_HEADERS = {
    "COAL", "PETROLEUM", "RAILWAYS", "POWER", "ROAD TRANSPORT AND HIGHWAYS",
    "ROAD TRANSPORT & HIGHWAYS", "WATER RESOURCES", "CIVIL AVIATION",
    "URBAN DEVELOPMENT", "STEEL", "MINES", "HEALTH AND FAMILY WELFARE",
    "TELECOMMUNICATIONS", "ATOMIC ENERGY", "HOME AFFAIRS", "FINANCE",
    "DPIIT", "SHIPPING AND PORTS", "SOCIAL JUSTICE", "DONER",
    "RENEWABLE ENERGY", "DEPARTMENT OF HIGHER EDUCATION",
}


def parse_completed_projects(text: str, report_id: str) -> list[CompletedProjectRecord]:
    """Parses the 'Month wise List of Completed Projects' tables.

    Strategy: locate the section, then walk line-by-line tracking the most
    recent all-caps sector header, and anchor each record on its bracketed
    ID -- pulling the project name from lines seen since the last record and
    the (cost, date, expenditure) triple from the lines immediately after.
    """
    # The literal string "Month wise List of Completed Projects" also appears
    # in the Table of Contents near the top of the document, immediately
    # followed by dot-leaders and a page number (no "during <year>" suffix).
    # Anchoring on the body-only phrasing ("...above during") skips the ToC
    # false match and lands on the actual table.
    start_match = re.search(r"Month wise List of Completed Projects[^\n]*?during", text)
    if not start_match:
        return []
    section_start = start_match.start()
    # The Completed Projects table is immediately followed by "Month wise
    # List of Deleted Projects" (confirmed in both 2021 and 2024 editions).
    # Stopping there -- rather than at the next literal "Annexure", which is
    # many unrelated tables further into the document (Deleted/Added
    # projects, overrun-extent tables, milestones, sector/state analyses) --
    # avoids misattributing those other tables' project rows as completions.
    end_match = re.search(r"Month wise List of Deleted Projects", text[section_start:])
    if end_match:
        section_end = section_start + end_match.start()
    else:
        section_end = text.find("Annexure", section_start)
        if section_end == -1:
            section_end = len(text)
    section = text[section_start:section_end]

    lines = [l.strip() for l in section.split("\n") if l.strip()]
    records: list[CompletedProjectRecord] = []
    current_sector = None
    pending_name_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.upper() in SECTOR_HEADERS:
            current_sector = line.upper()
            pending_name_lines = []
            i += 1
            continue

        id_match = ID_PATTERN.search(line)
        if id_match:
            project_id = id_match.group(1)
            name = " ".join(pending_name_lines).strip()
            # Strip a leading serial number from the very first name line, e.g. "12 PROJECT NAME"
            name = re.sub(r"^\d+\s+", "", name)
            pending_name_lines = []

            # Look ahead up to 3 lines for the cost/date/expenditure triple.
            triple = None
            for j in range(i, min(i + 4, len(lines))):
                m = re.search(rf"({MONEY})\s+({DATE})\s+({MONEY})", lines[j])
                if m:
                    triple = m
                    i = j
                    break
            records.append(
                CompletedProjectRecord(
                    report_id=report_id,
                    project_id=project_id,
                    project_name=name,
                    sector=current_sector,
                    original_cost_cr=_to_float(triple.group(1)) if triple else None,
                    original_commissioning_date=triple.group(2) if triple else None,
                    cumulative_expenditure_cr=_to_float(triple.group(3)) if triple else None,
                )
            )
            i += 1
            continue

        # Accumulate probable name text (skip pure section/page-header noise).
        if not re.match(r"^(Sl\.?\s*No|Original|Cumulative|Month wise|Ministry of|Page \d)", line, re.I):
            pending_name_lines.append(line)
        i += 1

    if len(records) > MAX_PLAUSIBLE_COMPLETED_RECORDS_PER_REPORT:
        print(
            f"  WARNING [{report_id}]: parsed {len(records)} completed-project "
            f"records, exceeding the plausibility ceiling of "
            f"{MAX_PLAUSIBLE_COMPLETED_RECORDS_PER_REPORT} -- treating as a "
            f"section-boundary detection failure for this report edition and "
            f"discarding (not including in the dataset)."
        )
        return []

    return records


def _parse_ongoing_value_block(window: str) -> dict:
    """Bracket-type-aware parser for the value block following a project ID
    in the Ongoing Projects table.

    The table's own layout is a fixed 3-tier structure per field:
    Original (plain) / (Revised, in parens) / [Anticipated, in brackets] --
    for BOTH the commissioning date and the cost, followed by a plain
    cumulative-expenditure figure and a final (cost overrun) / [time
    overrun] pair. Any tier can be blank ("-").

    The original implementation extracted "all dates" and "all costs" as
    flat lists and assigned them by absolute position (dates[0], dates[1],
    ...). That silently breaks whenever a tier is blank: a missing value
    isn't a placeholder in a flat re.findall() list, so every subsequent
    field shifts into the wrong slot. This version instead extracts each
    bracket TYPE (plain / paren / bracket) as its own ordered group, so a
    blank "-" in one tier consumes its own slot in that tier's group
    without disturbing the others -- this is the actual fix, not just a
    rename.
    """
    paren_values = re.findall(r"\(([^)]*)\)", window)
    bracket_values = re.findall(r"\[([^\]]*)\]", window)
    plain_text = re.sub(r"\([^)]*\)", " ", window)
    plain_text = re.sub(r"\[[^\]]*\]", " ", plain_text)
    plain_dates = re.findall(DATE, plain_text)
    plain_monies = re.findall(MONEY, plain_text)

    def _clean(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return None if v in ("", "-", "--") else v

    def _clean_money(v: Optional[str]) -> Optional[float]:
        v = _clean(v)
        return _to_float(v) if v is not None else None

    def _clean_int(v: Optional[str]) -> Optional[str]:
        v = _clean(v)
        if v is None:
            return None
        return v if re.fullmatch(r"-?\d+", v) else None

    # paren_values, in the table's fixed order: [revised_commissioning,
    # revised_cost, cost_overrun]. bracket_values: [anticipated_commissioning,
    # anticipated_cost, time_overrun].
    return {
        "approval_date": _clean(plain_dates[0]) if len(plain_dates) > 0 else None,
        "commissioning_original": _clean(plain_dates[1]) if len(plain_dates) > 1 else None,
        "commissioning_revised": _clean(paren_values[0]) if len(paren_values) > 0 else None,
        "commissioning_anticipated": _clean(bracket_values[0]) if len(bracket_values) > 0 else None,
        "cost_original_cr": _clean_money(plain_monies[0]) if len(plain_monies) > 0 else None,
        "cumulative_expenditure_cr": _clean_money(plain_monies[1]) if len(plain_monies) > 1 else None,
        "cost_revised_cr": _clean_money(paren_values[1]) if len(paren_values) > 1 else None,
        "cost_anticipated_cr": _clean_money(bracket_values[1]) if len(bracket_values) > 1 else None,
        "cost_overrun_cr": _clean_money(paren_values[2]) if len(paren_values) > 2 else None,
        "time_overrun_months": _clean_int(bracket_values[2]) if len(bracket_values) > 2 else None,
    }


def parse_ongoing_projects(text: str, report_id: str) -> list[OngoingProjectRecord]:
    """Parses 'Annexure XVIII: Details of Ongoing Projects'.

    Same ID-anchored strategy, tracking State then Sector headers (this
    table nests Sector under State), and pulling the bracketed
    Original/(Revised)/[Anticipated] triples for both the commissioning
    date and the cost from the block of lines following each ID.
    """
    # Same ToC-vs-body pitfall as the Completed Projects section: "Annexure
    # XVIII" also appears once in the Table of Contents (colon + dot-leaders
    # + page number, e.g. "...459"), well before the real body table. Since
    # the ToC lists it exactly once and the body heading appears exactly
    # once (confirmed by inspection), taking the LAST occurrence reliably
    # lands on the body rather than the ToC entry.
    section_start = text.rfind("Annexure  XVIII")
    if section_start == -1:
        section_start = text.rfind("Annexure XVIII")
    if section_start == -1:
        return []
    section = text[section_start:]

    lines = [l.strip() for l in section.split("\n") if l.strip()]
    records: list[OngoingProjectRecord] = []
    current_state = None
    current_sector = None
    pending_name_lines: list[str] = []

    state_like = re.compile(r"^[A-Z][A-Z &,]{4,60}$")

    i = 0
    while i < len(lines):
        line = lines[i]
        upper = line.upper()
        if upper in SECTOR_HEADERS:
            current_sector = upper
            pending_name_lines = []
            i += 1
            continue
        if state_like.match(line) and upper not in SECTOR_HEADERS and len(line.split()) <= 6:
            # Heuristic: a short all-caps line not matching a known sector is
            # treated as a state header (this table's structure has no
            # bracketed ID for state/sector header rows themselves).
            current_state = line
            pending_name_lines = []
            i += 1
            continue

        id_match = ID_PATTERN.search(line)
        if id_match:
            project_id = id_match.group(1)
            name = re.sub(r"^\d+\s+", "", " ".join(pending_name_lines).strip())
            pending_name_lines = []

            window_lines = lines[i : min(i + 12, len(lines))]
            window = "\n".join(window_lines)
            # Only consider text AFTER the ID's own "[...]" -- otherwise the
            # ID bracket itself would be misread as the first bracket-group
            # (anticipated) value.
            window_after_id = window[window.find(f"[{project_id}]") + len(project_id) + 2 :]

            # Find which subsequent line (if any) starts the NEXT project's
            # ID, so we know exactly how many lines this record's value
            # block occupies -- both to keep that next ID's fields out of
            # window_after_id, AND (critically) to advance `i` past all of
            # them. Advancing by only 1 line here was the root cause of the
            # project_name contamination bug: the unconsumed value-block
            # lines (dates/costs) were being re-read as name text for the
            # *next* record.
            next_id_line_offset = None
            for offset in range(1, len(window_lines)):
                if ID_PATTERN.search(window_lines[offset]):
                    next_id_line_offset = offset
                    break

            if next_id_line_offset is not None:
                window_after_id = "\n".join(window_lines[:next_id_line_offset])[
                    window.find(f"[{project_id}]") + len(project_id) + 2 :
                ]

            record = _parse_ongoing_value_block(window_after_id)
            records.append(
                OngoingProjectRecord(
                    report_id=report_id,
                    project_id=project_id,
                    project_name=name,
                    state=current_state,
                    sector=current_sector,
                    **record,
                )
            )
            i += next_id_line_offset if next_id_line_offset is not None else 1
            continue

        if not re.match(r"^(SI\.?No|Project\b|Date of|Original|Cumulative|Expenditure|Cost|\(|\[|Page \d)", line, re.I):
            pending_name_lines.append(line)
        i += 1

    return records


@dataclass
class CompletedProjectRecordV2:
    """Schema for the newer 'PAIMANA' report format (confirmed present from
    July 2025 onward), which replaced the older OCMS Flash Report layout.
    Richer than the v1 schema: includes a real Date of Approval, State, and
    a separate Revised Cost -- none of which exist in v1. Uses the report's
    own plain numeric "Project Code" (e.g. 609041), a different, non-
    overlapping ID namespace from v1's bracketed alphanumeric IDs
    (e.g. N16000289) -- these are two internally-stable but mutually
    incompatible identifier systems, never fuzzy-merged."""

    report_id: str
    project_code: str
    project_name: str
    sector: Optional[str]
    ministry: Optional[str]
    state: Optional[str]
    approval_date: Optional[str]
    original_cost_cr: Optional[float]
    revised_cost_cr: Optional[float]
    cumulative_expenditure_cr: Optional[float]


def parse_completed_projects_v2(text: str, report_id: str) -> list[CompletedProjectRecordV2]:
    """EXPERIMENTAL -- NOT wired into process_report()/main(), and NOT used
    to build Dataset A. Confirmed real bugs remain (header-line detection
    misfires on multi-word Title Case project names, e.g. "Durgawati
    Reservoir Project" gets misread as a ministry/sector header), verified
    by manual inspection against the September 2025 report's own text. Kept
    here, unused, as a documented, honest record of what was attempted:
    the newer PAIMANA-format reports (July 2025 onward) DO contain richer
    approval-time fields (State, real Date of Approval, separate Revised
    Cost) and are a genuine, real data source -- but this parser is not yet
    reliable enough to trust its output, so none of its records are
    included in Dataset A. Fixing it properly is future work, not done
    here, rather than shipping partially-misaligned rows.

    Parses Table 3: 'Completed Projects During Month' from the PAIMANA-
    format reports. Project-code-anchored (bare 6-digit number on its own
    line, immediately after the "N Name (Agency)" line), matching the same
    defensive philosophy as the v1 ongoing-projects parser: extract paren-
    groups and plain values as separate ordered sequences rather than a
    flat mixed list, so a blank/absent tier doesn't shift later fields.
    """
    start_match = re.search(r"Completed Projects During Month", text)
    if not start_match:
        return []
    section_start = start_match.start()
    end_match = re.search(r"Table \d+: Newly Added Projects|Newly Added Projects\s*\n\s*[A-Z]+ \d{4}", text[section_start + 30 :])
    section_end = section_start + 30 + end_match.start() if end_match else len(text)
    section = text[section_start:section_end]

    lines = [l.strip() for l in section.split("\n") if l.strip()]
    code_pattern = re.compile(r"^\d{5,7}$")
    records: list[CompletedProjectRecordV2] = []
    current_ministry = None
    current_sector = None
    pending_name_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if code_pattern.match(line):
            name = re.sub(r"^\d+\s+", "", " ".join(pending_name_lines).strip())
            pending_name_lines = []
            window_lines = lines[i + 1 : min(i + 14, len(lines))]
            # Stop the window at the next project code or "Total" line.
            stop = len(window_lines)
            for j, wl in enumerate(window_lines):
                if code_pattern.match(wl) or wl.startswith("Total"):
                    stop = j
                    break
            window = "\n".join(window_lines[:stop])

            state_match = re.match(r"^([A-Za-z][A-Za-z &,]+)$", window_lines[0]) if window_lines else None
            state = state_match.group(1).strip() if state_match else None

            paren_values = re.findall(r"\(([^)]*)\)", window)
            plain_dates = re.findall(DATE, re.sub(r"\([^)]*\)", " ", window))
            plain_monies = re.findall(MONEY, re.sub(r"\([^)]*\)", " ", window))

            def _clean(v):
                v = (v or "").strip()
                return None if v in ("", "-", "--") else v

            records.append(
                CompletedProjectRecordV2(
                    report_id=report_id,
                    project_code=line,
                    project_name=name,
                    sector=current_sector,
                    ministry=current_ministry,
                    state=state,
                    approval_date=_clean(plain_dates[0]) if plain_dates else None,
                    original_cost_cr=_to_float(_clean(plain_monies[0])) if plain_monies else None,
                    revised_cost_cr=_to_float(_clean(paren_values[-2])) if len(paren_values) >= 2 else None,
                    cumulative_expenditure_cr=_to_float(_clean(plain_monies[1])) if len(plain_monies) > 1 else None,
                )
            )
            i += 1 + stop
            continue

        if re.match(r"^[A-Z][a-zA-Z &,]+$", line) and i + 1 < len(lines) and not code_pattern.match(lines[i + 1]):
            # Two consecutive short title-case/caps lines before a numbered
            # project row are Ministry then Sector (confirmed by inspection:
            # "Department of Higher Education" / "Education").
            if re.match(r"^\d+\s+", lines[i + 1] if i + 1 < len(lines) else ""):
                current_sector = line
            else:
                current_ministry = line
            pending_name_lines = []
            i += 1
            continue

        if not re.match(r"^(Sl\.?No|Project Name|State|Date of|Orignal|Original|Cumulative|Total|Note:|Page \d)", line, re.I):
            pending_name_lines.append(line)
        i += 1

    return records


def process_report(pdf_path: Path) -> dict:
    report_id = pdf_path.stem
    text, num_pages = extract_text(pdf_path)
    completed = parse_completed_projects(text, report_id)
    ongoing = parse_ongoing_projects(text, report_id)

    meta = ReportMeta(
        report_id=report_id,
        source_file=pdf_path.name,
        sha256=sha256_of(pdf_path),
        size_bytes=pdf_path.stat().st_size,
        num_pages=num_pages,
        extracted_chars=len(text),
        extraction_method="pypdf.extract_text",
    )

    all_ids = set(m.group(1) for m in ID_PATTERN.finditer(text))

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXTRACTED_DIR / f"{report_id}_raw_text.txt").write_text(text, encoding="utf-8")
    (EXTRACTED_DIR / f"{report_id}_completed.json").write_text(
        json.dumps([asdict(r) for r in completed], indent=2), encoding="utf-8"
    )
    (EXTRACTED_DIR / f"{report_id}_ongoing.json").write_text(
        json.dumps([asdict(r) for r in ongoing], indent=2), encoding="utf-8"
    )

    summary = {
        **asdict(meta),
        "total_bracket_ids_in_report": len(all_ids),
        "completed_records_parsed": len(completed),
        "ongoing_records_parsed": len(ongoing),
    }
    return summary


def main() -> None:
    # NOTE: an in-process ProcessPoolExecutor with future.result(timeout=...)
    # was tried here to guard against a pypdf hang, but on this Windows/
    # git-bash setup it did not reliably interrupt a stuck worker (observed
    # ~11 minutes with only ~19s of CPU consumed and zero files produced).
    # The reliable guard is scripts/module1_infra_pipeline/run_extraction.sh,
    # which drives extract_one_report.py per-file under a real OS-level
    # `timeout`, and merges the resulting *_summary_partial.json files. This
    # main() remains for direct single-process use when every PDF is known-good.
    reports = sorted(RAW_DIR.glob("*.pdf"))
    summaries = []
    for pdf_path in reports:
        print(f"Processing {pdf_path.name} ...")
        try:
            summary = process_report(pdf_path)
        except Exception as exc:  # noqa: BLE001 -- one bad PDF must not abort the batch
            print(f"  FAILED: {exc!r}")
            summaries.append(
                {
                    "report_id": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "sha256": sha256_of(pdf_path),
                    "size_bytes": pdf_path.stat().st_size,
                    "error": repr(exc),
                }
            )
            continue
        summaries.append(summary)
        print(f"  pages={summary['num_pages']} chars={summary['extracted_chars']} "
              f"bracket_ids={summary['total_bracket_ids_in_report']} "
              f"completed={summary['completed_records_parsed']} "
              f"ongoing={summary['ongoing_records_parsed']}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXTRACTED_DIR / "extraction_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote extraction_summary.json for {len(summaries)} reports.")


if __name__ == "__main__":
    main()
