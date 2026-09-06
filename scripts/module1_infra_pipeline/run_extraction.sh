#!/bin/bash
# Drives extract_one_report.py over every raw PDF under a real OS-level
# `timeout`, so a pypdf hang on one malformed PDF cannot block the batch --
# a Python-internal ProcessPoolExecutor timeout was tried first and proved
# unreliable on this Windows/git-bash setup (see extract_reports.py main()).
set -u
cd "$(dirname "$0")/../.."
RAW_DIR="data/module1_infra_source/raw_reports"
EXTRACTED_DIR="data/module1_infra_source/extracted"
TIMEOUT_SECS=180

mkdir -p "$EXTRACTED_DIR"
rm -f "$EXTRACTED_DIR"/*_summary_partial.json

for pdf in "$RAW_DIR"/*.pdf; do
    name=$(basename "$pdf")
    echo "Processing $name ..."
    timeout "$TIMEOUT_SECS" python scripts/module1_infra_pipeline/extract_one_report.py "$pdf"
    rc=$?
    if [ $rc -eq 124 ]; then
        stem="${name%.pdf}"
        echo "  FAILED: timed out after ${TIMEOUT_SECS}s (pypdf hang)"
        python -c "
import json, hashlib
from pathlib import Path
p = Path('$pdf')
h = hashlib.sha256(p.read_bytes()).hexdigest()
out = {
    'report_id': '$stem',
    'source_file': '$name',
    'sha256': h,
    'size_bytes': p.stat().st_size,
    'error': 'TimeoutError: extraction exceeded ${TIMEOUT_SECS}s',
}
Path('$EXTRACTED_DIR/${stem}_summary_partial.json').write_text(json.dumps(out, indent=2))
"
    fi
done

python -c "
import json
from pathlib import Path
extracted = Path('$EXTRACTED_DIR')
summaries = []
for f in sorted(extracted.glob('*_summary_partial.json')):
    summaries.append(json.loads(f.read_text(encoding='utf-8')))
    f.unlink()
(extracted / 'extraction_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
print(f'Wrote extraction_summary.json for {len(summaries)} reports.')
"
