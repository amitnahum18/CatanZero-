"""Board exclusions compatible with both historical JSON hash conventions."""
import hashlib
import json
from pathlib import Path


def board_hash_variants(document):
    return {hashlib.sha256(json.dumps(document, sort_keys=True, **kwargs).encode()).hexdigest()
            for kwargs in ({}, {'separators': (',', ':')})}


def normalization_source_board_hashes(export_directory):
    """Exclude all source boards, including holdouts, from fresh gameplay tests.

The data report, not a hard-coded dataset name, identifies raw source maps.
Missing provenance fails closed; no silent omission of normalization sources.
"""
    root = Path(export_directory)
    report = json.loads((root/'data_report.json').read_text(encoding='utf-8'))
    if not report.get('sources'):
        raise ValueError('Missing normalization dataset source provenance')
    hashes = set()
    for item in report['sources']:
        source = Path(item['source'])
        with (source/'maps.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                hashes.update(board_hash_variants(json.loads(line)['map']))
    return sorted(hashes)
