"""Original record mapping and subject-disjoint split audit."""
import csv
from pathlib import Path

def build_records(manifest_path, data_root, metadata_csv):
    with open(metadata_csv, encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['status'] == 'ok']
    metadata = {(r['phase'], r['subject'], r['trial']): r for r in rows}
    if len(metadata) != len(rows):
        raise ValueError('duplicate pressure metadata keys')
    records = []
    for line in Path(manifest_path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        parts = line.strip().replace('\\', '/').split('/')
        start = next((i for i, p in enumerate(parts) if p in ('Phase1And2_Complete', 'Phase3_Complete')), None)
        if start is None:
            raise ValueError(f'Cannot resolve dataset path: {line}')
        suffix = parts[start:]
        if len(suffix) != 5 or suffix[1] != 'Complete':
            raise ValueError(f'Unexpected dataset path: {line}')
        path = Path(data_root).joinpath(*suffix).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = suffix[0], suffix[2], suffix[3]
        row = metadata[key]
        source = row['pressure_source']
        records.append(dict(path=str(path), subject=key[1], trial=key[2],
                            pressure_source=source,
                            pressure_available=source != 'zero_fill_no_pressure_source'))
    if not records:
        raise ValueError(f'Empty manifest: {manifest_path}')
    return records

def audit_splits(splits):
    used_paths, used_subjects, summary = {}, {}, {}
    for split, records in splits.items():
        subjects = sorted({r['subject'] for r in records})
        for r in records:
            key = str(Path(r['path']).resolve()).casefold()
            if key in used_paths:
                raise ValueError(f'duplicate file in {split} and {used_paths[key]}: {key}')
            used_paths[key] = split
        for subject in subjects:
            if subject in used_subjects:
                raise ValueError(f'subject overlap: {subject} in {split} and {used_subjects[subject]}')
            used_subjects[subject] = split
        summary[split] = dict(files=len(records), subjects=subjects,
                              missing_pressure_files=sum(not r['pressure_available'] for r in records))
    return summary
