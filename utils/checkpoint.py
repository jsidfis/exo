"""Portable records and integrity checks for saved training artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_records(splits, data_root):
    return {split: [dict(record, path=Path(record['path']).resolve().relative_to(
        Path(data_root).resolve()).as_posix()) for record in records]
        for split, records in splits.items()}


def resolve_records(records, data_root):
    return [dict(record, path=str((Path(data_root) / record['path']).resolve())) for record in records]


def snapshot_sources(root, run):
    paths = [root / name for name in ('train.py', 'val.py', 'loss.py')]
    for package in ('model', 'utils'):
        paths.extend((root / package).glob('*.py'))
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        target = run / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[relative.as_posix()] = sha256(target)
    save_json(run / 'provenance.json', hashes)


def frozen_runner(run):
    snapshot = run / 'source_snapshot'
    sources = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    for name, digest in sources.items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f'Saved source changed: {name}')
    sys.dont_write_bytecode = True
    for name in list(sys.modules):
        if name.split('.')[0] in {'model', 'utils', 'train', 'val', 'loss'}:
            del sys.modules[name]
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location('_saved_training', snapshot / 'train.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner
