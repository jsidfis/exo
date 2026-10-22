"""Validate a saved run using its frozen model source, best weights and training scaler."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.checkpoint import frozen_runner, resolve_records, save_json, sha256

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('threads and batch-size must be positive')
    torch.set_num_threads(args.threads)
    run, output = project_path(args.run_dir), project_path(args.output_dir)
    if output == run or output.is_relative_to(run):
        parser.error('output-dir must be outside the saved run')
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((run / 'data_provenance.json').read_text(encoding='utf-8'))
    for name, digest in provenance['artifact_sha256'].items():
        if sha256(run / name) != digest:
            raise ValueError(f'Saved training artifact changed: {name}')
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    records = json.loads((run / 'records.json').read_text(encoding='utf-8'))
    data_root = project_path(args.data_root or config['data_root'])
    cache_root = project_path(args.cache_root or config['cache_root'])
    device = torch.device(args.device)
    checkpoint = torch.load(run / 'best_model.pt', map_location=device, weights_only=True)
    if checkpoint['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    runner = frozen_runner(run)
    model = runner.create_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    dataset = runner.make_dataset(resolve_records(records['val'], data_root), config,
                                  run / 'source_validity_manifest.json', data_root, cache_root,
                                  config.get('max_val_windows'))
    with np.load(run / 'scaler.npz', allow_pickle=False) as saved:
        dataset.set_scaler({name: saved[name] for name in saved.files})
    output.mkdir(parents=True)
    runner.evaluate(model, dataset, device, args.batch_size or config['batch_size'], output)
    save_json(output / 'report.json', dict(
        split='val', checkpoint_epoch=checkpoint['epoch'],
        checkpoint_sha256=sha256(run / 'best_model.pt'),
        scaler_sha256=sha256(run / 'scaler.npz'),
        source_validity_sha256=sha256(run / 'source_validity_manifest.json'),
        validation_windows=len(dataset), valid_targets_per_joint=np.isfinite(dataset.targets).sum(axis=0).tolist(),
        limited_run=json.loads((run / 'execution.json').read_text(encoding='utf-8'))['limited_run'],
        device=str(device), torch=str(torch.__version__), numpy=np.__version__))


if __name__ == '__main__':
    main()
