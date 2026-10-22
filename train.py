"""Train the original 16-input parallel TCN/SSM on fixed subject-disjoint splits."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from model import create_model
from loss import supervised_mse
from utils.dataset import CleanTorqueDataset
from utils.metrics import grouped_metrics
from utils.split import audit_splits, build_records
from utils.seed import seed_everything
from utils.checkpoint import portable_records, save_json, sha256, snapshot_sources

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def make_dataset(records, config, manifest, data_root, cache_root, max_windows=None):
    return CleanTorqueDataset(
        records, config['features'], manifest, data_root, cache_root,
        window=config['window'], stride=config['stride'],
        include_availability=config['include_availability'],
        endpoint_window=config['endpoint_window'], max_windows=max_windows)


def make_optimizer(model, config):
    if config['optimizer'] != 'Adam':
        raise ValueError('The supplied training protocol uses Adam')
    return torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = []
    for x, _, _, _ in loader:
        prediction = model(x.to(device))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite model prediction during validation')
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs)


def evaluate(model, dataset, device, batch_size, output):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scaler = dataset.scaler
    predictions = predict(model, loader, device)
    predictions = predictions * scaler['output_std'] + scaler['output_mean']
    result = grouped_metrics(dataset, predictions)
    result['parameters'] = sum(p.numel() for p in model.parameters())
    result['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    np.savez_compressed(output / 'predictions.npz', predictions=predictions,
                        targets=dataset.targets, valid_targets=np.isfinite(dataset.targets),
                        record_ids=dataset.window_record_ids, endpoints=dataset.window_endpoints)
    save_json(output / 'metrics.json', result)
    print(json.dumps({key: result[key] for key in ('avg_r2', 'avg_rmse', 'avg_mae', 'subject_macro_r2')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/online.yaml'))
    parser.add_argument('--run-dir', type=Path, help='New output directory, relative to this repository')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-train-windows', type=int)
    parser.add_argument('--max-val-windows', type=int)
    args = parser.parse_args()
    for name in ('epochs', 'batch_size', 'threads', 'max_train_windows', 'max_val_windows'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8'))
    if (not config['include_availability'] or config['pressure_dropout'] != 0
            or config['grad_clip_norm'] is not None):
        raise ValueError('This protocol requires availability flags, no pressure dropout and no gradient clipping')
    limited = (args.epochs is not None and args.epochs != config['epochs']) or any(
        value is not None for value in (args.max_train_windows, args.max_val_windows))
    if args.data_root is not None:
        config['data_root'] = str(args.data_root)
    if args.cache_root is not None:
        config['cache_root'] = str(args.cache_root)
    config.update(seed=args.seed, epochs=args.epochs or config['epochs'],
                  batch_size=args.batch_size or config['batch_size'],
                  max_train_windows=args.max_train_windows, max_val_windows=args.max_val_windows)
    data_root, cache_root = project_path(config['data_root']), project_path(config['cache_root'])
    split_dir = project_path(config['split_dir'])
    metadata = project_path(config['metadata_csv'])
    manifest = project_path(config['source_validity_manifest'])
    splits = {s: build_records(split_dir / f'{s}.txt', data_root, metadata) for s in ('train', 'val')}
    audit = audit_splits(splits)
    run = project_path(args.run_dir or Path('result') / (
        f'{args.config.stem}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'))
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {run}')
    run.mkdir(parents=True, exist_ok=True)
    frozen_manifest = run / 'source_validity_manifest.json'
    frozen_manifest.write_bytes(manifest.read_bytes())
    save_json(run / 'config.json', config)
    save_json(run / 'records.json', portable_records(splits, data_root))
    save_json(run / 'split_audit.json', audit)
    save_json(run / 'execution.json', dict(python=sys.version, torch=str(torch.__version__),
                                          numpy=np.__version__, device=str(device), threads=args.threads,
                                          limited_run=limited))
    snapshot_sources(ROOT, run)
    seed_everything(args.seed, device)
    print('Loading training and validation records.', flush=True)
    train = make_dataset(splits['train'], config, frozen_manifest, data_root, cache_root, args.max_train_windows)
    val = make_dataset(splits['val'], config, frozen_manifest, data_root, cache_root, args.max_val_windows)
    scaler = train.fit_scaler()
    train.set_scaler(scaler)
    val.set_scaler(scaler)
    np.savez(run / 'scaler.npz', **scaler)
    save_json(run / 'data_provenance.json', {
        'artifact_sha256': {name: sha256(run / name) for name in (
            'config.json', 'records.json', 'scaler.npz', 'source_validity_manifest.json')},
        'split_sha256': {s: sha256(split_dir / f'{s}.txt') for s in splits},
        'metadata_sha256': sha256(metadata)})
    model = create_model(config).to(device)
    save_json(run / 'initialization.json', dict(seed=args.seed, parameters=sum(p.numel() for p in model.parameters())))
    print(f'Train={len(train)}, val={len(val)}, params={sum(p.numel() for p in model.parameters())}', flush=True)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config['scheduler_patience'], factor=0.5)
    target = (val.targets - scaler['output_mean']) / scaler['output_std']
    best, stale, log = float('inf'), 0, []
    for epoch in range(1, config['epochs'] + 1):
        start = time.perf_counter()
        model.train()
        total, count, supervised_values = 0.0, 0, 0
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_mse(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            valid_values = int(torch.isfinite(y).sum())
            total += loss.item() * valid_values
            supervised_values += valid_values
            count += len(x)
        predictions = predict(model, val_loader, device)
        val_mse = float(np.nanmean(np.square(predictions - target)))
        scheduler.step(val_mse)
        entry = dict(epoch=epoch, train_mse=total / supervised_values, val_mse=val_mse,
                     seconds=time.perf_counter() - start, lr=optimizer.param_groups[0]['lr'],
                     windows=count, updates=len(loader), supervised_values=supervised_values)
        log.append(entry)
        save_json(run / 'history.json', log)
        with (run / 'train.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)
        if val_mse < best:
            best, stale = val_mse, 0
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, val_mse=best, config=config),
                       run / 'best_model.pt')
        else:
            stale += 1
        if stale >= config['patience']:
            break
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), epoch=epoch, val_mse=val_mse, config=config),
               run / 'last_model.pt')
    checkpoint = torch.load(run / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = run / 'val_evaluation'
    output.mkdir()
    evaluate(model, val, device, config['batch_size'], output)
    save_json(run / 'completed.json', dict(best_epoch=checkpoint['epoch'], best_val_mse=best,
                                         epochs_run=len(log), seconds=sum(r['seconds'] for r in log),
                                         training_windows=len(train), validation_windows=len(val),
                                         updates=sum(r['updates'] for r in log), limited_run=limited))


if __name__ == '__main__':
    main()
