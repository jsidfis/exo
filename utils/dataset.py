import hashlib
import io
import json
from pathlib import Path

"""File-local windows, explicit pressure provenance, and train-only scalers."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import build_features_single_csv, ANGLE_COLS, OUTPUT_COLS, PRESSURE_COLS


CSV_COLUMNS = ('time', 'hip_angle_l', 'hip_angle_r', 'hip_torque_l', 'hip_torque_r',
               'knee_angle_l', 'knee_angle_r', 'knee_torque_l', 'knee_torque_r',
               'insole_l_force_y', 'insole_r_force_y')






class TorqueDataset(Dataset):
    def __init__(self, records, feature_config, window=180, stride=30,
                 include_availability=False, max_windows=None, endpoint_window=None):
        endpoint_window = window if endpoint_window is None else endpoint_window
        if endpoint_window < window:
            raise ValueError('endpoint_window must cover the input window')
        self.records = records
        self.window = window
        self.include_availability = include_availability
        self.scaler = None
        self.frames, targets, ids, endpoints = [], [], [], []
        for rid, record in enumerate(records):
            with open(record['path'], encoding='utf-8-sig') as f:
                header = f.readline().strip()
            delimiter = ',' if ',' in header else '\t'
            if tuple(header.split(delimiter)) != CSV_COLUMNS:
                raise ValueError(f'Invalid CSV header or column order: {record["path"]}')
            values = np.loadtxt(record['path'], delimiter=delimiter, skiprows=1, ndmin=2)
            if values.shape[1] != 11 or not np.isfinite(values).all():
                raise ValueError(f'Invalid CSV shape or nonfinite values: {record["path"]}')
            steps = np.diff(values[:, 0])
            dt = feature_config.get('dt', 0.005)
            if not (steps > 0).all() or not np.allclose(steps, dt, rtol=1e-5, atol=1e-8):
                raise ValueError(f'Invalid timestamps: expected increasing {dt}s intervals: {record["path"]}')
            angle = values[:, ANGLE_COLS].astype(np.float32)
            pressure = np.clip(values[:, PRESSURE_COLS], 0, 1500).astype(np.float32)
            features = build_features_single_csv(angle, pressure, **feature_config)
            if not np.isfinite(features).all():
                raise ValueError(f'Nonfinite features: {record["path"]}')
            self.frames.append(features.astype(np.float32))
            end = np.arange(endpoint_window - 1, len(values), stride, dtype=np.int64)
            targets.append(values[end][:, OUTPUT_COLS])
            ids.extend([rid] * len(end))
            endpoints.extend(end.tolist())
        self.window_record_ids = np.asarray(ids, dtype=np.int64)
        self.window_endpoints = np.asarray(endpoints, dtype=np.int64)
        self.targets = np.concatenate(targets).astype(np.float32)
        if max_windows is not None and len(self.targets) > max_windows:
            selected = np.linspace(0, len(self.targets) - 1, max_windows, dtype=np.int64)
            self.window_record_ids = self.window_record_ids[selected]
            self.window_endpoints = self.window_endpoints[selected]
            self.targets = self.targets[selected]
        if not len(self.targets):
            raise ValueError('No complete windows in dataset')

    def fit_scaler(self):
        count = sum(len(f) for f in self.frames)
        sums = sum(np.sum(f, axis=0, dtype=np.float64) for f in self.frames)
        squares = sum(np.einsum('ij,ij->j', f, f, dtype=np.float64) for f in self.frames)
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
        return dict(input_mean=mean.astype(np.float32), input_std=np.maximum(std, 1e-6).astype(np.float32),
                    output_mean=self.targets.mean(axis=0, dtype=np.float64).astype(np.float32),
                    output_std=np.maximum(self.targets.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32))

    def set_scaler(self, scaler):
        self.scaler = {k: np.asarray(v, dtype=np.float32) for k, v in scaler.items()}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        rid, end = self.window_record_ids[index], self.window_endpoints[index]
        x = self.frames[rid][end + 1 - self.window:end + 1].copy()
        y = self.targets[index].copy()
        available = float(self.records[rid]['pressure_available'])
        if self.scaler is not None:
            x = (x - self.scaler['input_mean']) / self.scaler['input_std']
            y = (y - self.scaler['output_mean']) / self.scaler['output_std']
        if self.include_availability:
            x = np.concatenate([x, np.full((self.window, 2), available, dtype=np.float32)], axis=-1)
        return torch.from_numpy(x), torch.from_numpy(y), int(rid), available


class _CleanSourceMixin:
    def _load_source_validity(self, source_validity_manifest, data_root, cache_root):
        manifest_bytes = Path(source_validity_manifest).read_bytes()
        self.source_validity_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        entries = {row['corrected_csv'].casefold(): row
                   for row in json.loads(manifest_bytes.decode('utf-8-sig'))['records']}
        self.pressure_validity_frames, original_targets = [], []
        for record, frames in zip(self.records, self.frames):
            key = Path(record['path']).resolve().relative_to(Path(data_root).resolve()).as_posix().casefold()
            if key not in entries:
                raise ValueError(f'Missing original source record in validity manifest: {record["path"]}')
            entry = entries[key]
            corrected_bytes = Path(record['path']).read_bytes()
            if hashlib.sha256(corrected_bytes).hexdigest() != entry['corrected_sha256']:
                raise ValueError(f'Corrected CSV SHA256 mismatch: {record["path"]}')
            cache_bytes = (Path(cache_root) / entry['cache']).read_bytes()
            if hashlib.sha256(cache_bytes).hexdigest() != entry['cache_sha256']:
                raise ValueError(f'Original source cache SHA256 mismatch: {entry["cache"]}')
            with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as cache:
                targets = cache['original_targets']
                pressure = cache['original_pressure']
                relative_time = cache['relative_time']
            count = len(frames)
            if (entry['rows'] != count or targets.shape != (count, 4) or
                    pressure.shape != (count, 2) or relative_time.shape != (count,)):
                raise ValueError(f'Original source row count or shape mismatch: {entry["cache"]}')
            delimiter = ',' if b',' in corrected_bytes.split(b'\n', 1)[0] else '\t'
            corrected = np.loadtxt(io.BytesIO(corrected_bytes), delimiter=delimiter, skiprows=1,
                                   usecols=(0, 3, 4, 7, 8), ndmin=2)
            if not np.array_equal(relative_time, corrected[:, 0] - corrected[0, 0]):
                raise ValueError(f'Original source relative time alignment mismatch: {entry["cache"]}')
            finite_targets = np.isfinite(targets)
            if not np.array_equal(targets[finite_targets].astype(np.float32),
                                  corrected[:, 1:][finite_targets].astype(np.float32)):
                raise ValueError(f'Original finite target alignment mismatch: {entry["cache"]}')
            valid_pressure = np.isfinite(pressure)
            frames[:, 12:14] = np.where(valid_pressure, np.clip(pressure, 0, 1500), 0).astype(np.float32)
            self.pressure_validity_frames.append(valid_pressure)
            original_targets.append(np.where(finite_targets, targets, np.nan).astype(np.float32))
        targets = np.asarray([original_targets[rid][end] for rid, end in
                              zip(self.window_record_ids, self.window_endpoints)], dtype=np.float32)
        keep = np.isfinite(targets).any(axis=1)
        if not keep.any():
            raise ValueError('No endpoints with an originally finite target')
        self.targets = targets[keep]
        self.window_record_ids = self.window_record_ids[keep]
        self.window_endpoints = self.window_endpoints[keep]

    def fit_scaler(self):
        if not np.isfinite(self.targets).any(axis=0).all():
            raise ValueError('Each joint requires a finite training target to fit its scaler')
        scaler = super().fit_scaler()
        scaler['output_mean'] = np.nanmean(self.targets, axis=0, dtype=np.float64).astype(np.float32)
        std = np.nanstd(self.targets, axis=0, dtype=np.float64)
        scaler['output_std'] = np.maximum(std, 1e-6).astype(np.float32)
        return scaler

    def __getitem__(self, index):
        x, y, rid, available = super().__getitem__(index)
        if self.include_availability:
            end = self.window_endpoints[index]
            x[:, 14:16] = torch.from_numpy(self.pressure_validity_frames[rid][end + 1 - self.window:end + 1])
        return x, y, rid, available


class CleanTorqueDataset(_CleanSourceMixin, TorqueDataset):
    def __init__(self, records, feature_config, source_validity_manifest, data_root, cache_root, **dataset_kwargs):
        if not all(feature_config.get(key) for key in ('use_diff', 'use_ddiff', 'use_pressure')):
            raise ValueError('Clean torque requires the base 14-feature layout')
        dataset_kwargs.setdefault('include_availability', True)
        super().__init__(records, feature_config, **dataset_kwargs)
        self._load_source_validity(source_validity_manifest, data_root, cache_root)
