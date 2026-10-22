"""Original physical-unit metrics and participant/source aggregation."""
import numpy as np

def regression_metrics(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target)
    if not valid.all():
        joints = [regression_metrics(target[keep, j:j + 1], prediction[keep, j:j + 1]) if keep.any() else None
                  for j, keep in enumerate(valid.T)]
        result = dict(n=len(target), n_per_joint=valid.sum(axis=0).tolist())
        for key in ('r2', 'rmse', 'mae'):
            values = [m[f'avg_{key}'] if m is not None else None for m in joints]
            result[key] = values
            observed = [v for v in values if v is not None]
            result[f'avg_{key}'] = float(np.mean(observed)) if observed else None
        return result
    errors = prediction - target
    mse = np.mean(errors ** 2, axis=0)
    variance = np.var(target, axis=0)
    r2 = 1 - mse / np.maximum(variance, 1e-15)
    return dict(n=len(target), avg_r2=float(r2.mean()), avg_rmse=float(np.sqrt(mse).mean()),
                avg_mae=float(np.abs(errors).mean()), r2=r2.tolist(), rmse=np.sqrt(mse).tolist(),
                mae=np.abs(errors).mean(axis=0).tolist())

def grouped_metrics(dataset, prediction):
    result = regression_metrics(dataset.targets, prediction)
    record_ids = dataset.window_record_ids
    subjects = np.asarray([dataset.records[r]['subject'] for r in record_ids])
    availability = np.asarray([dataset.records[r]['pressure_available'] for r in record_ids])
    result['subjects'] = {s: regression_metrics(dataset.targets[subjects == s], prediction[subjects == s])
                          for s in np.unique(subjects)}
    result['subject_macro_r2'] = float(np.mean([m['avg_r2'] for m in result['subjects'].values()]))
    for label, keep in [('pressure_available', availability), ('pressure_missing', ~availability)]:
        if keep.any():
            result[label] = regression_metrics(dataset.targets[keep], prediction[keep])
    return result
