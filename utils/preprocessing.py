"""The online Butterworth and offline Gaussian feature pipelines."""
import numpy as np

ANGLE_COLS = [1, 2, 5, 6]
PRESSURE_COLS = [9, 10]
OUTPUT_COLS = [3, 4, 7, 8]


def build_features_single_csv(angle, pressure, use_diff, use_ddiff, use_pressure,
                              diff_method, dt=0.005, butter_cutoff=10.0,
                              butter_order=4, gauss_sigma=3.0):
    if not (use_diff and use_ddiff and use_pressure):
        raise ValueError('The model requires angles, first/second derivatives and pressure')
    angle_base = np.zeros_like(angle, dtype=np.float64)
    if diff_method == 'butterworth_causal':
        from scipy.signal import butter, lfilter
        fs = 1.0 / dt
        nyq = fs / 2.0
        normalized_cutoff = min(butter_cutoff / nyq, 0.99)
        b_filt, a_filt = butter(butter_order, normalized_cutoff, btype='low')
        for col in range(angle.shape[1]):
            angle_base[:, col] = lfilter(b_filt, a_filt, angle[:, col].astype(np.float64))
        angle_base = angle_base.astype(np.float32)
        dx = np.zeros_like(angle_base)
        dx[1:, :] = (angle_base[1:, :] - angle_base[:-1, :]) / dt
        ddx = np.zeros_like(angle_base)
        ddx[1:, :] = (dx[1:, :] - dx[:-1, :]) / dt
    elif diff_method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        for col in range(angle.shape[1]):
            angle_base[:, col] = gaussian_filter1d(angle[:, col].astype(np.float64), sigma=gauss_sigma)
        angle_base = angle_base.astype(np.float32)
        dx = np.gradient(angle_base, dt, axis=0)
        ddx = np.gradient(dx, dt, axis=0)
    else:
        raise ValueError(f'Unsupported preprocessing method: {diff_method}')
    return np.concatenate([angle, dx.astype(np.float32), ddx.astype(np.float32), pressure], axis=-1)
