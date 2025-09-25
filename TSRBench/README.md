# TSRBench: Time Series Robustness Benchmark

A framework for evaluating time series forecasting robustness under realistic, statistically-grounded corruptions. TSRBench injects two canonical corruption types into time series data using Extreme Value Theory (EVT):

- **Level Shift**: Persistent changes in the baseline level
- **Exponential Spike**: Transient spikes with exponential decay

## Quick Start

### Generate corrupted datasets for all benchmarks

```bash
python tsrbench.py
```

This generates corrupted versions of ETTh1, ETTh2, ETTm1, ETTm2, Electricity, and Weather datasets at 5 severity levels, producing files like:
```
{dataset}_level_{1-5}_type_shift.csv
{dataset}_level_{1-5}_type_spike.csv
{dataset}_level_{1-5}_type_combined.csv
```

### Basic API usage

```python
from contaminated_ts import CollectiveNoise
import numpy as np

cn = CollectiveNoise(seed=2025)

X = np.random.randn(1000)

# Inject level shift noise at severity level 3
noise = cn.inject_level_shift(X, noise_level=3)
noisy_signal = X + noise

# Inject both types of noise
shift_noise, spike_noise = cn.inject_noise(X, noise_level=3)
combined_signal = X + shift_noise + spike_noise
```

## Noise Parameters

5 severity levels with predefined frequency, duration, and amplitude:

| Level | Frequency | Duration | Amplitude |
|-------|-----------|----------|-----------|
| 1     | 0.002     | 6        | 0.0016    |
| 2     | 0.004     | 9        | 0.0016    |
| 3     | 0.004     | 12       | 0.0004    |
| 4     | 0.008     | 12       | 0.0004    |
| 5     | 0.008     | 15       | 0.0001    |

- **Frequency**: Poisson rate controlling how often anomalies occur
- **Duration**: Geometric distribution parameter for anomaly length
- **Amplitude**: SPOT algorithm quantile parameter controlling anomaly magnitude

## Using Custom Datasets

TSRBench can be applied to any time series dataset in CSV format. The expected input format is:

```
date,feature1,feature2,feature3,...
2020-01-01,1.23,4.56,7.89,...
2020-01-02,1.24,4.57,7.90,...
```

The first column should be a date/index column. All remaining columns are treated as feature columns and will have noise injected independently.

### Method 1: Using `make_noise_datasets`

```python
from contaminated_ts import CollectiveNoise
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='my_data.csv')
parser.add_argument('--root-path', type=str, default='./dataset/my_data/')
parser.add_argument('--spot-type', type=str, default='bidspot')
parser.add_argument('--spot-n-points', type=int, default=8)
parser.add_argument('--spot-depth', type=float, default=0.01)
parser.add_argument('--spot-init-points', type=float, default=0.05)
parser.add_argument('--spot-init-level', type=float, default=0.98)
parser.add_argument('--zero-clip', type=bool, default=False)
args = parser.parse_args()

cn = CollectiveNoise(seed=2025)
cn.make_noise_datasets(args)
```

This will read `./dataset/my_data/my_data.csv`, standardize the data, inject noise at all 5 levels, inverse-transform back to the original scale, and save the results as CSV files in the same directory.

### Method 2: Per-column injection

For finer control, inject noise into individual columns:

```python
import pandas as pd
import numpy as np
from contaminated_ts import CollectiveNoise

df = pd.read_csv('my_data.csv')
cn = CollectiveNoise(seed=2025)

for col in df.columns[1:]:  # skip date column
    X = df[col].values.astype(float)
    shift_noise = cn.inject_level_shift(X, noise_level=3)
    spike_noise = cn.inject_exp_spike(X, noise_level=3)
    df[f'{col}_shift'] = X + shift_noise
    df[f'{col}_spike'] = X + spike_noise
```

### SPOT parameter tuning for custom data

The SPOT algorithm parameters may need adjustment depending on your data characteristics:

- **`spot_depth`**: Controls drift detection window size (fraction of data length). Increase for data with strong trends (e.g., `0.02` for Electricity dataset vs. `0.01` for ETT datasets).
- **`spot_n_points`**: Number of extreme points for EVT fitting. Default `8` works well for most cases.
- **`spot_init_points`**: Fraction of data used for SPOT initialization. Default `0.05` (5%).
- **`spot_init_level`**: Initial quantile level. Default `0.98`.
- **`spot_type`**: Choose from `spot`, `bispot`, `dspot`, `bidspot`. Use `bidspot` (default) for data with both positive and negative trends.
- **`zero_clip`**: Set to `True` if your data should be non-negative (e.g., power consumption).

### Custom noise parameters

```python
level_shift_args = {
    1: {'freq': 0.001, 'dur': 5, 'amp': 0.001},
    2: {'freq': 0.002, 'dur': 8, 'amp': 0.001},
    3: {'freq': 0.004, 'dur': 10, 'amp': 0.0005},
    4: {'freq': 0.006, 'dur': 12, 'amp': 0.0005},
    5: {'freq': 0.008, 'dur': 15, 'amp': 0.0001},
}

cn = CollectiveNoise(
    seed=2025,
    level_shift_args=level_shift_args,
    exp_spike_args=level_shift_args,  # can use different args
)
```

## Data Validation and Regeneration

For some datasets, SPOT may produce extreme values in certain columns. Use `data_validation_and_regeneration.py` to detect and fix these:

```bash
# Full pipeline: detect problems, regenerate, merge, and validate
python data_validation_and_regeneration.py \
    --data-name electricity \
    --dataset-path dataset/electricity \
    --output-name electricity2 \
    --threshold-multiplier 3.0 \
    --regenerate \
    --merge \
    --validate
```

## SPOT Algorithm Variants

| Variant    | Description                                      |
|------------|--------------------------------------------------|
| `spot`     | Univariate streaming peaks-over-threshold         |
| `bispot`   | Bidirectional SPOT (upper and lower thresholds)    |
| `dspot`    | SPOT with drift detection                         |
| `bidspot`  | Bidirectional SPOT with drift detection (default)  |

## Noise Injection Process

1. **Standardize** input data using StandardScaler
2. **Sample anomaly locations** from a Poisson process
3. **Sample anomaly durations** from a geometric distribution
4. **Compute amplitude thresholds** using the SPOT algorithm (EVT-based)
5. **Apply noise** (level shift or exponential spike) to selected time periods
6. **Inverse-transform** back to the original data scale
