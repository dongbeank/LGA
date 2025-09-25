from contaminated_ts import CollectiveNoise
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Making Time Series Robust Benchmarks')
    parser.add_argument('--data-path', type=str, required=True, help='dataset filename (e.g. ETTh1.csv)')
    parser.add_argument('--root-path', type=str, required=True, help='root path containing the original CSV')
    parser.add_argument('--output-path', type=str, default=None, help='output path for noise files (default: same as root-path)')
    parser.add_argument('--spot-type', type=str, default='bidspot')
    parser.add_argument('--spot-n-points', type=int, default=8)
    parser.add_argument('--spot-depth', type=float, default=0.01)
    parser.add_argument('--spot-init-points', type=float, default=0.05)
    parser.add_argument('--spot-init-level', type=float, default=0.98)
    parser.add_argument('--zero-clip', type=bool, default=False)
    args = parser.parse_args()

    cn = CollectiveNoise(seed=2025)
    print(f"Noise Injection into {args.data_path}")
    cn.make_noise_datasets(args)
