import numpy as np
from ads_evt.spot import SPOT, biSPOT, dSPOT, bidSPOT
import os
import pandas as pd
import random
import time
from sklearn.preprocessing import StandardScaler

class CollectiveNoise:
    def __init__(self, seed=2025, level_shift_args=None, exp_spike_args=None, spot_args=None):
        """
        Initialize the CollectiveNoise class with default parameters for level shift and exponential spike.
        Parameters:
        - seed: Random seed for reproducibility
        - level_shift_args: Dictionary containing frequency, duration, and amplitude for level shift
        - exp_spike_args: Dictionary containing frequency, duration, and amplitude for exponential spike
        - spot_args: Dictionary containing arguments for the SPOT algorithm
        """
        if level_shift_args is None:
            self.level_shift_args_map = {
                1: {'freq': 0.002, 'dur': 6, 'amp': 0.0016},
                2: {'freq': 0.004, 'dur': 9, 'amp': 0.0016},
                3: {'freq': 0.004, 'dur': 12, 'amp': 0.0004},
                4: {'freq': 0.008, 'dur': 12, 'amp': 0.0004},
                5: {'freq': 0.008, 'dur': 15, 'amp': 0.0001},
            }
        else:
            self.level_shift_args_map = level_shift_args

        if exp_spike_args is None:
            self.exp_spike_args_map = {
                1: {'freq': 0.002, 'dur': 6, 'amp': 0.0016},
                2: {'freq': 0.004, 'dur': 9, 'amp': 0.0016},
                3: {'freq': 0.004, 'dur': 12, 'amp': 0.0004},
                4: {'freq': 0.008, 'dur': 12, 'amp': 0.0004},
                5: {'freq': 0.008, 'dur': 15, 'amp': 0.0001},
            }
        else:
            self.exp_spike_args_map = exp_spike_args

        if spot_args is None:
            self.spot_args = {
                'type': 'bidspot',
                'n_points': 8,
                'depth': 0.01,
                'init_points': 0.05,
                'init_level': 0.98
            }
        else:
            self.spot_args = spot_args

        self.evt_values = {}
        self.evt_for_col = []

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def custom_inject_level_shift(self, X: np.array, freq, dur, amp):
        """
        Injects a level shift into the input signal X based on the specified frequency,
        duration, and amplitude parameters.
s
        Parameters:
        - X: Input signal (numpy array)
        - freq: Frequency parameter of poisson distribution for the level shift, which means the risk of the anomaly
        - dur: Duration parameter of exponential distribution for the level shift
        - amp: Amplitude arguments of extreme distirbution for the level shift

        Returns:
        - noise: noise injected
        - X_shifted: Signal with injected level shift
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")
            
        noise = np.zeros_like(X)

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T)
        init_data = min(5000, int(self.spot_args['init_points'] * X.shape[0]))

        m = np.random.randint(0, T + 1, N)
        m = m[m >= X.shape[0]]
        m = m - X.shape[0]
        n = m.shape[0]

        dur = 1 / (dur - 1)
        d = np.random.geometric(dur, n) + 1

        try:
            if self.spot_args['type'] == 'spot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = SPOT(q=amp, n_points=self.spot_argsspot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bispot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = biSPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            elif self.spot_args['type'] == 'dspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = dSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bidspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = bidSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            else:
                raise ValueError("Invalid type for amplitude arguments. Choose from 'spot', 'bispot', 'dspot', or 'bidspot'.")
            
        except ValueError as e:
            print(f"Error in amplitude arguments: {e}")
            raise

        for i in range(n):
            if self.spot_args['type'] == 'spot' or self.spot_args['type'] == 'dspot':
                noise[m[i]:m[i] + d[i]] += np.min(upper[m[i]:m[i] + d[i]])
            else:
                if np.random.rand() < 0.5:
                    noise[m[i]:min(m[i] + d[i], X.shape[0])] = np.maximum(np.min(upper[m[i]:min(m[i] + d[i], X.shape[0])]), noise[m[i]:min(m[i] + d[i], X.shape[0])])
                else:
                    noise[m[i]:min(m[i] + d[i], X.shape[0])] = np.minimum(np.max(lower[m[i]:min(m[i] + d[i], X.shape[0])]), noise[m[i]:min(m[i] + d[i], X.shape[0])])

        return noise
    
    def inject_level_shift(self, X: np.array, noise_level: int):
        """
        Injects level shift noise into the input signal using predefined parameters for the specified noise level.
        
        Parameters:
        - X: Input signal (numpy array)
        - noise_level: Noise level (1-5) determining the intensity of the level shift
        
        Returns:
        - noise: Level shift noise injected into the signal
        """
        return self.custom_inject_level_shift(X, self.level_shift_args_map[noise_level]['freq'], self.level_shift_args_map[noise_level]['dur'], self.level_shift_args_map[noise_level]['amp'])
        
    def custom_inject_exp_spike(self, X: np.array, freq, dur, amp):
        """
        Injects a exponential spike into the input signal X based on the specified frequency,
        duration, and amplitude parameters.

        Parameters:
        - X: Input signal (numpy array)
        - freq: Frequency parameter of poisson distribution for the exponential spike, which means the risk of the anomaly
        - dur: Duration parameter of exponential distribution for the exponential spike
        - amp: Amplitude arguments of extreme distirbution for the exponential spike

        Returns:
        - noise: noise injected
        - X_shifted: Signal with injected exponential spike
        """
        if X.ndim != 1:
            raise ValueError("Input signal X must be a 1D numpy array.")

        noise = np.zeros_like(X)

        T = 2 * X.shape[0] - 1
        N = np.random.poisson(freq * T) # Poisson distribution for the number of level shifts with steady state
        init_data = min(int(self.spot_args['init_points'] * X.shape[0]), 5000)

        m = np.random.randint(0, T + 1, N) # Randomly select the start points for level shifts
        m = m[m >= X.shape[0]] # filter out the start points that are big than the length of X
        m = m - X.shape[0]
        n = m.shape[0]

        dur = 2 / dur
        d1 = np.random.geometric(dur, n) # Geometric distribution for the front duration of exponential distribution
        d2 = np.random.geometric(dur, n) # Geometric distribution for the back duration of exponential distribution

        try:
            if self.spot_args['type'] == 'spot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = SPOT(q=amp, n_points=amp)
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bispot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = biSPOT(q=amp, n_points=self.spot_args['n_points'])
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            elif self.spot_args['type'] == 'dspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = dSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.abs(np.concatenate((np.full(init_data, res['thresholds'][0]), res['thresholds'])) - X)

            elif self.spot_args['type'] == 'bidspot':
                if self.evt_values.get(amp) is not None:
                    res = self.evt_values[amp]
                else:
                    spot = bidSPOT(q=amp, n_points=self.spot_args['n_points'], depth=int(self.spot_args['depth'] * X.shape[0]))
                    spot.fit(init_data=init_data, data=X)
                    spot.initialize(self.spot_args['init_level'])
                    res = spot.run()
                    self.evt_values[amp] = res
                upper = np.concatenate((np.full(init_data, res['upper_thresholds'][0]), res['upper_thresholds'])) - X
                upper[upper < 0] = 0
                lower = np.concatenate((np.full(init_data, res['lower_thresholds'][0]), res['lower_thresholds'])) - X
                lower[lower > 0] = 0

            else:
                raise ValueError("Invalid type for amplitude arguments. Choose from 'spot', 'bispot', 'dspot', or 'bidspot'.")
            
        except ValueError as e:
            print(f"Error in amplitude arguments: {e}")
            raise

        for i in range(n):
            if self.spot_args['type'] == 'spot' or self.spot_args['type'] == 'dspot':
                noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] += self.__exp_spike_curve(upper[min(m + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])]

            else:
                if np.random.rand() < 0.5:
                    noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] = np.maximum(self.__exp_spike_curve(upper[min(m[i] + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])], noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])])
                else:
                    noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])] = np.minimum(self.__exp_spike_curve(lower[min(m[i] + d1[i], X.shape[0] - 1)], 1e-4, d1[i], d2[i])[:min(d1[i] + d2[i] + 1, X.shape[0] - m[i])], noise[m[i]:min(m[i] + d1[i] + d2[i] + 1, X.shape[0])])
        
        return noise
    
    def inject_exp_spike(self, X: np.array, noise_level: int):
        """
        Injects exponential spike noise into the input signal using predefined parameters for the specified noise level.
        
        Parameters:
        - X: Input signal (numpy array)
        - noise_level: Noise level (1-5) determining the intensity of the exponential spike
        
        Returns:
        - noise: Exponential spike noise injected into the signal
        """
        return self.custom_inject_exp_spike(X, self.exp_spike_args_map[noise_level]['freq'], self.exp_spike_args_map[noise_level]['dur'], self.exp_spike_args_map[noise_level]['amp'])
        
    def __exp_spike_curve(self, h, beta, epsilon1, epsilon2):
        """
        Generates an exponential spike curve function.

        Parameters:
        - h: Height of the curve
        - beta: Parameter controlling the steepness of the curve
        - epsilon1: Parameter controlling the width of the ascent part of the curve
        - epsilon2: Parameter controlling the width of the descent part of the curve

        Returns:
        - y: Ascent curve values
        """

        x = np.arange(0, epsilon1 + epsilon2 + 1, dtype=float)
        y = np.zeros_like(x)
        
        y[:epsilon1] = h / np.exp(np.log(beta) / epsilon1 * (x[:epsilon1] - epsilon1))
        y[epsilon1:] = h * np.exp(np.log(beta) / epsilon2 * (x[epsilon1:] - epsilon1))
        return y
    
    def inject_noise(self, X: np.array, noise_level: int):
        """
        Injects both level shift and exponential spike noise into the input signal.
        
        Parameters:
        - X: Input signal (numpy array)
        - noise_level: Noise level (1-5) determining the intensity of both noise types
        
        Returns:
        - noise_shift: Level shift noise
        - noise_spike: Exponential spike noise
        """
        noise_shift = self.inject_level_shift(X, noise_level)
        noise_spike = self.inject_exp_spike(X, noise_level)

        return noise_shift, noise_spike

    def make_noise_datasets(self, args):
        """
        Creates noisy datasets by injecting level shift and exponential spike noise into time series data.
        
        This method reads the original dataset, applies standardization, injects different types of noise
        at various levels, and saves the resulting noisy datasets to CSV files.
        
        Parameters:
        - args: Argument object containing:
            - root_path: Directory path containing the dataset
            - data_path: Filename of the original dataset
            - spot_type: Type of SPOT algorithm ('spot', 'bispot', 'dspot', 'bidspot')
            - spot_n_points: Number of points for SPOT algorithm
            - spot_depth: Depth parameter for SPOT algorithm
            - spot_init_points: Initial points ratio for SPOT algorithm
            - spot_init_level: Initial level for SPOT algorithm
            - zero_clip: Whether to clip negative values to zero
        
        Returns:
        - None (saves files to disk)
        """
        df = pd.read_csv(os.path.join(args.root_path, args.data_path))
        output_path = getattr(args, 'output_path', None) or args.root_path
        os.makedirs(output_path, exist_ok=True)
        data_cols = df.columns[1:]
        X = df[data_cols].values.copy()
        self.evt_values = {}
        self.evt_for_col = [{} for _ in range(X.shape[1])]
        self.spot_args['type'] = args.spot_type
        self.spot_args['n_points'] = args.spot_n_points
        self.spot_args['depth'] = args.spot_depth
        self.spot_args['init_points'] = args.spot_init_points
        self.spot_args['init_level'] = args.spot_init_level

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if self.exp_spike_args_map.keys() != self.level_shift_args_map.keys():
            raise ValueError("The level shift and exponential spike arguments map keys must be the same.")

        for noise_level in self.exp_spike_args_map.keys():
            np.random.seed(self.seed)
            random.seed(self.seed)
            start_time = time.time()
            print(f'noise level: {noise_level}')

            file_name, file_ext = os.path.splitext(args.data_path)
            new_file_shift = f'{file_name}_level_{noise_level}_type_shift{file_ext}'
            new_file_spike = f'{file_name}_level_{noise_level}_type_spike{file_ext}'
            new_file_combined = f'{file_name}_level_{noise_level}_type_combined{file_ext}'

            new_X_shift = np.zeros_like(X)
            new_X_spike = np.zeros_like(X)
            new_X_combined = np.zeros_like(X)

            for i in range(X.shape[1]):
                print(f'column: {data_cols[i]}')
                self.evt_values = self.evt_for_col[i]
                X_col = X[:, i]
                shift, spike = self.inject_noise(X_col, noise_level)
                new_X_shift[:, i] = shift + X_col
                new_X_spike[:, i] = spike + X_col
                new_X_combined[:, i] = np.where(np.abs(spike) > np.abs(shift), spike + X_col, shift + X_col)

                if args.zero_clip:
                    new_X_shift[:, i] = np.clip(new_X_shift[:, i], 0, None)
                    new_X_spike[:, i] = np.clip(new_X_spike[:, i], 0, None)
                    new_X_combined[:, i] = np.clip(new_X_combined[:, i], 0, None)

                self.evt_for_col[i] = self.evt_values

            new_df_shift = df.copy()
            new_df_spike = df.copy()
            new_df_combined = df.copy()

            new_X_shift = scaler.inverse_transform(new_X_shift)
            new_X_spike = scaler.inverse_transform(new_X_spike)
            new_X_combined = scaler.inverse_transform(new_X_combined)

            new_df_shift[data_cols] = new_X_shift
            new_df_spike[data_cols] = new_X_spike
            new_df_combined[data_cols] = new_X_combined

            new_df_shift.to_csv(os.path.join(output_path, new_file_shift), index=False)
            new_df_spike.to_csv(os.path.join(output_path, new_file_spike), index=False)
            new_df_combined.to_csv(os.path.join(output_path, new_file_combined), index=False)
            end_time = time.time()
            print(f'noise level: {noise_level} done! time: {end_time - start_time:.2f}s')
        print('all done!')
