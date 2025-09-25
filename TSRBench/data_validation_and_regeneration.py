import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from contaminated_ts import ContaminatedTS
from collective_noise import CollectiveNoise

class DataValidationAndRegeneration:
    def __init__(self, seed=2025):
        """
        Initialize the data validation and regeneration class
        
        Parameters:
        - seed: Random seed for reproducibility
        """
        self.seed = seed
        self.problematic_columns = {}
        
    def check_problematic_columns(self, data_name, dataset_path, level=3, threshold_multiplier=3):
        """
        Check for problematic columns in the generated noise data
        
        Parameters:
        - data_name: Name of the dataset (e.g., 'ETTm2', 'electricity')
        - dataset_path: Path to the dataset directory
        - level: Noise level to check
        - threshold_multiplier: Multiplier for determining problematic values
        
        Returns:
        - tmp_list: List of problematic columns
        """
        print(f"Checking problematic columns for {data_name} at level {level}...")
        
        # Load noise data files
        noise_df1 = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_shift.csv')
        noise_df2 = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_spike.csv')
        noise_df3 = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_combined.csv')
        noise_df4 = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_combined_old.csv')
        
        # Load original data
        df = pd.read_csv(f'{dataset_path}/{data_name}.csv')
        
        print(f"Data lengths: {len(noise_df1)}, {len(noise_df2)}, {len(noise_df3)}, {len(noise_df4)}, {len(df)}")
        
        tmp_list = []
        for col in df.columns[1:]:  # Skip date column
            # Check for values that are too high
            if noise_df3[col].max() > df[col].max() * threshold_multiplier:
                tmp_list.append(col)
            # Check for values that are too low (for negative values)
            if noise_df3[col].min() < df[col].min() * threshold_multiplier and df[col].min() < 0:
                tmp_list.append(col)
        
        # Remove duplicates and sort
        tmp_list = list(set(tmp_list))
        if data_name == 'electricity':
            # For electricity data, sort numerically
            tmp_list = sorted(tmp_list, key=int)
        else:
            tmp_list = sorted(tmp_list)
        
        print(f"Found {len(tmp_list)} problematic columns: {tmp_list}")
        self.problematic_columns[data_name] = tmp_list
        
        return tmp_list
    
    def visualize_problematic_column(self, data_name, dataset_path, col, level=5, 
                                   noise_type='combined', save_plot=False, plot_path=None):
        """
        Visualize a specific problematic column to manually verify issues
        
        Parameters:
        - data_name: Name of the dataset
        - dataset_path: Path to the dataset directory
        - col: Column name to visualize
        - level: Noise level
        - noise_type: Type of noise ('shift', 'spike', 'combined', 'missing', etc.)
        - save_plot: Whether to save the plot
        - plot_path: Path to save the plot
        """
        print(f"Visualizing column '{col}' for {data_name} at level {level}, type {noise_type}")
        
        # Load noise data and original data
        if noise_type == 'missing':
            noise_df = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_missing2.csv')
        else:
            noise_df = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_{noise_type}2.csv')
        
        df = pd.read_csv(f'{dataset_path}/{data_name}.csv')
        
        # Create visualization
        plt.figure(figsize=(30, 5))
        plt.plot(noise_df[col][:], color='blue', label=f'Noise {noise_type}', alpha=0.8)
        plt.plot(df[col][:], color='green', label='Original', alpha=0.7)
        plt.title(f'{data_name} - Column {col} - Level {level} - Type {noise_type}')
        plt.legend()
        
        if save_plot and plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def extract_problematic_columns(self, data_name, dataset_path, problematic_cols, output_filename=None):
        """
        Extract problematic columns and save as a separate dataset
        
        Parameters:
        - data_name: Name of the dataset
        - dataset_path: Path to the dataset directory
        - problematic_cols: List of problematic column names
        - output_filename: Name for the output file (if None, uses default naming)
        """
        if output_filename is None:
            output_filename = f'{data_name}2.csv'
        
        # Load original data
        df = pd.read_csv(f'{dataset_path}/{data_name}.csv')
        
        # Create list including date column
        tmp_list = ['date'] + problematic_cols
        
        # Extract problematic columns
        fault_df = df[tmp_list]
        
        # Save extracted data
        output_path = f'{dataset_path}/{output_filename}'
        fault_df.to_csv(output_path, index=False)
        
        print(f"Extracted {len(problematic_cols)} problematic columns to {output_path}")
        print(f"Columns: {problematic_cols}")
        
        return fault_df
    
    def regenerate_noise_data(self, data_path, root_path, spot_args=None):
        """
        Regenerate noise data using ContaminatedTS
        
        Parameters:
        - data_path: Name of the data file (e.g., 'electricity2.csv')
        - root_path: Root path to the dataset
        - spot_args: SPOT algorithm arguments
        """
        if spot_args is None:
            spot_args = {
                'spot_type': 'bidspot',
                'spot_n_points': 8,
                'spot_depth': 0.01,
                'spot_init_points': 0.05,
                'spot_init_level': 0.98,
                'zero_clip': False
            }
        
        print(f"Regenerating noise data for {data_path}")
        
        # Create argument object
        class Args:
            def __init__(self, data_path, root_path, **kwargs):
                self.data_path = data_path
                self.root_path = root_path
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(data_path, root_path, **spot_args)
        
        # Initialize ContaminatedTS and generate noise
        ct = ContaminatedTS(seed=self.seed)
        print(f"Noise Injection into {args.data_path}")
        noise_df = ct.make_noise_datasets(args)
        
        return noise_df
    
    def merge_datasets(self, data_name, dataset_path, original_levels=[1, 2, 3, 4, 5], 
                      regenerated_levels=[1, 2, 3, 4, 5], noise_types=None):
        """
        Merge original and regenerated datasets
        
        Parameters:
        - data_name: Name of the dataset
        - dataset_path: Path to the dataset directory
        - original_levels: List of original noise levels
        - regenerated_levels: List of regenerated noise levels (mapped to original levels)
        - noise_types: List of noise types to merge
        """
        if noise_types is None:
            noise_types = ['shift', 'spike', 'combined', 'missing', 'missing_shift', 'missing_spike', 'missing_combined']
        
        print(f"Merging datasets for {data_name}...")
        
        for level_idx, level in enumerate(original_levels):
            regenerated_level = regenerated_levels[level_idx]
            print(f"Processing level {level} (regenerated level {regenerated_level})...")
            
            for noise_type in noise_types:
                try:
                    # Load original noise data
                    original_file = f'{dataset_path}/{data_name}_level_{level}_type_{noise_type}.csv'
                    if os.path.exists(original_file):
                        noise_df = pd.read_csv(original_file)
                    else:
                        print(f"Warning: {original_file} not found, skipping...")
                        continue
                    
                    # Load regenerated data
                    regenerated_file = f'{dataset_path}/{data_name}2_level_{regenerated_level}_type_{noise_type}.csv'
                    if os.path.exists(regenerated_file):
                        tmp_df = pd.read_csv(regenerated_file)
                        
                        # Merge data (replace problematic columns with regenerated ones)
                        noise_df[tmp_df.columns[1:]] = tmp_df[tmp_df.columns[1:]]
                        
                        # Save merged data
                        output_file = f'{dataset_path}/{data_name}_level_{level}_type_{noise_type}2.csv'
                        noise_df.to_csv(output_file, index=False)
                        print(f"Merged and saved: {output_file}")
                    else:
                        print(f"Warning: {regenerated_file} not found, skipping...")
                        
                except Exception as e:
                    print(f"Error processing {noise_type} at level {level}: {e}")
    
    def validate_full_dataset(self, data_name, dataset_path, levels=[1, 2, 3, 4, 5], 
                            threshold_multiplier=3):
        """
        Validate the full dataset after regeneration and merging
        
        Parameters:
        - data_name: Name of the dataset
        - dataset_path: Path to the dataset directory
        - levels: List of noise levels to validate
        - threshold_multiplier: Multiplier for determining problematic values
        
        Returns:
        - validation_results: Dictionary with validation results for each level
        """
        print(f"Validating full dataset for {data_name}...")
        
        validation_results = {}
        df = pd.read_csv(f'{dataset_path}/{data_name}.csv')
        
        for level in levels:
            print(f"Validating level {level}...")
            problematic_cols = []
            
            try:
                # Check combined noise data
                noise_df = pd.read_csv(f'{dataset_path}/{data_name}_level_{level}_type_combined2.csv')
                
                for col in df.columns[1:]:  # Skip date column
                    # Check for values that are too high
                    if noise_df[col].max() > df[col].max() * threshold_multiplier:
                        problematic_cols.append(col)
                    # Check for values that are too low (for negative values)
                    if noise_df[col].min() < df[col].min() * threshold_multiplier and df[col].min() < 0:
                        problematic_cols.append(col)
                
                problematic_cols = list(set(problematic_cols))
                validation_results[level] = {
                    'problematic_columns': problematic_cols,
                    'count': len(problematic_cols),
                    'status': 'PASS' if len(problematic_cols) == 0 else 'FAIL'
                }
                
                print(f"Level {level}: {validation_results[level]['status']} - {len(problematic_cols)} problematic columns")
                
            except Exception as e:
                print(f"Error validating level {level}: {e}")
                validation_results[level] = {
                    'problematic_columns': [],
                    'count': -1,
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        return validation_results


def main():
    """
    Main function to run the data validation and regeneration process
    """
    parser = argparse.ArgumentParser(description='Data Validation and Regeneration for Time Series Noise Injection')
    parser.add_argument('--data-name', type=str, default='electricity', help='Dataset name')
    parser.add_argument('--dataset-path', type=str, default='dataset/electricity', help='Path to dataset directory')
    parser.add_argument('--output-name', type=str, default='electricity', help='Path to output directory')
    # parser.add_argument('--check-level', type=int, default=3, help='Level to check for problematic columns')
    parser.add_argument('--threshold-multiplier', type=float, default=3.0, help='Threshold multiplier for detecting problems')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate noise data for problematic columns')
    parser.add_argument('--merge', action='store_true', help='Merge original and regenerated datasets')
    parser.add_argument('--validate', action='store_true', help='Validate the full dataset')
    parser.add_argument('--visualize', type=str, default=None, help='Column name to visualize')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DataValidationAndRegeneration(seed=args.seed)
    
    # Step 1: Check for problematic columns
    problematic_cols = []
    for level in range(1,6):
        problematic_cols.extend(validator.check_problematic_columns(
            args.data_name, args.dataset_path, level, args.threshold_multiplier
        ))
    problematic_cols = list(set(problematic_cols))
    problematic_cols = sorted(problematic_cols, key=int)

    # Step 2: Visualize specific column if requested
    if args.visualize and args.visualize in problematic_cols:
        validator.visualize_problematic_column(
            args.data_name, args.dataset_path, args.visualize
        )
    
    # Step 3: Extract problematic columns
    if problematic_cols:
        validator.extract_problematic_columns(
            args.data_name, args.dataset_path, problematic_cols, args.output_name
        )
        
        # Step 4: Regenerate noise data if requested
        if args.regenerate:
            validator.regenerate_noise_data(
                f'{args.output_name}.csv', 
                args.dataset_path
            )
        
        # Step 5: Merge datasets if requested
        if args.merge:
            validator.merge_datasets(args.data_name, args.dataset_path)
    
    # Step 6: Validate full dataset if requested
    if args.validate:
        validation_results = validator.validate_full_dataset(
            args.data_name, args.dataset_path, threshold_multiplier=args.threshold_multiplier
        )
        
        print("\n=== Final Validation Results ===")
        for level, result in validation_results.items():
            print(f"Level {level}: {result['status']} ({result['count']} problematic columns)")


if __name__ == "__main__":
    main()
