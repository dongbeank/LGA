import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from exp.exp_main import Exp_Main
import random
from utils.tools import StandardScaler
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    parser = argparse.ArgumentParser(description='LGA')
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse',type=bool, default=False,help='use inverse transform')
    parser.add_argument('--ratios', type=str, default='0.7,0.1,0.2', help='train,validation,test ratios (comma-separated, must sum to 1)')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fc_dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--query_independence', action='store_true', default=False, help='sharing query across dimension')
    parser.add_argument('--store_attn', action='store_true', default=False, help='store attention score')
    parser.add_argument('--lga', action='store_true', default=False, help='use local geometry attention')
    parser.add_argument('--schedulefree', action='store_true', default=False, help='use schedulefree')
    parser.add_argument('--eps', type=float, default=1e-2, help='sigma of lga')
    parser.add_argument('--d_G', type=int, default=64, help='embedding dimension of f_g')
    parser.add_argument('--test_noise', action='store_true', default=False, help='test on noise')
    
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    parser.add_argument('--QAM_end', type=float, default=0.3)
    
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0)
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='decoder input size')
    parser.add_argument('--factor', type=int, default=3, help='decoder input size')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    #args.use_multi_gpu = True
    #args.devices = '2,3'
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    Exp = Exp_Main
    print('Args in experiment:')
    print(args)
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            
            exp = Exp(args)  # set experiments
            setting = '{}_{}_sl{}_pl{}_dm{}_nh{}_dl{}_df{}_rie{}_{}'.format(
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.d_layers,
                args.d_ff,
                args.lga, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        all_mse_results = []
        all_mae_results = []
        clean_mses=[]
        clean_maes=[]
        noise_types = ['shift', 'spike', 'combined']
        noise_levels = [1, 2, 3, 4, 5]
        for ii in range(args.itr):
            Exp = Exp_Main
            exp = Exp(args)  # set experiments
            setting = '{}_{}_sl{}_pl{}_dm{}_nh{}_dl{}_df{}_rie{}_{}'.format(
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.d_layers,
                args.d_ff,
                args.lga, ii)
            #setting = 'test_PatchTST_random_modes64_custom_ftM_sl512_ll48_pl96_dm128_nh16_el3_dl0_df256_fc1_ebtimeF_dtTrue_test_0'
            clean_mse, clean_mae = exp.test(setting,test=1)
            mse_result, mae_result = exp.test_noise(setting, noise_types, noise_levels,'dataset/noise_result.txt')
            clean_mses.append(clean_mse)
            clean_maes.append(clean_mae)
            all_mse_results.append(mse_result)
            all_mae_results.append(mae_result)
            torch.cuda.empty_cache()
        clean_mses = np.array(clean_mses)
        clean_maes = np.array(clean_maes)
        all_mse_results = np.array(all_mse_results)  # shape: (itr, noise_type, noise_level)
        all_mae_results = np.array(all_mae_results)

        clean_mse_mean = np.mean(clean_mses)
        clean_mse_std = np.std(clean_mses)
        clean_mae_mean = np.mean(clean_maes)
        clean_mae_std = np.std(clean_maes)

        mse_mean = np.mean(all_mse_results, axis=0)
        mse_std = np.std(all_mse_results, axis=0)
        mae_mean = np.mean(all_mae_results, axis=0)
        mae_std = np.std(all_mae_results, axis=0)
        print("=" * 60)
        print("NOISE TEST RESULTS (Mean ± Std)")
        print("=" * 60)

        print(f"\nClean Data (Noise Level 0):")
        print("-" * 40)
        print(f"  MSE: {clean_mse_mean:.6f} ± {clean_mse_std:.6f}")
        print(f"  MAE: {clean_mae_mean:.6f} ± {clean_mae_std:.6f}")
        
        for i, nt in enumerate(noise_types):
            print(f"\nNoise Type: {nt}")
            print("-" * 40)
            for j, nl in enumerate(noise_levels):
                print(f"Level {nl}:")
                print(f"  MSE: {mse_mean[i,j]:.6f} ± {mse_std[i,j]:.6f}")
                print(f"  MAE: {mae_mean[i,j]:.6f} ± {mae_std[i,j]:.6f}")

        csv_data = []
        csv_data.append({
            'Noise_Type': 'clean',
            'Noise_Level': 0,
            'MSE_Mean': clean_mse_mean,
            'MSE_Std': clean_mse_std,
            'MAE_Mean': clean_mae_mean,
            'MAE_Std': clean_mae_std,
            'MSE_Format': f"{clean_mse_mean:.6f} ± {clean_mse_std:.6f}",
            'MAE_Format': f"{clean_mae_mean:.6f} ± {clean_mae_std:.6f}"
        })
        for i, nt in enumerate(noise_types):
            for j, nl in enumerate(noise_levels):
                csv_data.append({
                    'Noise_Type': nt,
                    'Noise_Level': nl,
                    'MSE_Mean': mse_mean[i,j],
                    'MSE_Std': mse_std[i,j],
                    'MAE_Mean': mae_mean[i,j],
                    'MAE_Std': mae_std[i,j],
                    'MSE_Format': f"{mse_mean[i,j]:.6f} ± {mse_std[i,j]:.6f}",
                    'MAE_Format': f"{mae_mean[i,j]:.6f} ± {mae_std[i,j]:.6f}"
                })
        df = pd.DataFrame(csv_data)
        csv_filename = f'noise_results/{args.model_id}_{args.model}_lga_{args.lga}_noise_results.csv'
        df.to_csv(csv_filename, index=False)
        
        print(f"\nResults saved to: {csv_filename}")
        print("CSV columns: Noise_Type, Noise_Level, MSE_Mean, MSE_Std, MAE_Mean, MAE_Std, MSE_Format, MAE_Format")