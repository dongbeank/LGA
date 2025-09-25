from data.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from schedulefree import RAdamScheduleFree, AdamWScheduleFree, SGDScheduleFree, AdamWScheduleFreeReference

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from models import PatchTST


model_dict = {'PatchTST': PatchTST}

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, noise_file = None):
        data_set, data_loader = data_provider(self.args, flag, noise_file)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.schedulefree:
            model_optim = AdamWScheduleFree(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        t_start = time.time()
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = None
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()#.cpu()
                true = batch_y.detach()#.cpu()

                loss = criterion(pred, true)#.detach().cpu()

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        t_end = time.time()
        print(f"Validation took {t_end - t_start:.4f} seconds")
        return total_loss

    def train(self, setting, gradiv = False, loss_adap=False, mmd_loss = False):
        train_data, train_loader = self._get_data(flag='train')
        tds = []
        tls = []
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        trainlosss=[]
        vallosss=[]
        testlosss=[]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                           steps_per_epoch = train_steps,
                                           pct_start = self.args.pct_start,
                                           epochs = self.args.train_epochs,
                                           max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            if self.args.schedulefree:
                model_optim.train()
            epoch_time = time.time()
            g_loss = 0.0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                #g_optim.zero_grad()
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs, gloss = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].float().to(self.device, non_blocking=True)#.to(self.device)
                    
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    gloss=gloss.mean()
                    (loss+gloss).backward()     
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.args.schedulefree:
                model_optim.eval()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            if self.args.schedulefree:
                model_optim.train()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            trainlosss.append(train_loss)
            vallosss.append(vali_loss)
            testlosss.append(test_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if not self.args.schedulefree:
                if self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
                else:
                    print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.schedulefree:
            model_optim.eval()
        self.model.load_state_dict(torch.load(best_model_path,map_location='cuda:0'))

        return self.model, trainlosss, vallosss, testlosss

    def test(self, setting, test=0, noise_file = None, save_file = 'test'):
        test_data, test_loader = self._get_data(flag='test', noise_file = noise_file)
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),map_location='cuda:0'))

        preds = []
        trues = []
        inputx = []
        if self.args.schedulefree:
            model_optim = self._select_optimizer()
            model_optim.eval()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = None
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs, _ = self.model(batch_x)
                    #outputs = self.model(batch_x)#, geodesic=True)

                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        
        if self.args.inverse:
            preds=test_data.inverse_transform(preds.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)
            trues=test_data.inverse_transform(trues.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)
            inputx=test_data.inverse_transform(inputx.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        #print(save_file + " Performance")
        #print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        print('{}\t{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write(save_file + " Performance \n")
        f.write('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        #np.save(folder_path + save_file + 'pred.npy', preds)
        #np.save(folder_path + save_file + 'true.npy', trues)
        #np.save(folder_path + save_file + 'x.npy', inputx)
        return mse, mae

    def test_noise(self, setting, noise_type, noise_level, output_file=None):
        file_name, file_ext = os.path.splitext(self.args.data_path)
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        mse_results = np.zeros((len(noise_type), len(noise_level)))
        mae_results = np.zeros((len(noise_type), len(noise_level)))        
        if output_file:
            out_f = open(output_file, 'w')
        for i, nt in enumerate(noise_type):
            for j, nl in enumerate(noise_level):
                noise_file = f"{file_name}_noise/{file_name}_level_{nl}_type_{nt}{file_ext}"
                save_file = f"level_{nl}_type_{nt}"
                mse, mae = self.test(setting, noise_file=noise_file, save_file=save_file)
                mse_results[i, j] = mse
                mae_results[i, j] = mae
                if output_file:
                    out_f.write(f"{save_file} Performance\n")
                    out_f.write(f'mse:{mse}\n\n')
            print()
        if output_file:
            out_f.close()
        return mse_results, mae_results
    
    def predict(self, setting, load=True):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        
        preds=test_data.inverse_transform(preds.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)
        trues=test_data.inverse_transform(trues.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)
        inputx=test_data.inverse_transform(inputx.reshape(-1, self.args.dec_in)).reshape(-1,self.args.pred_len, self.args.dec_in)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'real_prediction_pred.npy', preds)
        np.save(folder_path + 'real_prediction_true.npy', trues)
        np.save(folder_path + 'real_prediction_x.npy', inputx)

        return
