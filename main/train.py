import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from models.ANN import ANN
from models.Transformer import PatchTST
from tqdm.auto import trange
from sklearn.metrics import r2_score

# For PatchTST
class PatchTSDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_length:int=16, n_patches:int=6, forecast_size:int=4):
    self.P = patch_length
    self.N = n_patches
    self.L = int(patch_length * n_patches / 2)  # look-back window length
    self.T = forecast_size
    self.data = ts.squeeze()

  def __len__(self):
    return len(self.data) - self.L - self.T + 1

  def __getitem__(self, i):
    look_back = self.data[i:(i+self.L)]
    look_back = np.concatenate([look_back, look_back[-1]*np.ones(int(self.P / 2), dtype=np.float32)])
    x = np.array([look_back[i*int(self.P/2):(i+2)*int(self.P/2)] for i in range(self.N)])
    y = self.data[(i+self.L):(i+self.L+self.T)]
    return x, y


# FOR ANN
class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    TODO(영준)
    멀티 채널이 입력으로 들어갈 떄, y가 목표 컬럼만 나올 수 있도록
    '''
    def __init__(self, ts:np.array, lookback_size:int, forecast_size:int, target_column:int = None):
        self.lookback_size = lookback_size
        self.forecast_size = forecast_size
        self.data = ts
        self.target_column = target_column

    def __len__(self):
        return len(self.data) - self.lookback_size - self.forecast_size + 1

    def __getitem__(self, i):
        idx = (i+self.lookback_size)
        look_back = self.data[i:idx]
        forecast = self.data[idx:idx+self.forecast_size]
        
        '''
        Data shape
        single-channel: (len_dataset, 1)
        multi-channel: (len_dataset, c_in)
        '''
        if self.data.shape[1] != 1: # 컬럼 수가 1개가 아니라면
            if not self.target_column: # 타겟 컬럼 설정이 안되어있다면
                raise NotImplementedError("multi-columns 입력은 타겟 컬럼이 설정되어야 합니다.")
            forecast = forecast[:,self.target_column] # (32,22) -> (32)
        return look_back, forecast.squeeze() # squeeze : (32, 1) -> (32)

def mse_func(y_pred, y_true):
    return float(np.square(y_true-y_pred).mean())

def rmse_func(y_pred, y_true):
    return float(np.sqrt(np.square(y_true-y_pred).mean()))

def mape_func(y_pred, y_true):
    return float((np.abs((y_pred - y_true)/y_true)).mean()) * 100

def mae_func(y_pred, y_true):
    return float(np.abs(y_pred - y_true).mean())

def main(cfg):
    ################ 1. Dataset Load  ################
    dataset_setting = cfg.get("dataset_setting")
    use_single_channel = cfg.get("use_single_channel")
    main_csv = dataset_setting.get("main_csv")
    time_axis = dataset_setting.get("time_axis")
    target = dataset_setting.get("target")
    
    data = pd.read_csv(main_csv)
    if use_single_channel:
        data = pd.DataFrame(data.loc[:, [target, time_axis]])
    # data_only_pm25 = pd.DataFrame(data.loc[:, ["PM-2.5", "일시"]])
    data[time_axis] = pd.to_datetime(data[time_axis])
    data.index = data[time_axis]
    del data[time_axis]
    
    data = data.dropna()
    target_column = data.columns.get_loc(target)
    
    # hyperparameter
    if use_single_channel:
        c_in = 1
    else:
        c_in = data.shape[1]
        
    ##################################################
    
    ############### 2. Preprocessing  ################
    # hyperparameter
    window_params = cfg.get("window_params")
    tst_size = cfg.get("tst_size")

    model = cfg.get("model")
    
    if model == ANN:
        #for ANN
        lookback_size = window_params.get("lookback_size")
        window_params["target_column"] = None if use_single_channel else target_column 
    elif model == PatchTST:
        # for Transformer
        patch_length = window_params.get("patch_length")
        n_patches = window_params.get("n_patches")
        lookback_size = int(patch_length * n_patches / 2) # same as "window_size" of patchtst
        
    forecast_size = window_params.get("forecast_size") # same as "prediction_length" for patchtst
    train_params = cfg.get("train_params")
    epochs = train_params.get("epochs")
    data_loader_params = train_params.get("data_loader_params")

    # 결측치 처리는 완전히 되었다고 가정
    
    # scaling

    if model == ANN:
        dt_class = TimeSeriesDataset
    elif model == PatchTST:
        dt_class = PatchTSDataset
    
    scaler = MinMaxScaler()
    
    if model == ANN or use_single_channel:
        trn_scaled = scaler.fit_transform(data[:-tst_size].to_numpy(dtype=np.float32))
        tst_scaled = scaler.transform(data[-tst_size-lookback_size:].to_numpy(dtype=np.float32))
        trn_ds = dt_class(trn_scaled, **window_params)
    else:
        # trn setting
        data_scaled = scaler.fit_transform(data.to_numpy(dtype=np.float32))
        trn_list = [data_scaled[:-tst_size, i].flatten() for i in range(c_in)]
        trn_ds_list = [dt_class(trn_list[i], **window_params) for i in range(c_in)]
        trn_ds = torch.utils.data.ConcatDataset(trn_ds_list)

        # tst setting
        scaler_target = MinMaxScaler()
        target_scaled = scaler_target.fit_transform(data.iloc[:, target_column:target_column+1].to_numpy(dtype=np.float32))
        tst_scaled = target_scaled[-tst_size-lookback_size:].flatten()
    
    trn_dl = torch.utils.data.DataLoader(trn_ds, **data_loader_params)
    
    tst_ds = dt_class(tst_scaled, **window_params)
    tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=tst_size, shuffle=False)
    ##################################################
    
    ########## 3. Train Hyperparams setting ##########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # model stting
    model_params = cfg.get("model_params")
    # for ANN
    if model == ANN:
        model_params["d_in"] = lookback_size
        model_params["d_out"] = forecast_size
        model_params["c_in"] = c_in
    
    # for patchTST
    elif model == PatchTST:
        model_params["n_token"] = n_patches
        model_params["input_dim"] = patch_length
        model_params["output_dim"] = forecast_size
    
    model = model(**model_params)
    model.to(device)
    
    # optimzer / loss setting
    Optim = train_params.get('optim')
    optim_params = train_params.get('optim_params')
    optim = Optim(model.parameters(), **optim_params)
    
    loss_func = train_params.get("loss")
    
    pbar = trange(epochs)
    ##################################################
    
    ################### 4. Train #####################
    trn_loss_list = []
    for i in pbar:
        model.train()
        trn_loss = .0
        for x, y in trn_dl:
            x, y = x.to(device), y.to(device)   # (32, 18), (32, 4)
            p = model(x)
            optim.zero_grad()
            loss = loss_func(p, y)
            loss.backward()
            optim.step()
            trn_loss += loss.item()*len(y)
        trn_loss = trn_loss/len(trn_ds)

        model.eval()
        with torch.inference_mode():
            x, y = next(iter(tst_dl))
            x, y = x.to(device), y.to(device)
            p = model(x)
            tst_loss = loss_func(p,y)
        trn_loss_list.append(trn_loss)
        pbar.set_postfix({'loss':trn_loss, 'tst_loss':tst_loss.item()})
    plt.title(f"Train loss graph")
    plt.plot(range(len(trn_loss_list)), trn_loss_list)
    plt.savefig("figs/single_patchtst/train_loss13.jpg", format="jpeg")
    plt.cla()
    
    pth_save_path = "a.pth"
    
    torch.save(model.state_dict(), pth_save_path)
    print("done")
    ##################################################
    
    ################# 5. Evaluation ##################
    eval_params = cfg.get("eval_params")
    model.load_state_dict(torch.load(pth_save_path))
    model.eval()
    
    with torch.inference_mode():
        # tst_size=200
        # x : -565~-200 까지의 데이터 (365)
        # y : -200~-1 : 마지막 200 개 데이터
        if not eval_params.get("dynamic"):
            x, y = next(iter(tst_dl))
            x, y = x.to(device), y.to(device)
            print(x.shape, y.shape)
            p = model(x)
            if use_single_channel:
                y = y.cpu()/scaler.scale_[0] + scaler.min_[0] #shape: (200 - 7 + 1, 7)
                p = p.cpu()/scaler.scale_[0] + scaler.min_[0] #shape: (200 - 7 + 1, 7)
            else:
                y = y.cpu()/scaler.scale_[target_column] + scaler.min_[target_column] #shape: (200 - 7 + 1, 7)
                p = p.cpu()/scaler.scale_[target_column] + scaler.min_[target_column] #shape: (200 - 7 + 1, 7)

            
            total_y = np.concatenate([y[:,0], y[-1,1:]])
            total_p = np.concatenate([p[:,0], p[-1,1:]])
             
        else:
            prediction_size = eval_params.get("prediction_size")
            x = torch.from_numpy(tst_scaled[:-tst_size].reshape(1, lookback_size, -1)) # (1, 365, 1)
            y = tst_scaled[-tst_size:]

            pred_list = []   
            for _ in range(int(tst_size / prediction_size)):      
                x = x.to(device) # shape: (1, 365, 22)
                p = model(x) # shape: (1, 7)
            
                pred_list.append(p[:, :prediction_size])
                x = torch.cat([x[: ,prediction_size:, :], p[:, :prediction_size].unsqueeze(-1)], axis=1)
            
            pred = torch.cat(pred_list)
            
            y = y.squeeze()/scaler.scale_[0] + scaler.min_[0]
            p = pred.cpu()/scaler.scale_[0] + scaler.min_[0]

    ##################################################
    
    ################ 6. Plot and save ################
    # 36: 초미세먼지 수치가 "나쁨" 인 기중
    peak_idx = np.unique(np.where(y > 36)[0])
    print(y.shape)
    print(peak_idx)
    peak_y, peak_p = y[peak_idx], p[peak_idx]
    save_files_path = cfg.get("save_files")
    csv_path, day_path, peak_path = save_files_path.get("csv"), save_files_path.get("day"), save_files_path.get("peak")

    # 평균 오차 저장
    mse, rmse, r2, mae, mape = mse_func(p,y), rmse_func(p,y), r2_score(p,y), mae_func(p,y), mape_func(p,y)
    peak_mse, peak_rmse, peak_r2, peak_mae, peak_mape = mse_func(peak_p,peak_y), rmse_func(peak_p,peak_y), r2_score(peak_p,peak_y), mae_func(peak_p,peak_y), mape_func(peak_p,peak_y)
    result = {"Result_everyday": {"MSE": mse, "RMSE": rmse, "R2": r2, "MAE": mae, "MAPE": mape},
              "Result_peak": {"MSE": peak_mse, "RMSE": peak_rmse, "R2": peak_r2, "MAE": peak_mae, "MAPE": peak_mape}}
    
    # 건수 says, 매일매일 데이터는 기존 방식처럼 하나씩 그려도 될것 같음
    
    pd.DataFrame(result).to_csv(csv_path)
    
    plt.title(f"PatchTST, MSE:{mse:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}, \nMAE:{mae:.4f}, MAPE:{mape:.4f}")
    plt.plot(range(tst_size), total_y, label="True")
    plt.plot(range(tst_size), total_p, label="Prediction")
    plt.legend()

    plt.savefig(day_path, format="jpeg")
    plt.cla()
        
    for i in peak_idx:
        mse, rmse, r2, mae, mape = mse_func(p[i],y[i]), rmse_func(p[i],y[i]), r2_score(p[i],y[i]), mae_func(p[i],y[i]), mape_func(p[i],y[i])
        plt.title(f"PatchTST, MSE:{mse:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}, \nMAE:{mae:.4f}, MAPE:{mape:.4f}")
        plt.plot(range(forecast_size), y[i], label="True")
        plt.plot(range(forecast_size), p[i], label="Prediction")
        plt.legend()

        plt.savefig(peak_path + f"{i}.jpg", format="jpeg")
        plt.cla()
    ##################################################
    

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Time-Series Prediction with ANN, PatchTST", add_help=add_help)
  parser.add_argument("-c", "--config", default="pretrained_config/single_patchtst/short_term/config_patchtst12_epochlr3_dim.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  
  main(config)