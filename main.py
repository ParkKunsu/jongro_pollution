import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ANN import ANN
from models.patchTST import PatchTST
from dataset import TimeSeriesDataset_ANN, TimeSeriesDataset_TST
from tqdm.auto import trange

if __name__ == "__main__":
    print(sm.datasets.sunspots.NOTE)
    data = sm.datasets.sunspots.load_pandas().data

    data.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
    data.index.freq = data.index.inferred_freq
    del data["YEAR"]

    # set dataset, dataloader
    scaler = MinMaxScaler()
    lookback_size = 9
    forecast_size = 4
    ## for patchetst ##
    patch_length = 16
    n_patches = 8
    prediction_length = 4
    ###################
    window_size = int(patch_length * n_patches / 2)

    tst_size = 20
    trn, tst = data[:-tst_size], data[-tst_size:]

    trn_scaled = scaler.fit_transform(data[:-tst_size].to_numpy(dtype=np.float32)).flatten()
    tst_scaled = scaler.transform(data[-tst_size-lookback_size:].to_numpy(dtype=np.float32)).flatten()

    trn_ds = TimeSeriesDataset_TST(trn_scaled, patch_length, n_patches)
    tst_ds = TimeSeriesDataset_TST(tst_scaled, patch_length, n_patches)

    # trn_ds = TimeSeriesDataset_TST(trn_scaled, lookback_size, forecast_size)
    # tst_ds = TimeSeriesDataset_TST(tst_scaled, lookback_size, forecast_size)

    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=32, shuffle=True)
    tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=tst_size, shuffle=False)

    # model set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = ANN(9, 4, 512)
    net = PatchTST(n_patches, patch_length, 512, 8, 4, prediction_length)
    net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=0.0001)

    pbar = trange(2000)
    # train
    for i in pbar:
        net.train()
        trn_loss = .0
        for x, y in trn_dl:
            x, y = x.to(device), y.to(device)
            p = net(x)
            optim.zero_grad()
            loss = F.mse_loss(p, y)
            loss.backward()
            optim.step()
            trn_loss += loss.item()*len(y)
        trn_loss = trn_loss/len(trn_ds)

        net.eval()
        with torch.inference_mode():
            x, y = next(iter(tst_dl))
            x, y = x.to(device), y.to(device)
            p = net(x)
            tst_loss = F.mse_loss(p,y)
            # tst_mape = mape(p,y)
            # tst_mae = mae(p,y)
        pbar.set_postfix({'loss':trn_loss, 'tst_loss':tst_loss.item()})#, 'tst_mape':tst_mape.item(), 'tst_mae':tst_mae.item()})
    
    # eval
    with torch.inference_mode():
        x, y = next(iter(tst_dl))
        x, y = x.to(device), y.to(device)
        p = net(x)

    def mape(y_pred, y_true):
        return (np.abs(y_pred - y_true)/y_true).mean() * 100

    def mae(y_pred, y_true):
        return np.abs(y_pred - y_true).mean()

    y = scaler.inverse_transform(y.cpu())
    p = scaler.inverse_transform(p.cpu())

    y = np.concatenate([y[:,0], y[-1,1:]])
    p = np.concatenate([p[:,0], p[-1,1:]])

    print(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}")

