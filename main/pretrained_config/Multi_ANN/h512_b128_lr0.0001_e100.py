import torch
import torch.nn.functional as F
import torchmetrics
from models.ANN import ANN
from models.LSTM import StatefulLSTM, StatelessLSTM
from models.Transformer import PatchTST

config = {
  "use_single_channel": False,
  'dataset_setting':{
    "main_csv": "/home/dataset/complete_dataset.csv",
    "time_axis": "일시",
    "target": "PM-2.5"
  },
  "window_params":{
    "lookback_size": 365,
    "forecast_size": 7
  },
  "tst_size": 200,
  
  'model': ANN, # or RandomForestRegressor
  'model_params': {
    'd_hidden': 512,
    'activation': F.relu,
  },
  
  
  'train_params': {
    'data_loader_params': {
      'batch_size': 128
      ,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cuda',
    'epochs': 100,
  },
  'eval_params':{
      "dynamic": False,
      "prediction_size": 1
  },
  "save_files":{
      "csv": "csv/h512_b128_lr0.0001_e100.csv",
      "day": "./figs/everyday/h512_b128_lr0.0001_e1000.jpg",
      "peak": "./figs/peakday/"
  }

}