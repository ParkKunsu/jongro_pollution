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
      'batch_size': 32,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cuda',
    'epochs': 200,
  },
  'eval_params':{
      "dynamic": False,
      "prediction_size": 1
  },

  "save_files":{
      "csv": "csv/ANN-d_h512.csv",
      "graph": "figs/graph-d_h512.jpg"
  }

}