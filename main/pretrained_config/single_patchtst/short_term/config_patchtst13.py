import torch
import torch.nn.functional as F
import torchmetrics
from models.ANN import ANN
from models.LSTM import StatefulLSTM, StatelessLSTM
from models.Transformer import PatchTST

config = {
  "use_single_channel": True, #False일 경우 멀티채널
  'dataset_setting':{
    "main_csv": "/home/dataset/complete_dataset.csv",
    "time_axis": "일시",
    "target": "PM-2.5"
  },
  "window_params":{
    "patch_length": 16, #고정
    "n_patches": 24,
    "forecast_size": 7
  },
  "tst_size": 200,
  
  'model': PatchTST, # or RandomForestRegressor
  'model_params': {
    "model_dim": 32, 
    "num_heads": 4, 
    "num_layers": 6
  },
  
  
  'train_params': {
    'data_loader_params': {
      'batch_size': 64,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001, # 0.001~0.0001
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cuda',
    'epochs': 100, # 10~100이하
  },
  
  'eval_params':{
      "dynamic": False, 
      "prediction_size": 1
  },

  "save_files":{
      "csv": "csv/single_patchtst/patchtst13.csv",
      "day": "figs/single_patchtst/everyday/graph13.jpg",
      "peak": "figs/single_patchtst/peakday/config13/"
  }
}