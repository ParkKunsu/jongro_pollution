from typing import Literal, List
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd

@dataclass
class HomeData:
    file_trn: str = '/home/data/train.csv'
    file_tst: str = '/home/data/test_eda.csv'
    target_col: str = 'target'
    features: List[str] = field(default_factory=list)
    encoding_columns: List[str] = field(default_factory=list)
    scaler: Literal['None', 'standard', 'minmax'] = 'None'
    scale_columns: List[str] = field(default_factory=list)  # 수정: field 함수 사용

    def _read_df(self, split: Literal['train', 'test'] = 'train'):
        if split == 'train':
            df = pd.read_csv(self.file_trn)
            df_X = df[self.features]
            target = df[self.target_col]
            return df_X, target
        elif split == 'test':
            df = pd.read_csv(self.file_tst)
            df_X = df[self.features]  # df 대신 df_X를 반환
            return df_X
        raise ValueError(f'"{split}"은(는) 허용되지 않습니다.')

    def preprocess(self):
        X_trn, y_trn = self._read_df(split="train")
        X_tst = self._read_df(split="test")

        # 스케일러 초기화
        if self.scaler == 'standard':
            scaler = StandardScaler()
        elif self.scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None  # 스케일러가 None인 경우도 처리
        # 특정 컬럼에만 스케일러 적용
        if scaler:

            # 주석: 스케일러 적용할 컬럼들만 추출하여 스케일링 진행
            X_trn[self.scale_columns] = scaler.fit_transform(X_trn[self.scale_columns])
            X_tst[self.scale_columns] = scaler.transform(X_tst[self.scale_columns])

        X_trn = pd.get_dummies(X_trn, columns=self.encoding_columns)
        X_tst = pd.get_dummies(X_tst, columns=self.encoding_columns)

        return X_trn, y_trn, X_tst

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":

  args = get_args_parser().parse_args()
  
  exec(open(args.config).read())
  cfg = config

  preprocess_params = cfg.get('preprocess')

  args = get_args_parser().parse_args()
  home_data = HomeData(
    features=preprocess_params.get('features'),
    file_trn=preprocess_params.get('train-csv'),
    file_tst=preprocess_params.get('test-csv'),
    target_col=preprocess_params.get('target-col'),
    scaler=preprocess_params.get('scaler'),
    scale_columns=preprocess_params.get('scale-columns'),  # Use the correct attribute name
    encoding_columns=preprocess_params.get('encoding-columns'),  # Use the correct attribute name
  )
  trn_X, trn_y, tst_X = home_data.preprocess()

  trn_X.to_csv(preprocess_params.get('output-train-feas-csv'))
  tst_X.to_csv(preprocess_params.get('output-test-feas-csv'))
  trn_y.to_csv(preprocess_params.get('output-train-target-csv'))