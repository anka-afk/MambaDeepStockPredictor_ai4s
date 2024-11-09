import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse
from pandas.plotting import register_matplotlib_converters
import glob

register_matplotlib_converters()

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True, help='是否使用 CUDA 进行训练。')
parser.add_argument('--seed', type=int, default=1, help='随机种子。')
parser.add_argument('--epochs', type=int, default=100, help='训练的轮数。')
parser.add_argument('--lr', type=float, default=0.01, help='学习率。')
parser.add_argument('--wd', type=float, default=1e-5, help='权重衰减（参数的 L2 损失）。')
parser.add_argument('--hidden', type=int, default=16, help='表示的维度。')
parser.add_argument('--layer', type=int, default=2, help='层的数量。')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start', series[0])
    print('Training end', series[lt - n_test - 1])
    print('Testing start', series[lt - n_test])
    print('Testing end', series[lt - 1])

set_seed(args.seed, args.cuda)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden, out_dim),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]), 1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda:
        mat = mat.cpu()
    return mat.detach().numpy().flatten()[0]

# 替换原有的数据处理和预测部分
data_list = []  # 用于存储所有股票的预测结果

# 读取所有股票代码
stock_files = glob.glob('stock/merged_stock data/*.csv')  # 需要在文件开头添加 import glob

for stock_file in stock_files:
    ts_code = stock_file.split('/')[-1].replace('.csv', '')
    
    # 读取数据
    data = pd.read_csv(stock_file)
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data.sort_values('trade_date').reset_index(drop=True)
    
    # 提取特征和目标值
    ratechg = data['pct_chg'].apply(lambda x: 0.01 * x).values
    
    # 删除不需要的列
    data.drop(columns=['pre_close', 'change', 'pct_chg', 'close'], inplace=True)
    
    # 提取特征数据
    features = [
        # 基础交易数据
        'open', 'high', 'low', 'vol', 'amount',
        
        # 市场表现指标
        'turnover_rate', 'turnover_rate_f', 'volume_ratio',
        
        # 估值指标
        'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 
        
        # 公司基本面
        'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
        
        # 技术指标
        'ma_bfq_5', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_30',
        'ema_bfq_5', 'ema_bfq_10', 'ema_bfq_20',
        'macd_dif_bfq', 'macd_dea_bfq', 'macd_bfq',
        'kdj_k_bfq', 'kdj_d_bfq',
        'rsi_bfq_6', 'rsi_bfq_12',
        'boll_upper_bfq', 'boll_mid_bfq', 'boll_lower_bfq',
        'vr_bfq',
        'obv_bfq'
    ]
    
    # 检查并移除完全缺失的特征
    valid_features = []
    for feature in features:
        if not data[feature].isna().all():  # 如果该列不是全部缺失
            valid_features.append(feature)
        else:
            print(f"警告：特征 {feature} 在股票 {ts_code} 中完全缺失，将被移除")
    
    features = valid_features  # 更新特征列表

    def handle_outliers(df, columns, n_sigmas=3):
        """处理异常值"""
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(mean - n_sigmas * std, mean + n_sigmas * std)
        return df

    # 在填充缺失值之前先处理异常值
    data = handle_outliers(data, features)

    # 然后进行缺失值填充
    for feature in features:
        data[feature] = data[feature].fillna(method='ffill').fillna(method='bfill')
   
    dat = data[features].values
    
    # 使用所有历史数据进行训练
    trainX, testX = dat[:-1, :], dat[-1:, :]  # 最后一天的数据用于预测
    trainy = ratechg[:-1]
    
    # 预测下一个交易日的涨跌幅
    pred_pct_chg = PredictWithData(trainX, trainy, testX) * 100  # 转换回百分比
    
    # 存储结果
    data_list.append({
        'ts_code': ts_code,
        'pct_chg': pred_pct_chg
    })

# 生成结果表格
result_df = pd.DataFrame(data_list)
print(result_df)
# 可选：保存到文件
result_df.to_csv('predictions.csv', index=False)

# 删除原有的评估和绘图代码


