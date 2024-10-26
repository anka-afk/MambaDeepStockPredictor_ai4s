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

register_matplotlib_converters()

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='是否使用 CUDA 进行训练。')
parser.add_argument('--seed', type=int, default=1, help='随机种子。')
parser.add_argument('--epochs', type=int, default=300, help='训练的轮数。')
parser.add_argument('--lr', type=float, default=0.01, help='学习率。')
parser.add_argument('--wd', type=float, default=1e-5, help='权重衰减（参数的 L2 损失）。')
parser.add_argument('--hidden', type=int, default=16, help='表示的维度。')
parser.add_argument('--layer', type=int, default=2, help='层的数量。')
parser.add_argument('--n-test', type=int, default=100, help='测试集的大小。')
parser.add_argument('--ts-code', type=str, default='000001.SZ', help='股票代码。')

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
    yhat = mat.detach().numpy().flatten()
    return yhat

# 读取数据
data = pd.read_csv('stock/stock data/' + args.ts_code + '.csv')

# 将 'trade_date' 列转换为日期时间格式
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.sort_values('trade_date').reset_index(drop=True)

# 计算 'ratechg'，使用前复权收盘价计算变化率
data['ratechg'] = data['change']/ data['pre_close']

# 删除不需要的列
data.drop(columns=['pre_close', 'change', 'pct_chg'], inplace=True)

# 明确选择特征列
features = ['open', 'high', 'low', 'vol', 'amount', 'turnover_rate', 'volume_ratio', 'pe', 'pb', 'ps', 
            'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']

# 检查特征列是否存在
for col in features:
    if col not in data.columns:
        print(f"Column {col} not found in data")

# 提取特征数据
dat = data[features].values

# 划分训练集和测试集
trainX, testX = dat[:-args.n_test, :], dat[-args.n_test:, :]
trainy = data['ratechg'][:-args.n_test].values

# 模型训练和预测
predictions = PredictWithData(trainX, trainy, testX)

# 评估和可视化
time = data['trade_date'][-args.n_test:]
actual_ratechg = data['ratechg'][-args.n_test:].values

dateinf(data['trade_date'], args.n_test)
print('MSE RMSE MAE R2')
evaluation_metric(actual_ratechg, predictions)

# 绘图部分，展示实际和预测的收盘价变化率
plt.figure(figsize=(10, 6))
plt.plot(time, actual_ratechg, label='Actual Rate Change')
plt.plot(time, predictions, label='Predicted Rate Change')
plt.title('Stock Rate Change Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Rate Change', fontsize=14, horizontalalignment='center')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
