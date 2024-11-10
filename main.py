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
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

try:
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    chinese_font = fm.FontProperties(fname=font_path)
except:
    try:
        font_path = '/System/Library/Fonts/PingFang.ttc'
        chinese_font = fm.FontProperties(fname=font_path)
    except:
        try:
            font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
            chinese_font = fm.FontProperties(fname=font_path)
        except:
            print("警告：未能找到合适的中文字体文件")
            chinese_font = None

register_matplotlib_converters()

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True, help='是否使用 CUDA 进行训练。')
parser.add_argument('--seed', type=int, default=1, help='随机种子。')
parser.add_argument('--epochs', type=int, default=100, help='训练的轮数。')
parser.add_argument('--lr', type=float, default=0.01, help='学习率。')
parser.add_argument('--wd', type=float, default=1e-5, help='权重衰减（参数的 L2 损失）。')
parser.add_argument('--hidden', type=int, default=16, help='表示的维度。')
parser.add_argument('--layer', type=int, default=2, help='层的数量。')
parser.add_argument('--n-test', type=int, default=300, help='测试集的大小。')
parser.add_argument('--ts-code', type=str, default='601988.SH', help='股票代码。')

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

data = pd.read_csv('stock/merged_stock data/' + args.ts_code + '.csv')

data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.sort_values('trade_date').reset_index(drop=True)

close = data.pop('close').values

ratechg = data['pct_chg'].apply(lambda x: 0.01 * x).values

data.drop(columns=['pre_close', 'change', 'pct_chg'], inplace=True)

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

available_features = []
for col in features:
    if col in data.columns:
        available_features.append(col)
    else:
        print(f"警告: 特征 {col} 在数据中不存在，将被忽略")

features = available_features

# 打印处理前的缺失值情况
# print("处理前的缺失值统计：")
# print(data[features].isnull().sum())

for feature in features:
    # 1. 首先使用前向填充(forward fill)处理连续缺失值
    data[feature] = data[feature].fillna(method='ffill')
    
    # 2. 对于序列开始处的缺失值，使用后向填充(backward fill)
    data[feature] = data[feature].fillna(method='bfill')

# 打印处理后的缺失值情况
# print("\n处理后的缺失值统计：")
# print(data[features].isnull().sum())


dat = data[features].values

# 划分训练集和测试集
trainX, testX = dat[:-args.n_test, :], dat[-args.n_test:, :]
trainy = ratechg[:-args.n_test]

# 模型训练和预测
predictions = PredictWithData(trainX, trainy, testX)

# 评估和可视化
time = data['trade_date'][-args.n_test:]
data1 = close[-args.n_test:]
finalpredicted_stock_price = []
pred = close[-args.n_test - 1]
for i in range(args.n_test):
    pred = close[-args.n_test - 1 + i] * (1 + predictions[i])
    finalpredicted_stock_price.append(pred)

dateinf(data['trade_date'], args.n_test)
print('MSE RMSE MAE R2')
evaluation_metric(data1, finalpredicted_stock_price)

data = data.sort_values('trade_date')

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['lines.linewidth'] = 2

fig, ax = plt.subplots()

ax.plot(data['trade_date'][-args.n_test:], data1, 
        label='实际股价', 
        color='#2E86C1',
        alpha=0.8)
ax.plot(data['trade_date'][-args.n_test:], finalpredicted_stock_price, 
        label='预测股价', 
        color='#E74C3C',
        linestyle='--',
        alpha=0.8)

ax.grid(True, linestyle='--', alpha=0.7)

if chinese_font:
    ax.set_title(f'{args.ts_code} 股价预测分析', 
                fontproperties=chinese_font,
                fontsize=16, 
                pad=20, 
                fontweight='bold')
    ax.set_xlabel('时间', fontproperties=chinese_font, fontsize=12)
    ax.set_ylabel('股价 (元)', fontproperties=chinese_font, fontsize=12)
    ax.legend(loc='upper left', 
             prop=chinese_font,
             frameon=True, 
             fancybox=True, 
             shadow=True, 
             fontsize=10)
else:
    ax.set_title(f'{args.ts_code} Stock Price Prediction', 
                fontsize=16, 
                pad=20, 
                fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price (CNY)', fontsize=12)
    ax.legend(loc='upper left', 
             frameon=True, 
             fancybox=True, 
             shadow=True, 
             fontsize=10)

plt.xticks(rotation=30, ha='right')
ax.xaxis.set_major_locator(plt.MaxNLocator(10))

plt.tight_layout()

if chinese_font:
    fig.text(0.99, 0.01, 'MDSP', 
             fontproperties=chinese_font,
             ha='right', 
             va='bottom', 
             alpha=0.4, 
             fontsize=8)
else:
    fig.text(0.99, 0.01, 'MDSP', 
             ha='right', 
             va='bottom', 
             alpha=0.4, 
             fontsize=8)

plt.show()


