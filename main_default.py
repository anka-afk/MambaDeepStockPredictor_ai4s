# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse

# 设置参数解析器，用于解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='是否使用 CUDA 进行训练。')
parser.add_argument('--seed', type=int, default=1, help='随机种子。')
parser.add_argument('--epochs', type=int, default=100, help='训练的轮数。')
parser.add_argument('--lr', type=float, default=0.01, help='学习率。')
parser.add_argument('--wd', type=float, default=1e-5, help='权重衰减（参数的 L2 损失）。')
parser.add_argument('--hidden', type=int, default=16, help='隐藏层维度大小。')
parser.add_argument('--layer', type=int, default=2, help='神经网络层数。')
parser.add_argument('--n-test', type=int, default=300, help='测试集的大小。')
parser.add_argument('--ts-code', type=str, default='601988', help='股票代码')

# 解析参数并设置 CUDA 使用情况
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

# 评价指标函数，计算并输出 MSE、RMSE、MAE、R2
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)  # 均方误差
    RMSE = MSE ** 0.5  # 均方根误差
    MAE = mean_absolute_error(y_test, y_hat)  # 平均绝对误差
    R2 = r2_score(y_test, y_hat)  # R2 分数
    print('%.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2))

# 设置随机种子函数，用于保证实验的可重复性
def set_seed(seed, cuda):
    np.random.seed(seed)  # Numpy 随机种子
    torch.manual_seed(seed)  # PyTorch 随机种子
    if cuda:
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子

# 显示数据集训练和测试集的时间范围
def dateinf(series, n_test):
    lt = len(series)
    print('Training start', series[0])  # 训练集开始时间
    print('Training end', series[lt - n_test - 1])  # 训练集结束时间
    print('Testing start', series[lt - n_test])  # 测试集开始时间
    print('Testing end', series[lt - 1])  # 测试集结束时间

# 设置随机种子
set_seed(args.seed, args.cuda)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),  # 输入层到隐藏层的线性变换
            Mamba(self.config),  # 使用 Mamba 层
            nn.Linear(args.hidden, out_dim),  # 隐藏层到输出层的线性变换
            nn.Tanh()  # Tanh 激活函数
        )
    
    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()

# 训练和预测函数
def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]), 1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)  # Adam 优化器
    xt = torch.from_numpy(trainX).float().unsqueeze(0)  # 训练特征转换为张量
    xv = torch.from_numpy(testX).float().unsqueeze(0)  # 测试特征转换为张量
    yt = torch.from_numpy(trainy).float()  # 训练标签转换为张量
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)  # 前向传播
        loss = F.mse_loss(z, yt)  # 均方误差损失
        opt.zero_grad()
        loss.backward()  # 反向传播
        opt.step()  # 更新权重
        if e % 10 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda:
        mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()  # 转换为 numpy 数组
    return yhat

# 读取数据
data = pd.read_csv(args.ts_code + '.SH.csv')  # 加载指定股票代码的数据
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')  # 将交易日期转换为日期格式

# 提取 'close' 列作为收盘价格
close = data.pop('close').values

# 计算 'ratechg'，即变化率
ratechg = data['pct_chg'].apply(lambda x: 0.01 * x).values

# 删除不需要的列
data.drop(columns=['pre_close', 'change', 'pct_chg'], inplace=True)

# 提取特征数据
dat = data.iloc[:, 2:].values

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

# 输出训练和测试数据的日期范围
dateinf(data['trade_date'], args.n_test)
print('MSE RMSE MAE R2')
evaluation_metric(data1, finalpredicted_stock_price)

# 绘制实际股票价格与预测价格的对比图
plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')  # 实际股票价格
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')  # 预测股票价格
plt.title('Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
