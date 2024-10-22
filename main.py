import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument(
    "--use-cuda", default=False, action="store_true", help="是否使用 CUDA 进行训练。"
)
parser.add_argument("--seed", type=int, default=1, help="随机种子。")
parser.add_argument("--epochs", type=int, default=100, help="训练的轮数。")
parser.add_argument("--lr", type=float, default=0.01, help="学习率。")
parser.add_argument(
    "--wd", type=float, default=1e-5, help="权重衰减（参数的 L2 正则化）。"
)
parser.add_argument("--hidden", type=int, default=16, help="表示维度的大小。")
parser.add_argument("--layer", type=int, default=2, help="层的数量。")
parser.add_argument("--n-test", type=int, default=300, help="测试集的大小。")
parser.add_argument("--ts-code", type=str, default="601988", help="股票代码。")

# 解析命令行参数
args = parser.parse_args()

# 判断是否使用 CUDA
args.cuda = args.use_cuda and torch.cuda.is_available()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch


def evaluation_metric(y_test, y_hat):
    """
    计算并打印评估指标，包括 MSE、RMSE、MAE 和 R2 分数。

    参数:
        y_test : 实际值
        y_hat : 预测值
    """
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print(f"MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, R2: {R2:.4f}")


def set_seed(seed, cuda):
    """
    设置随机种子以确保结果可复现。

    参数:
        seed : 随机种子值
        cuda : 是否使用 CUDA
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def dateinf(series, n_test):
    """
    输出训练和测试集的起始和结束日期信息。

    参数:
        series : 时间序列
        n_test : 测试集大小
    """
    lt = len(series)
    print("Training start:", series[0])
    print("Training end:", series[lt - n_test - 1])
    print("Testing start:", series[lt - n_test])
    print("Testing end:", series[lt - 1])


# 设置随机种子
set_seed(args.seed, args.cuda)


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 创建 Mamba 模型的配置
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        # 定义网络结构，包括输入、Mamba 层、输出层和激活函数
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),  # 输入层，将输入维度转换为隐藏层维度
            Mamba(self.config),  # Mamba 模型
            nn.Linear(args.hidden, out_dim),  # 输出层，将隐藏层维度转换为输出维度
            nn.Tanh(),  # Tanh 激活函数
        )

    def forward(self, x):
        # 前向传播
        x = self.mamba(x)
        return x.flatten()  # 将输出展平成一维


def PredictWithData(trainX, trainy, testX):
    # 创建模型实例，输入维度为 trainX 的特征数，输出维度为 1
    clf = Net(len(trainX[0]), 1)

    # Adam 优化器，使用学习率和权重衰减从 args 中获取
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)

    # 将训练集和测试集转换为 PyTorch 张量，并添加 batch 维度
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()

    # 如果使用 CUDA，将模型和张量移至 GPU
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()

    # 训练循环，基于 epochs 的数量
    for e in range(args.epochs):
        clf.train()  # 设置模型为训练模式
        z = clf(xt)  # 前向传播
        loss = F.mse_loss(z, yt)  # 计算均方误差损失

        opt.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        opt.step()  # 更新模型参数

        # 每 10 个 epoch 输出一次损失
        if e % 10 == 0 and e != 0:
            print("Epoch %d | Loss: %.4f" % (e, loss.item()))

    # 测试阶段，将模型设置为评估模式
    clf.eval()
    mat = clf(xv)  # 对测试数据进行预测

    # 如果使用了 CUDA，需将结果移回 CPU
    if args.cuda:
        mat = mat.cpu()

    # 将预测结果转换为 NumPy 数组并展平为一维
    yhat = mat.detach().numpy().flatten()

    return yhat


# 加载数据
data = pd.read_csv(args.ts_code + ".SH.csv")
data["trade_date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d")

# 获取收盘价和变化率
close = data.pop("close").values
ratechg = data["pct_chg"].apply(lambda x: 0.01 * x).values

# 删除无关列
data.drop(columns=["pre_close", "change", "pct_chg"], inplace=True)

# 获取训练和测试数据
dat = data.iloc[:, 2:].values
trainX, testX = dat[: -args.n_test, :], dat[-args.n_test :, :]
trainy = ratechg[: -args.n_test]

# 使用模型进行预测
predictions = PredictWithData(trainX, trainy, testX)

# 获取测试时间和实际收盘价
time = data["trade_date"][-args.n_test :]
data1 = close[-args.n_test :]

# 根据预测的变化率计算最终预测的股票价格
finalpredicted_stock_price = []
pred = close[-args.n_test - 1]
for i in range(args.n_test):
    pred = close[-args.n_test - 1 + i] * (1 + predictions[i])
    finalpredicted_stock_price.append(pred)

# 输出训练和测试集的日期信息
dateinf(data["trade_date"], args.n_test)

# 打印评估指标
print("MSE RMSE MAE R2")
evaluation_metric(data1, finalpredicted_stock_price)

# 绘制实际股票价格和预测股票价格的对比图
plt.figure(figsize=(10, 6))
plt.plot(time, data1, label="Stock Price")  # 实际股票价格
plt.plot(
    time, finalpredicted_stock_price, label="Predicted Stock Price"
)  # 预测股票价格
plt.title("Stock Price Prediction")
plt.xlabel("Time", fontsize=12, verticalalignment="top")
plt.ylabel("Close", fontsize=14, horizontalalignment="center")
plt.legend()
plt.show()
