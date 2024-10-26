# MambaDeepStockPredictor: 用于股票预测的选择性状态空间模型

Mamba（带有选择机制和扫描模块的结构化状态空间序列模型，S6）在序列建模任务中取得了显著的成功。这里我们将其用于 ai4s 竞赛。

## 环境要求

代码在 Python 3.7.4 下测试通过，并安装了以下包及其依赖项：

```

numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
pandas==0.25.1
pytorch==1.7.1

```

本仓库使用的股票数据为 CSI300。

## 使用方法

```

python main.py

```
