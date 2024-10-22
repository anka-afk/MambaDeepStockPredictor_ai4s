```
# MambaStock: 用于股票预测的选择性状态空间模型

Mamba（带有选择机制和扫描模块的结构化状态空间序列模型，S6）在序列建模任务中取得了显著的成功。本文提出了一种基于 Mamba 的模型用于股票价格预测。

## 环境要求

代码在 Python 3.7.4 下测试通过，并安装了以下包及其依赖项：
```

numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
pandas==0.25.1
pytorch==1.7.1

```

本仓库使用的股票数据来自 [TuShare](https://tushare.pro/)。[TuShare](https://tushare.pro/) 上的股票数据是公开可用的。部分 Mamba 模型的代码来自 https://github.com/alxndrTL/mamba.py

## 使用方法

```

python main.py

````

## 选项

我们采用了 Python 中的 `argparse` 包来解析参数，运行代码的选项定义如下：

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='是否使用 CUDA 进行训练。')
parser.add_argument('--seed', type=int, default=1, help='随机种子。')
parser.add_argument('--epochs', type=int, default=100,
                    help='训练的轮数。')
parser.add_argument('--lr', type=float, default=0.01,
                    help='学习率。')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='权重衰减（参数的 L2 损失）。')
parser.add_argument('--hidden', type=int, default=16,
                    help='表示的维度。')
parser.add_argument('--layer', type=int, default=2,
                    help='层的数量。')
parser.add_argument('--n-test', type=int, default=300,
                    help='测试集的大小。')
parser.add_argument('--ts-code', type=str, default='601988',
                    help='股票代码。')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
````

## 引用

```
@article{shi2024mamba,
  title={MambaStock: Selective state space model for stock prediction},
  author={Zhuangwei Shi},
  journal={arXiv preprint arXiv:2402.18959},
  year={2024},
}
```

```

```
