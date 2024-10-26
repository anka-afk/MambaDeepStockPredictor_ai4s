import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan

"""

该文件紧密遵循了官方 Mamba 实现中的 mamba_simple.py 以及 @johnma2006 的 mamba-minimal。
主要区别是：
- 卷积操作使用了 torch.nn.Conv1d
- 选择性扫描是在 PyTorch 中完成的

一个用于对比的顺序版选择性扫描也可用。

- Mamba 模型由多个层组成，每一层是一个 ResidualBlock（残差块）。
- ResidualBlock 由一个 MambaBlock、归一化操作和残差连接组成：ResidualBlock(x) = mamba(norm(x)) + x
- 这使得我们关注 MambaBlock：它的输入 x 是 (B, L, D)，输出 y 也是 (B, L, D) （B=批大小，L=序列长度，D=模型维度）。
首先，我们将 x 扩展为 (B, L, 2*ED)（其中 E 通常为 2），并将其分成 x 和 z，每个为 (B, L, ED)。
然后，对 x 应用短 1D 卷积，之后是激活函数（silu），然后是 SSM。
接着，将其与 silu(z) 相乘。

"""


@dataclass
class MambaConfig:
    d_model: int  # 模型的维度（D）
    n_layers: int  # 模型中的层数
    dt_rank: Union[int, str] = (
        "auto"  # 时间步长矩阵的秩，可以是整数或者 'auto'，自动计算
    )
    d_state: int = 16  # 状态维度（N），论文中的符号
    expand_factor: int = 2  # 扩展因子（E），论文中的符号
    d_conv: int = 4  # 卷积层的维度

    dt_min: float = 0.001  # 时间步长的最小值
    dt_max: float = 0.1  # 时间步长的最大值
    dt_init: str = (
        "random"  # 时间步长的初始值类型，"random" 表示随机初始化，"constant" 表示常量初始化
    )
    dt_scale: float = 1.0  # 时间步长的缩放因子
    dt_init_floor = 1e-4  # 时间步长初始化的下限值

    bias: bool = False  # 是否在模型中使用偏置项
    conv_bias: bool = True  # 卷积层是否使用偏置项

    pscan: bool = (
        True  # 是否在训练时使用并行扫描模式，True 为并行扫描，False 为顺序扫描
    )

    def __post_init__(self):
        # d_inner 代表扩展后的模型维度，等于扩展因子（E）乘以模型维度（D）
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED，论文中的符号

        # 如果 dt_rank 被设置为 'auto'，则自动将 dt_rank 设为 d_model 的 1/16，向上取整
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        # 保存配置文件
        self.config = config

        # 创建 n_layers 层 ResidualBlock（残差块），并存入 nn.ModuleList 中
        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

        # 使用 RMSNorm 对输出进行归一化处理
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # 如果输入是二维的，添加一个时间步维度
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # x 的形状: (B, L, D)
        # B: 批大小, L: 序列长度, D: 模型维度

        # 逐层通过每个 ResidualBlock
        for layer in self.layers:
            x = layer(x)

        # 最终输出进行归一化处理
        x = self.norm_f(x)

        # 如果需要，可以在这里去掉时间步维度
        return x.squeeze(1)

    def step(self, x, caches):
        # x 的形状: (B, L, D)
        # caches: 每一层对应的缓存状态，缓存格式为 (h, inputs)

        # 逐层处理，并更新每一层的缓存
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches  # 返回更新后的输出和缓存


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        # MambaBlock 用于执行主要的计算
        self.mixer = MambaBlock(config)

        # RMSNorm 用于对输入进行归一化处理
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x 的形状: (B, L, D)
        # B: 批大小, L: 序列长度, D: 模型维度

        # 执行归一化，然后通过 MambaBlock 处理，并添加残差连接
        output = self.mixer(self.norm(x)) + x
        return output  # 输出的形状仍然是 (B, L, D)

    def step(self, x, cache):
        # x 的形状: (B, D)  注意这里是每个时间步的输入
        # cache: 缓存状态，包含 h 和 inputs
        # h: 缓存状态，形状为 (B, ED, N)，ED 是扩展后的维度
        # inputs: 之前的输入，形状为 (B, ED, d_conv-1)

        # 逐步执行操作，将归一化后的 x 输入 MambaBlock，并传入缓存
        output, cache = self.mixer.step(self.norm(x), cache)

        # 添加残差连接
        output = output + x
        return output, cache  # 返回更新后的输出和缓存


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        # 保存配置文件
        self.config = config

        # 将输入从维度 D 投影到 2*ED（两个分支）
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # 一维卷积操作，对扩展后的维度 ED 进行处理
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,  # 深度卷积，每个通道独立卷积
            padding=config.d_conv - 1,
        )  # 确保输出的长度与输入一致

        # 投影 x 以生成与输入相关的 Δ、B、C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # 投影 Δ（时间步长）从 dt_rank 到 d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt 初始化（时间步长）
        # dt 权重初始化
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)  # 常量初始化
        elif config.dt_init == "random":
            nn.init.uniform_(
                self.dt_proj.weight, -dt_init_std, dt_init_std
            )  # 随机初始化
        else:
            raise NotImplementedError

        # dt 偏置初始化
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(
            min=config.dt_init_floor
        )  # 确保 dt 不低于初始化下限
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # 计算 softplus 的逆函数
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)  # 将计算好的 dt 偏置赋值

        # S4D 真实部分的初始化
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # 将 A 存储为 log 形式，确保 A 为负数以稳定梯度
        self.D = nn.Parameter(torch.ones(config.d_inner))  # D 为标量参数

        # 将模块的输出从 ED 投影回 D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x 的形状: (B, L, D)
        # B: 批大小, L: 序列长度, D: 模型维度

        # 输出 y 的形: (B, L, D)

        _, L, _ = x.shape

        # 通过 in_proj 将输入投影为 2 * ED 的维度
        xz = self.in_proj(x)  # 形状: (B, L, 2*ED)

        # 将投影的结果分为 x 和 z 两个分支
        x, z = xz.chunk(2, dim=-1)  # 形状: (B, L, ED), (B, L, ED)

        # 处理 x 分支
        x = x.transpose(1, 2)  # 变换维度为 (B, ED, L)

        # 对时间维度执行深度卷积操作，滤波器较短
        x = self.conv1d(x)[:, :, :L]  # 形状仍为 (B, ED, L)

        # 将卷积后的结果恢复为 (B, L, ED)
        x = x.transpose(1, 2)

        # 使用 silu 激活函数
        x = F.silu(x)

        # 应用状态空间模型 (SSM) 操作
        y = self.ssm(x)

        # 处理 z 分支
        z = F.silu(z)

        # 将 x 和 z 的结果相乘
        output = y * z

        # 通过 out_proj 将结果投影回原始维度 D
        output = self.out_proj(output)  # 形状为 (B, L, D)

        return output

    def ssm(self, x):
        # x 的形状: (B, L, ED)

        # 输出 y 的形状: (B, L, ED)

        # 使用 log 存储的 A 参数，确保 A 为负值以保持稳定性
        A = -torch.exp(self.A_log.float())  # 形状: (ED, N)
        D = self.D.float()

        # 通过 x_proj 投影生成 delta、B、C
        deltaBC = self.x_proj(x)  # 形状: (B, L, dt_rank + 2*N)

        # 将 deltaBC 分为 delta、B、C 三部分
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)

        # 使用 softplus 函数计算 delta
        delta = F.softplus(self.dt_proj(delta))  # 形状: (B, L, ED)

        # 根据配置选择并行或顺序扫描模式
        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x 的形状: (B, L, ED)
        # Δ 的形状: (B, L, ED)
        # A 的形状: (ED, N)
        # B 的形状: (B, L, N)
        # C 的形状: (B, L, N)
        # D 的形状: (ED)

        # 输出 y 的形状: (B, L, ED)

        # 计算 delta 和 A 的乘积，并取指数，形状为 (B, L, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)

        # 计算 delta 和 B 的乘积，形状为 (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        # 计算 BX，形状为 (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        # 使用并行扫描函数 pscan 处理 deltaA 和 BX
        hs = pscan(deltaA, BX)

        # 通过矩阵乘法将 hs 和 C 相乘，输出形状为 (B, L, ED)
        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # 添加残差连接，D 乘以输入 x
        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x 的形状: (B, L, ED)
        # Δ 的形状: (B, L, ED)
        # A 的形状: (ED, N)
        # B 的形状: (B, L, N)
        # C 的形状: (B, L, N)
        # D 的形状: (ED)

        # 输出 y 的形状: (B, L, ED)

        _, L, _ = x.shape

        # 计算 delta 和 A 的乘积，并取指数，形状为 (B, L, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)

        # 计算 delta 和 B 的乘积，形状为 (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        # 计算 BX，形状为 (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        # 初始化 h 为全零张量，形状为 (B, ED, N)
        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, ED, N)

        # 保��每个时间步的 h 值
        hs = []

        # 逐步执行顺序扫描
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]  # 逐步更新 h
            hs.append(h)  # 将每个时间步的 h 保存下来

        # 将 hs 堆叠为 (B, L, ED, N) 的张量
        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        # 通过矩阵乘法将 hs 和 C 相乘，输出形状为 (B, L, ED)
        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # 添加残差连接，D 乘以输入 x
        y = y + D * x

        return y

    # -------------------------- 推理（inference） -------------------------- #
    """
    关于自回归推理

    Mamba 模型的一个酷点在于：推理的复杂度与序列长度无关。
    我们只需要为每一层保存两样东西：
    - 隐藏状态 h（形状为 (B, ED, N)），类似于使用 RNN 进行推理时保存隐藏状态。
    - 该层的最后 d_conv-1 个输入，以便在时间维度上进行 1D 卷积。
    （d_conv 是固定的，因此缓存不会随着序列生成的进行而增长）
    （通常 d_conv 很小，例如 4，所以我们只需要“记住”最后 3 个输入）

    具体来说，这两种量被放入缓存元组中，分别命名为 h 和 inputs。
    h 的形状为 (B, ED, N)，而 inputs 的形状为 (B, ED, d_conv-1)。
    `MambaBlock.step()` 函数接收这个缓存，并且除了输出结果之外，还输出更新后的缓存以供下一次调用。

    缓存对象初始化为：(None, torch.zeros())。
    当 h 为 None 时，selective scan 函数会检测到它，并从 h=0 开始。
    `torch.zeros()` 不会有问题（这与直接提供输入效果相同，因为 conv1d 是带填充的）。

    由于每一层都需要一个这样的缓存变量，我们存储一个 caches 对象，该对象实际上是一个缓存对象的列表。（参见 mamba_lm.py）
    """

    def step(self, x, cache):
        # x 的形状: (B, D)
        # cache 包含两个部分: h（隐藏状态）和 inputs（卷积输入的缓存）
        # h 的形状为 (B, ED, N)
        # inputs 的形状为 (B, ED, d_conv-1)

        # 输出 y 的形状: (B, D)
        # 更新后的 cache 形状: (h, inputs)

        h, inputs = cache

        # 将输入 x 投影为 2 * ED 的维度并拆分为 x 和 z 两个分支
        xz = self.in_proj(x)  # 形状: (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  # 形状: (B, ED), (B, ED)

        # x 分支
        x_cache = x.unsqueeze(2)  # 扩展 x 以匹配卷积的输入维度
        # 将输入与缓存的 inputs 拼接，然后进行一维卷积操作，卷积滤波器大小为 d_conv，取最后的结果
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[
            :, :, self.config.d_conv - 1
        ]  # 形状: (B, ED)

        # 通过 silu 激活函数
        x = F.silu(x)

        # 执行 SSM 操作，更新隐藏状态 h
        y, h = self.ssm_step(x, h)

        # z 分支
        z = F.silu(z)

        # 结合 x 和 z 的输出
        output = y * z

        # 通过 out_proj 投影回原始维度 D
        output = self.out_proj(output)  # 形状: (B, D)

        # 更新 cache，将输入缓存更新为最新的 d_conv-1 个输入
        inputs = torch.cat(
            [inputs[:, :, 1:], x_cache], dim=2
        )  # 形状: (B, ED, d_conv-1)
        cache = (h, inputs)  # 更新缓存

        return output, cache

    def ssm_step(self, x, h):
        # x 的形状: (B, ED)
        # h 的形状: (B, ED, N)

        # 输出 y 的形状: (B, ED)
        # 更新后的 h 的形状: (B, ED, N)

        # 使用存储为 log 的 A 确保其为负值
        A = -torch.exp(self.A_log.float())  # 形状: (ED, N)
        D = self.D.float()  # D 的形状: (ED)

        # 投影 x，生成 delta、B、C
        deltaBC = self.x_proj(x)  # 形状: (B, dt_rank+2*N)

        # ��分 deltaBC 为 delta、B 和 C
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # 形状: (B, dt_rank), (B, N), (B, N)

        # 使用 softplus 函数计算 delta
        delta = F.softplus(self.dt_proj(delta))  # 形状: (B, ED)

        # 计算 deltaA 和 deltaB，用于更新隐藏状态 h
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # 形状: (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # 形状: (B, ED, N)

        # 计算 BX，用于更新隐藏状态 h
        BX = deltaB * x.unsqueeze(-1)  # 形状: (B, ED, N)

        # 如果 h 为空，则初始化为零张量
        if h is None:
            h = torch.zeros(
                x.size(0),
                self.config.d_inner,
                self.config.d_state,
                device=deltaA.device,
            )  # 形状: (B, ED, N)

        # 更新隐藏状态 h
        h = deltaA * h + BX  # 形状: (B, ED, N)

        # 计算输出 y
        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        # 添加残差连接
        y = y + D * x

        return y, h.squeeze(1)  # 返回输出 y 和更新后的 h


# 直接来自 https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        # 初始化 epsilon 和可训练的权重参数
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # 权重参数，形状为 (d_model,)

    def forward(self, x):
        # 对输入 x 进行均方根归一化
        # x.pow(2).mean(-1, keepdim=True): 计算最后一个维度上平方后的均值
        # torch.rsqrt: 计算平方根的倒数（以实现归一化）
        # 加上 epsilon 防止除零
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output



