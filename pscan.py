import math

import torch
import torch.nn.functional as F

"""
这是在 PyTorch 中实现的并行扫描操作（Blelloch 版本）。
"""


def npo2(len):
    """
    返回大于等于给定长度 len 的下一个 2 的幂
    """
    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    将输入的长度维度填充到下一个 2 的幂

    参数:
        X : (B, L, D, N)
        - B: 批次大小
        - L: 序列长度
        - D: 模型维度
        - N: 其他维度

    返回:
        Y : (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))  # 计算大于等于 L 的下一个 2 的幂
    # 构造填充元组，最后一个维度填充到 npo2(L)
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)  # 使用常数 0 进行填充


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A 的形状: (B, D, L, N)
        # X 的形状: (B, D, L, N)

        # 并行扫描操作，将 X 就地修改，公式为：
        # H[t] = A[t] * H[t-1] + X[t]，其中 H[0] = 0
        # 并行计算（理想情况下需要 2*log2(T) 步，而不是 T 步）

        # 只支持 L 为 2 的幂次（主要为了简化代码）

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # 向上的扫描阶段（最后两步展开）
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            # 将 Aa 和 Xa 的时间维度重塑为两部分
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            # 并行更新 Xa 和 Aa 的值
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]  # 保留下一级的值
            Xa = Xa[:, :, :, 1]

        # 剩余 4, 2 或 1 个节点
        if Xa.size(2) == 4:
            # 处理剩下的 4 个节点
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            # 处理剩下的 2 个节点
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            # 只有 1 个节点，直接返回
            return

        # 向下的扫描阶段（最初两步展开）
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        # 完成向下扫描的剩余步骤
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            # 并行更新 Xa 和 Aa 的值
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A 的形状: (B, D, L, N)
        # X 的形状: (B, D, L, N)

        # 该函数与前面的 pscan 功能相同，但方向相反
        # （如果你翻转输入，调用 pscan，然后再翻转输出，你就会得到该函数的结果）
        # 它在反向传播（backward pass）中使用

        # 只支持 L 为 2 的幂次（主要为了简化代码）

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # 向上的扫描阶段（最后两步展开）
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            # 将 Aa 和 Xa 的时间维度重塑为两部分
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            # 并行更新 Xa 和 Aa 的值
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]  # 保留左侧的值
            Xa = Xa[:, :, :, 0]

        # 剩余 4, 2 或 1 个节点
        if Xa.size(2) == 4:
            # 处理剩下的 4 个节点
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2])))
            )
        elif Xa.size(2) == 2:
            # 处理剩下的 2 个节点
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            # 只有 1 个节点，直接返回
            return

        # 向下的扫描阶段（最初两步展开）
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        # 完成向下扫描的剩余步骤
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            # 并行更新 Xa 和 Aa 的值
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        应用并行扫描操作，如上所定义的。返回一个新的张量。
        如果可能，优先使用序列长度为 2 的幂次的情况。

        参数:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        返回:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # 由于在-place操作，克隆输入张量
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # 如果 L 不是 2 的幂次，则进行填充并克隆张量
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        # 准备张量：转置维度以适应并行扫描
        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        # 并行扫描（在-place修改 X）
        PScan.pscan(A, X)

        # 保存输入以用于反向传播
        ctx.save_for_backward(A_in, X)

        # 切片 [:, :L]，如果进行了填充，则去除多余部分
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        从输出流向输入的梯度传播操作。返回两个新的张量。

        参数:
            ctx : 包含 A_in : (B, L, D, N) 和 X : (B, D, L, N) 的上下文
            grad_output_in : (B, L, D, N)

        返回:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # 由于 inplace 操作，需要克隆梯度输出
        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            # 如果 L 不是 2 的幂次，则进行填充并克隆张量
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        # 准备张量，转置维度
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)

        # 对 A_in 进行左移一位操作，向右填充一个零，以适应反向扫描
        A = torch.nn.functional.pad(
            A_in[:, :, 1:], (0, 0, 0, 1)
        )  # (B, D, npo2(L), N) 将 A 向左移动一位

        # 执行反向的并行扫描操作，修改 grad_output 就地（in-place）
        PScan.pscan_rev(A, grad_output)

        # 计算 Q 张量，用于保存中间结果
        Q = torch.zeros_like(X)
        # 对 Q 的非首元素进行更新，X 和 grad_output 的元素相乘
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        # 返回梯度张量 Q 和 grad_output，去掉填充部分
        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


# 将 PScan 应用于输入
pscan = PScan.apply
