import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class HybridConv1d(nn.Module):
    """
    分支1：原深度可分离卷积 (Depthwise + Pointwise)
    分支2：大核卷积分支 (DepthwiseLarge + Pointwise)
    输出：两分支相加
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 bias=False, small_kernel=None):
        super().__init__()

        # ---- 分支 1：原始深度可分离卷积 ----
        self.depthwise_sep = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )

        # ---- 分支 2：大核深度卷积（需与 dilation 对齐） ----
        effective_kernel = (kernel_size - 1) * dilation + 1

        # depthwise 大核
        self.dw_large = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=effective_kernel,
            stride=stride,
            padding=padding,
            dilation=1,              # 已包含在 effective kernel 里
            groups=in_channels,      # depthwise
            bias=bias
        )

        # pointwise（1x1）
        self.pw_large = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

        # ---- 可选小核分支 ----
        if small_kernel is not None:
            small_eff = (small_kernel - 1) * dilation + 1
            small_pad = (small_eff - 1)

            self.dw_small = nn.Conv1d(
                in_channels, in_channels,
                kernel_size=small_eff,
                padding=small_pad,
                groups=in_channels,
                bias=bias
            )
            self.pw_small = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=1,
                bias=bias
            )
        else:
            self.dw_small = None

    def forward(self, x):
        # 原分支
        out1 = self.depthwise_sep(x)

        # 大核分支
        out2 = self.pw_large(self.dw_large(x))

        # 小核可选分支
        if self.dw_small is not None:
            out2 = out2 + self.pw_small(self.dw_small(x))

        return out1 + out2


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
            # 因果卷积的填充裁剪模块，裁剪掉卷积操作中右侧的多余填充，确保时间卷积的因果性，（即输出只依赖于当前及之前的输入，不依赖未来输入）
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size        # 裁剪尺寸的大小（通常与卷积的 padding 相等）

    def forward(self, x):
        """
        前向传播
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, channels, length]
        返回:
            Tensor: 裁剪后的张量，形状为 [batch_size, channels, length - chomp_size]

            # 定义前向传播逻辑，对输入张量的最后一个维度（即时间维度）进行裁剪右侧部分（去掉多余的填充）
            # [:, :, :-self.chomp_size] 表示：
                第一个维度：所有批次，第二个维度：所有通道，第三个维度：从开始到倒数第chomp_size个元素
                contiguous() 确保内存连续存储，提高后续操作效率
                从输入张量 x 中取时间维度的前 x.size(2) - chomp_size 部分，去掉多余的填充
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):     # TCN 的基本构造模块
    """
        时间卷积块（TCN的基本构建模块），包含两个卷积层，使用残差连接
        因果性裁剪（Chomp1d）：保证卷积结果不依赖未来信息
        激活函数和 Dropout：增加非线性和正则化
        残差连接：缓解梯度消失问题，提升训练稳定性
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        初始化时间卷积块
        参数:
            n_inputs:           输入通道数
            n_outputs:          输出通道数
            kernel_size:        卷积核的大小
            stride:             卷积步长
            dilation:           扩张率，用于扩展感受野（控制感受野大小）
            padding:            填充大小，确保输出序列长度与输入相同;通常为 (kernel_size - 1) * dilation
            dropout:            Dropout 概率，用于正则化
        """
        super(TemporalBlock, self).__init__()

        self.conv1 = HybridConv1d(n_inputs, n_outputs, kernel_size,    # 定义第一个卷积层，并使用 weight_norm 进行权重归一化
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)      # 创建裁剪模块，去掉右侧多余的填充
        self.relu1 = nn.ReLU()              # ReLU激活函数
        self.dropout1 = nn.Dropout(dropout) # Dropout层，用于正则化

        self.conv2 = HybridConv1d(n_outputs, n_outputs, kernel_size,   # 定义第二个卷积层
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,    # 将两个卷积层（包括卷积、裁剪、激活、Dropout）组合成序列顺序执行的网络
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None          # 残差连接处理：如果输入和输出通道数不同，使用 1x1 卷积调整通道数
        self.relu = nn.ReLU()   # 残差连接后的激活函数
        self.init_weights()     # 初始化权重

    def init_weights(self):
        """
        兼容初始化：支持原始 DepthwiseSeparableConv1d、我们新增的 HybridConv1d、
        以及可能存在的 ReparamLargeKernelConv 样式（不同属性名）。
        """

        def init_ds_conv(ds):
            # 原 DepthwiseSeparableConv1d (有 depthwise 和 pointwise)
            if hasattr(ds, 'depthwise') and hasattr(ds, 'pointwise'):
                if ds.depthwise is not None:
                    ds.depthwise.weight.data.normal_(0, 0.01)
                if ds.pointwise is not None:
                    ds.pointwise.weight.data.normal_(0, 0.01)

        def init_reparam_like(rp):
            # 兼容不同命名的 reparam / large-kernel 分支
            # 常见字段: depthwise_large, pointwise_large, depthwise_small, pointwise_small
            if hasattr(rp, 'depthwise_large'):
                try:
                    rp.depthwise_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pointwise_large'):
                try:
                    rp.pointwise_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'dw_large'):
                try:
                    rp.dw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pw_large'):
                try:
                    rp.pw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'depthwise_small') and getattr(rp, 'depthwise_small') is not None:
                try:
                    rp.depthwise_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pointwise_small') and getattr(rp, 'pointwise_small') is not None:
                try:
                    rp.pointwise_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'dw_small') and getattr(rp, 'dw_small') is not None:
                try:
                    rp.dw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(rp, 'pw_small') and getattr(rp, 'pw_small') is not None:
                try:
                    rp.pw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass

        def init_hybrid(h):
            # HybridConv1d: 初始化内部 depthwise_sep 与 large/small 分支
            if hasattr(h, 'depthwise_sep') and h.depthwise_sep is not None:
                init_ds_conv(h.depthwise_sep)
            # legacy name: reparam (older impl)
            if hasattr(h, 'reparam') and h.reparam is not None:
                init_reparam_like(h.reparam)
            # our newer naming:
            if hasattr(h, 'dw_large') and getattr(h, 'dw_large') is not None:
                try:
                    h.dw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'pw_large') and getattr(h, 'pw_large') is not None:
                try:
                    h.pw_large.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'dw_small') and getattr(h, 'dw_small') is not None:
                try:
                    h.dw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass
            if hasattr(h, 'pw_small') and getattr(h, 'pw_small') is not None:
                try:
                    h.pw_small.weight.data.normal_(0, 0.01)
                except Exception:
                    pass

        # 对 conv1/conv2 做通用初始化（递归检查）
        for conv in (self.conv1, self.conv2):
            # 如果是原 DepthwiseSeparableConv1d
            if isinstance(conv, DepthwiseSeparableConv1d) or (
                    hasattr(conv, 'depthwise') and hasattr(conv, 'pointwise')):
                init_ds_conv(conv)
            # 如果是我们新加的 HybridConv1d
            elif isinstance(conv, HybridConv1d) or hasattr(conv, 'depthwise_sep') or hasattr(conv, 'dw_large'):
                init_hybrid(conv)
            # 如果是直接的 ReparamLargeKernelConv 风格
            elif hasattr(conv, 'depthwise_large') or hasattr(conv, 'pointwise_large') or hasattr(conv,
                                                                                                 'get_equivalent_kernel_bias'):
                init_reparam_like(conv)
            else:
                # 最后兜底：尝试初始化通用 weight 属性（如果存在）
                if hasattr(conv, 'weight'):
                    try:
                        conv.weight.data.normal_(0, 0.01)
                    except Exception:
                        pass

        # downsample（如果存在）
        if self.downsample is not None:
            try:
                self.downsample.weight.data.normal_(0, 0.01)
            except Exception:
                pass

    def forward(self, x):
        """
        前向传播，包含主路径（self.net(x)）和残差连接（x 或 self.downsample(x)）的计算，最终通过 ReLU 激活输出
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, in_channels, seq_len]

        返回:
            Tensor: 输出张量，形状为 [batch_size, out_channels, seq_len]
        """
        out = self.net(x)               # 主路径：通过两个卷积层
        res = x if self.downsample is None else self.downsample(x)  # 残差路径：如果输入输出通道数相同，直接使用输入x；否则通过1x1卷积调整通道数
        return self.relu(out + res)


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        # 深度卷积 (每个输入通道单独卷积)
        self.depthwise = weight_norm(
            nn.Conv1d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        )
        # 逐点卷积 (1x1卷积融合通道信息)
        self.pointwise = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AttentionFusionBlock(nn.Module):
    """
    包含两个并行的、具有不同扩张率的残差块，并融合了通道注意力机制。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        """
        初始化注意力时间卷积块
        参数:
            n_inputs:           输入通道数
            n_outputs:          输出通道数
            kernel_size:        卷积核的大小
            stride:             卷积步长
            dilation:           基础扩张率 (d)，用于第一个分支
            dropout:            Dropout 概率
        """
        super(AttentionFusionBlock, self).__init__()

        # 两个扩张率分支
        self.branch1 = TemporalBlock(n_inputs, n_outputs, kernel_size, stride, dilation,
                                     (kernel_size - 1) * dilation, dropout)
        self.branch2 = TemporalBlock(n_inputs, n_outputs, kernel_size, stride, dilation * 2,
                                     (kernel_size - 1) * dilation * 2, dropout)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # "调整"层：使用FC层生成两个分支的权重
        self.attention_fc = nn.Sequential(
            nn.Linear(n_outputs, n_outputs // 4),  # 使用一个缩减层来降低计算复杂度
            nn.ReLU(inplace=True),
            nn.Linear(n_outputs // 4, n_outputs * 2)  # 输出通道数的两倍，用于两个分支
        )
        self.softmax = nn.Softmax(dim=2) # 在分支维度上应用softmax

        # --- 残差连接 ---
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # 两个并行分支的前向传播
        out1 = self.branch1(x)  # (B, C, L)
        out2 = self.branch2(x)

        # --- 注意力权重生成 ---
        # 1. 逐元素相加初步融合
        merged = out1 + out2    # (B, C, L)

        # 2. 全局池化和FC层调整
        gap = self.global_avg_pool(merged).squeeze(-1)  # (B, C)
        batch_size, num_channels = gap.size()

        # 3. 通过全连接层和softmax得到两个互补的特征描述子，生成两个分支的通道注意力权重
        attention_weights = self.attention_fc(gap).view(batch_size, num_channels, 2) # (B, C, 2)
        attention_weights = self.softmax(attention_weights) # (B, C, 2)

        # --- 特征加权与融合 ---
        # 提取并重塑每个分支的权重（将特征描述子拆分为两个部分）
        w1 = attention_weights[:, :, 0].unsqueeze(-1)  # (B, C, 1)
        w2 = attention_weights[:, :, 1].unsqueeze(-1)  # (B, C, 1)

        # 4. 逐元素相乘得到混合特征
        mixed_feature_1 = out1 * w1
        mixed_feature_2 = out2 * w2

        # 5. 逐元素相加融合混合特征
        fused_out = mixed_feature_1 + mixed_feature_2

        # 最终输出：融合后的特征 + 残差连接，再通过ReLU
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(fused_out + res)


class TemporalConvNet(nn.Module):
    """
    完整的时间卷积网络（TCN）由多个AttentionFusionBlock堆叠而成。
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        初始化TCN
        参数:
            num_inputs:     输入通道数 int
            num_channels:   每层的输出通道数列表（隐藏通道数） list
            kernel_size:    卷积核尺寸大小 int
            dropout:        Dropout率 float
        """
        super(TemporalConvNet, self).__init__()
        layers = []     # TCN所有层数，存储所有时间卷积块
        num_levels = len(num_channels)  # 网络层数
        for i in range(num_levels):     # 逐层构建网络
            dilation_size = 2 ** i      # 每层的扩张率（膨胀率），按 2 的幂次递增（1, 2, 4, 8...）
            in_channels = num_inputs if i == 0 else num_channels[i-1]   # 确定当前层的输入通道数：第一层：使用num_inputs，后续层：使用前一层的输出通道数
            out_channels = num_channels[i]      # 当前层的输出通道数
            # 为每一层创建时间卷积块AttentionFusionBlock并添加到layers层列表
            layers += [AttentionFusionBlock(in_channels, out_channels, kernel_size,
                                             stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # 将所有层组合成一个顺序执行的网络

    def forward(self, x):
        """
        前向传播
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, in_channels, seq_len]

        返回:
            Tensor: 输出张量，形状为 [batch_size, out_channels[-1], seq_len]
        """
        return self.network(x)
