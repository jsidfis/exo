# 16 Parallel TCN–SSM

用于四关节力矩回归的并行 TCN–SSM 模型。输入为 `[batch, 180, 16]`，输出为 `[batch, 4]`，默认配置共 172,949 个可训练参数。

## 文件结构

```text
model/                 TCN、SSM、门控融合及模型入口
train.py               训练、Adam 优化、调度、早停和最佳权重选择
loss.py                原始有效标签上的 MSE
val.py                 加载冻结源码、最佳权重和训练 scaler 进行验证
config/online.yaml     在线末端窗口预测配置
config/splits/         固定的训练与验证划分
config/metadata.csv    对应记录的压力来源信息
utils/                 数据加载、预处理、归一化、指标、种子和检查点工具
result/                训练时自动创建的输出目录
requirements.txt       参考环境依赖
```

## 环境与数据

参考环境为 Python 3.12.4、PyTorch 2.3.0、NumPy 1.26.4、SciPy 1.13.1 和 PyYAML 6.0.1。建议在独立环境中安装：

```bash
python -m pip install -r requirements.txt
```

GPU 训练需要可用的 CUDA 版 PyTorch；默认命令使用 CPU。依赖清单没有包含绘图、文稿、仿真或测试框架。

## 训练

在线配置：

```bash
python train.py --config config/online.yaml --run-dir result/online_seed3407 --seed 3407 --device cpu
```

默认训练设置：

| 项目 | 设置 |
|---|---|
| 输入窗口 / 步长 | 180 / 30 帧，端点从索引 179 开始 |
| 采样间隔 | 0.005 秒（200 Hz） |
| TCN 通道 / 核大小 / dropout | 64、64、64 / 7 / 0.4 |
| SSM 层数 / state / expand / dropout | 1 / 8 / 2 / 0.1 |
| 融合 / 输出头 | 64 通道门控 / 共享线性 64→4 |
| 优化器 / 学习率 / weight decay | Adam / 0.0001 / 0 |
| 批量 / 最大轮数 | 64 / 100 |
| 学习率调度 | ReduceLROnPlateau，patience=5，factor=0.5 |
| 早停 | 验证 MSE 连续 10 轮未改善 |
| 梯度裁剪 / 压力随机丢弃 / AMP | 不使用 |

## 数据与损失定义

CSV 列顺序固定为：

```text
time,hip_angle_l,hip_angle_r,hip_torque_l,hip_torque_r,knee_angle_l,knee_angle_r,knee_torque_l,knee_torque_r,insole_l_force_y,insole_r_force_y
```

输入顺序为四个角度、四个一阶动态特征、四个二阶动态特征、左右压力、左右压力可用性。每组关节顺序均为左髋、右髋、左膝、右膝；输出沿用数据约定，单位为 N·m/kg。角度数值直接沿用配套 CSV，不额外变换单位或方向。



