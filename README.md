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

配套的 `original16_data.zip` 独立于代码包，包含本模型需要的 **306 份训练 CSV、102 份验证 CSV，以及对应的 408 份原始来源 NPZ**。代码仓库不自动下载数据。将数据包解压到本目录，得到：

```text
data/
  corrected/
    Phase1And2_Complete/Complete/<subject>/<trial>/exodata.csv
    Phase3_Complete/Complete/<subject>/<trial>/exodata.csv
  original_sources/
    <phase>/<subject>/<trial>/sources.npz
```

数据包的字节数及 SHA256 记录在 `config/data_bundle.json`，可用于核对传输完整性。

如果两个 ZIP 位于同一上级目录，可在解压后的代码目录执行：

```bash
python -c "import zipfile; zipfile.ZipFile('../original16_data.zip').extractall('.')"
```

也可保留数据在其他位置，给训练和验证命令提供 `--data-root /path/to/corrected --cache-root /path/to/original_sources`。所有相对参数路径均相对代码目录解析，与启动命令时的工作目录无关。路径含空格时加引号。

数据包内 CSV 和 NPZ 保持原始字节内容；加载时逐文件核对 SHA256、行数、相对时间和有效标签对齐。原始来源 NPZ 必须同时提供，不能只用已经填补缺失值的 CSV 替代。

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

`--epochs`、`--batch-size` 可显式覆盖训练预算。`--max-train-windows` 和 `--max-val-windows` 仅用于限制窗口数量；这类运行会标记 `limited_run=true`，不能代替正式结果。窗口限制不避免读取对应划分的全部源文件。种子固定不保证不同硬件或不同库版本逐位一致。

## 数据与损失定义

CSV 列顺序固定为：

```text
time,hip_angle_l,hip_angle_r,hip_torque_l,hip_torque_r,knee_angle_l,knee_angle_r,knee_torque_l,knee_torque_r,insole_l_force_y,insole_r_force_y
```

输入顺序为四个角度、四个一阶动态特征、四个二阶动态特征、左右压力、左右压力可用性。每组关节顺序均为左髋、右髋、左膝、右膝；输出沿用数据约定，单位为 N·m/kg。角度数值直接沿用配套 CSV，不额外变换单位或方向。

## 验证

训练结束后自动加载验证 MSE 最低的权重并计算物理单位指标。需要重新验证时使用独立输出目录：

```bash
python val.py --run-dir result/online_seed3407 --output-dir result/online_seed3407_validation --device cpu
```

验证入口校验冻结源码、训练配置、记录、scaler 及原始来源清单的哈希，使用保存的源码和最佳权重。迁移数据后可传入前述两个数据目录覆盖参数，无需改动训练产物。

每个运行保存：

- `config.json`、`records.json`、`split_audit.json`、原始来源清单与数据哈希。
- `scaler.npz`、源码快照及其哈希、执行环境、种子及参数数量。
- `history.json`、`train.log`、`best_model.pt`、`last_model.pt`、`completed.json`。
- `val_evaluation/metrics.json` 和 `predictions.npz`，包括物理单位预测、目标、有效标签掩码、文件索引和窗口端点。

`last_model.pt` 保存最后一轮的优化器和调度器状态，但本版本没有恢复训练入口。RMSE、MAE、R² 按各关节有效标签计算，另提供逐受试者结果及受试者宏平均 R²。压力来源分组是文件级分组，不能解释为逐帧缺压分析或压力的因果收益。指标沿用原实现；R² 的方差分母下限为 `1e-15`。

## 独立推理与部署

部署时需要 `model/` 和**同一次训练**生成的 `best_model.pt`、`scaler.npz`。本训练入口将配置保存在权重文件的 `config` 中，模型参数保存在 `model_state_dict` 中；不需要再复制训练数据、划分清单或验证源码快照。如果只有不含配置的裸 `state_dict`，还必须保存原训练配置。

当前仓库不包含正式训练权重。旧 14 维模型的权重不能用于本模型，也不能用 `strict=False` 跳过不匹配参数来替代训练。

以下示例接收已经按该训练配置预处理、标准化的 `windows.npy`，形状为 `[N,180,16]`。将 `model/`、`best_model.pt`、`scaler.npz` 和 `windows.npy` 放在同一目录运行：

```python
import numpy as np
import torch
from model import create_model

checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=True)
config = checkpoint['config']
model = create_model(config)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

windows = np.load('windows.npy', allow_pickle=False).astype(np.float32)
assert windows.ndim == 3 and windows.shape[1:] == (config['window'], 16)
assert len(windows) > 0 and np.isfinite(windows).all()
assert np.isin(windows[..., 14:16], [0, 1]).all()
with np.load('scaler.npz', allow_pickle=False) as scaler:
    output_mean = scaler['output_mean'].copy()
    output_std = scaler['output_std'].copy()

with torch.inference_mode():
    normalized_torque = model(torch.from_numpy(windows)).numpy()
torque = normalized_torque * output_std + output_mean
assert np.isfinite(torque).all()
np.save('prediction_nm_per_kg.npy', torque)
```

从传感器数据构造输入时，还需要 `utils/preprocessing.py`：

1. 四角度和两路压力按训练时相同的单位、方向、顺序，以 200 Hz 对齐。
2. 用实际传感器状态确定每帧左右压力是否可用；缺失压力置零、标志置 0，真实零压力的标志仍为 1；有效压力裁剪到 `[0,1500]`。
3. 调用 `build_features_single_csv(angles, pressure, **config['features'])` 生成前 14 维。使用同次训练 scaler 的 `input_mean`、`input_std` 标准化，再追加未经标准化的两路可用性标志。
4. 取最近 180 帧作为一个输入窗口。因果滤波应连续保持状态，或对已获得的历史重新计算；不要每次只对最近 180 帧重新置零滤波，否则与训练特征不一致。原始角度通道不替换成滤波角度。

在线配置使用四阶 10 Hz 因果 Butterworth 及后向差分，预测窗口末端的四关节力矩。网络包含窗口内时间池化及对称 SSM 卷积，不能把它当作可直接复用隐状态的逐帧递推模型。窗口推理本身不需要力矩标签或原始有效性 NPZ。

以上是 PyTorch 推理路径。Jetson/TensorRT 部署还需要独立的导出、转换、精度对照及目标设备延迟验证；本仓库没有为当前 16 维权重提供 ONNX/TensorRT 工件或实时性能结论。模型输出为 N·m/kg，不能直接作为电机的 N·m 指令。
