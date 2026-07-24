# Ridge 时序预测算子迁移设计方案

> 目标：将 `/Users/panhui/Desktop/ph_hw/agent/HBHD_ARX_v2.0` 中的岭回归预测模型迁移为 `src/tsas/engine/operator/forecasting` 下的通用算子。
> 设计原则：严格保持 HBHD 原逻辑（单目标、单 horizon、单步增量），同时适配 `BaseForecaster` 接口。

---

## 1. 核心定位

- **算子名**：`ridge_forecaster`
- **输出语义**：预测未来 `T + horizon` 时刻的**绝对物理值**
- **内部学习目标**：`y[anchor + horizon] - y[anchor]`（相对当前锚点的单步增量）
- **模型数量**：一个算子实例只服务一个 `horizon`；要预测 T+5/T+10/…/T+30，需创建多个实例
- **目标维度**：严格单目标，`fit(x, y)` 中 `y` 必须是 `(timesteps, 1)`

---

## 2. 新增文件

```text
src/tsas/engine/operator/forecasting/ridge.py
```

`src/tsas/engine/operator/forecasting/__init__.py` 无需修改，CLI registry 会自动扫描该包并注册算子。

---

## 3. 配置类 `RidgeForecasterConfig`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `seq_len` | `int` | `160` | 历史窗口长度 |
| `horizon` | `int` | `30` | 预测未来第几步（等价于 HBHD 的 `HORIZON`） |
| `alphas` | `list[float]` | `[0.1, 1.0, 10.0, 100.0]` | Ridge 候选正则强度，验证集选最优 |
| `fit_intercept` | `bool` | `True` | 是否拟合截距 |
| `solver` | `str` | `"lsqr"` | Ridge solver，与 HBHD 一致 |
| `standardize` | `bool` | `True` | 是否做标准化（HBHD 两阶段缩放） |
| `train_ratio` | `float` | `0.7` | 训练集占比 |
| `val_ratio` | `float` | `0.15` | 验证集占剩余比例 |
| `train_sample_step` | `int` | `max(1, seq_len // 10)` | 训练集 anchor 降采样步长，与 HBHD 一致 |
| `val_sample_step` | `int` | `4` | 验证集 anchor 降采样步长，与 HBHD 一致 |
| `seed` | `int` | `42` | 随机种子 |
| `target_idx` | `int` | `-1` | 目标列在 `x` 中的列索引，`-1` 表示最后一列 |

---

## 4. 类结构

```text
RidgeForecasterConfig(BaseModel)

RidgeForecaster(BaseForecaster[ForecastExtraOutput,
                               RidgeForecasterConfig,
                               None,
                               None])
    ├── name() -> "ridge_forecaster"
    ├── version() -> (1, 0, 0)
    ├── set_chunk_ids(chunk_ids)        # 支持不跨断层构造样本
    ├── _fit_data(x, y, params)
    ├── _run_data(x, params)
    ├── _save_fit_state(path)
    └── _load_fit_state(path)
```

---

## 5. 训练流程 `_fit_data`

输入约定：

- `x`: `(timesteps, num_features)`，DataFrame 或 ndarray
- `y`: `(timesteps, 1)`，DataFrame 或 ndarray

### 5.1 校验

- `y.shape[1] == 1`，否则抛出 `ValueError`
- `len(x) == len(y)` 且 `len(x) >= seq_len + horizon`
- 校验 `0 <= target_idx < x.shape[1]`
- 校验 `y[:, 0]` 与 `x[:, target_idx]` 是否一致，不一致时抛出 `ValueError`

### 5.2 确定目标历史序列与外生变量

目标历史从 `x` 的目标列提取，其余列作为外生变量：

```python
target = x[:, target_idx]
exog   = np.delete(x, target_idx, axis=1)
```

`y[:, 0]` 仅作为待预测的真实目标，训练前必须校验 `y[:, 0]` 与 `x[:, target_idx]` 完全一致。

### 5.3 构造样本

1. 提取所有合法 anchor 索引 `i ∈ [seq_len-1, len-horizon-1]`；
2. 若已调用 `set_chunk_ids`，anchor 不会跨越时间断层；
3. 按 5.4 划分 train / val anchor；
4. 对 train anchor 按 `train_sample_step` 降采样，对 val anchor 按 `val_sample_step` 降采样：

```python
selected_train = train_anchors[::train_sample_step]
selected_val   = val_anchors[::val_sample_step]
```

5. 计算 `delta_y`：

```python
delta_y = np.diff(target, prepend=target[0])
```

6. 拟合输入标准化器（与 HBHD 一致）：

```python
max_train_row = int(selected_train[-1] + horizon)
input_mean = exog[: max_train_row + 1].mean(axis=0)
input_std  = exog[: max_train_row + 1].std(axis=0) + 1e-12
```

7. 对每个选中的 anchor 构造样本：

```python
window_x      = exog[i - seq_len + 1 : i + 1]          # (seq_len, num_exog)
window_dy     = delta_y[i - seq_len + 1 : i + 1]       # (seq_len,)
inputs_scaled = (window_x - input_mean) / input_std    # 输入标准化
features      = np.column_stack([window_dy, inputs_scaled]).ravel()
base          = target[i]
label         = target[i + horizon] - base
```

得到：

- `X`: `(n_samples, seq_len * (num_exog + 1))`
- `bases`: `(n_samples,)`
- `y`: `(n_samples,)`

### 5.4 时序划分

按时间顺序切分 train / val，不打乱：

```python
n_train = int(n_total * train_ratio)
n_val   = int((n_total - n_train) * val_ratio)
```

### 5.5 特征标准化

在训练集上 fit `feature_scaler`，然后 transform train / val：

```python
feature_mean = X_train.mean(axis=0)
feature_std  = X_train.std(axis=0) + 1e-8
X_train_scaled = (X_train - feature_mean) / feature_std
X_val_scaled   = (X_val - feature_mean) / feature_std
```

### 5.6 alpha 网格选优

对每个候选 `alpha`：

```python
model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
model.fit(X_train_scaled, y_train)
```

在验证集上评估绝对值预测效果：

```python
pred_abs = val_bases + model.predict(X_val_scaled)
true_abs = val_bases + y_val
rmse = sqrt(mean((pred_abs - true_abs) ** 2))
```

选择 RMSE 最小的 `alpha` 作为 `best_alpha`。

### 5.7 最终模型

使用 `best_alpha` 在 **只在 train 上训练（与 HBHD 原代码一致）** 上重新训练最终 Ridge 模型。


---

## 6. 推理流程 `_run_data`

输入约定：

- `x`: `(seq_len, num_features)` 或 `(batch, seq_len, num_features)`

### 6.1 预处理

1. 从 `x[:, target_idx]` 提取目标历史，其余列作为外生变量；
2. 计算 `delta_y`；
3. 用保存的 `input_scaler` 标准化外生变量；
4. 构造展平特征向量，用保存的 `feature_scaler` 标准化。

### 6.2 预测

```python
pred_increment = np.dot(X_scaled, coef) + intercept
base           = target[-1]                       # 窗口最后一个目标值
pred_abs       = base + pred_increment
```

### 6.3 输出

- 单样本：`(1, 1)`
- 批量：`(batch, 1, 1)`

输出值表示 `T + horizon` 时刻的绝对预测值。

---

## 7. 保存与加载

### 7.1 保存内容 `_save_fit_state`

保存为 `npz` 文件，包含：

| 键 | 内容 |
|---|---|
| `coef` | Ridge 系数 `(n_features,)` |
| `intercept` | 截距标量 |
| `input_mean` | 输入标准化均值 |
| `input_scale` | 输入标准化标准差 |
| `feature_mean` | 特征标准化均值 |
| `feature_scale` | 特征标准化标准差 |
| `seq_len` | 历史窗口长度 |
| `horizon` | 预测 horizon |
| `num_features` | 原始 `x` 列数 |
| `num_exog` | 外生变量列数 |
| `target_idx` | 目标列索引 |
| `best_alpha` | 选中的正则强度 |

### 7.2 加载逻辑 `_load_fit_state`

1. 读取 `npz` 恢复 scaler 参数和元信息
2. 新建 `Ridge()` 实例，手动赋值 `coef_` 和 `intercept_`
3. 设置 `_fitted = True`

不依赖 pickle，保证跨环境可加载。

---

## 8. 与 HBHD 原代码的对应关系

| HBHD 原代码 | 新算子 |
|---|---|
| `SEQ_LEN` | `seq_len` |
| `HORIZON` | `horizon` |
| `TARGET_COL` + `INPUT_COLS` | `target_idx` 指定目标列，其余列作为外生变量 |
| `RIDGE_ALPHAS` | `alphas` 配置 |
| `TRAIN_SAMPLE_STEP` / `VAL_SAMPLE_STEP` | `train_sample_step` / `val_sample_step` 配置 |
| `double_ema_by_episode` | **不内置**，由数据预处理管线提供平滑后的 `y` |
| `build_arx_matrix` | 内部 `_build_samples` |
| `fit_input_scaler` + `standardize_feature_matrix` | 保留两阶段标准化 |
| `train_and_select_model` | alpha 网格验证选优 |
| `pred_residual + y_base_anchor` | 内部自动完成 |
| `apply_post_processing_ema` | **不内置**，作为独立后处理或业务层实现 |
| episode 切分 | 通过 `set_chunk_ids` 支持不跨断层采样 |
| 每个 horizon 一个 `.npz` | 一个算子实例对应一个 horizon |

---

## 9. 待确认事项

请在实现前确认以下问题：

1. **`target_idx` 默认值**
   - [x] 默认 `-1`（最后一列）
   - [x] `y[:, 0]` 必须等于 `x[:, target_idx]`，不一致时报错

2. **最终模型训练数据**
   - [x] 只用 train 训练（与 HBHD 原代码一致）

3. **标准化开关**
   - [x] 强制开启标准化（与 HBHD 一致）

4. **输出形式**
   - [x] 输出绝对物理值（`base + increment`）

---

## 10. 后续工作

- [ ] 实现 `ridge.py`
- [ ] 补充单元测试（单样本/批量推理、保存加载、chunk_ids、不同 `target_idx` 模式）
- [ ] 补充 CLI 使用示例
- [ ] 可选：编写从 HBHD `.npz` 模型权重迁移到新算子保存格式的转换脚本
