# -*- coding: utf-8 -*-

"""
Ridge 工业时序预测算子

- 输入为历史窗口展平特征（目标一阶差分 + 标准化外生变量）
- 输出为未来 ``T + horizon`` 时刻的绝对物理值
- 训练时内部学习相对当前锚点的单步增量

算子负责：
1. 训练阶段：构造样本、两阶段标准化、alpha 网格验证选优
2. 推理阶段：构造特征、标准化、预测增量、加基准值得到物理量预测
"""

from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import Ridge

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput

__all__ = [
    'RidgeForecasterConfig',
    'RidgeForecaster',
]


class RidgeForecasterConfig(BaseModel):
    """Ridge 预测算子实例参数。

    所有带 ``ge`` / ``le`` 约束的数值字段均可被 HPO 自动搜索。
    """

    # ---- 窗口 / 问题结构 ----
    seq_len: int = Field(default=160, ge=100, le=300, description="输入历史窗口长度")
    horizon: int = Field(default=30, ge=30, le=50, description="预测未来第几步")
    target_idx: int = Field(default=-1, ge=-1, description="目标列在 x 中的列索引，-1 表示最后一列")

    # ---- Ridge 超参数 ----
    alphas: list[float] = Field(
        default=[0.1, 1.0, 10.0, 100.0],
        description="Ridge 候选正则强度，验证集选最优",
    )
    fit_intercept: bool = Field(default=True, description="是否拟合截距")
    solver: str = Field(default='lsqr', description="Ridge solver")

    # ---- 训练 / 调参 ----
    standardize: bool = Field(default=True, description="是否做两阶段标准化")
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.95, description="训练集占比")
    val_ratio: float = Field(default=0.15, ge=0.0, le=0.45, description="验证集占剩余比例")
    train_sample_step: int | None = Field(
        default=None,
        ge=1,
        description="训练集 anchor 降采样步长，None 时自动取 max(1, seq_len // 10)",
    )
    val_sample_step: int = Field(
        default=4,
        ge=1,
        description="验证集 anchor 降采样步长",
    )
    seed: int = Field(default=42, ge=0, description="随机种子，用于保证训练可复现")

    @model_validator(mode='before')
    @classmethod
    def _set_train_sample_step(cls, values):
        """当 train_sample_step 为 None 时，根据 seq_len 自动计算默认值。"""
        if not isinstance(values, dict):
            return values
        if values.get('train_sample_step') is None:
            seq_len = values.get('seq_len', 160)
            values['train_sample_step'] = max(1, seq_len // 10)
        return values

    class Config:
        extra = 'forbid'


class RidgeForecaster(BaseForecaster[ForecastExtraOutput,
                                     RidgeForecasterConfig,
                                     None,
                                     None]):
    """Ridge 工业时序预测算子。

    每个算子实例服务于一个 ``horizon``：学习 ``T + horizon`` 时刻相对当前锚点的
    单步增量，推理时输出 ``T + horizon`` 时刻的绝对物理值。

    输入输出约定遵循 ``BaseForecaster``::

        fit(x, y):
            x: (timesteps, num_features)  DataFrame / ndarray
               其中第 ``target_idx`` 列为目标变量历史
            y: (timesteps, 1)             DataFrame / ndarray
               必须与 ``x[:, target_idx]`` 一致

        run(x):
            x: (seq_len, num_features) 或 (batch, seq_len, num_features)
               其中第 ``target_idx`` 列为目标变量历史
            返回: (1, 1) 或 (batch, 1, 1)，表示 T + horizon 时刻的绝对预测值
    """

    _MODEL_FILE = '_model.npz'
    _STATE_FILE = '_forecaster_state.npz'

    @classmethod
    def name(cls) -> str:
        return "ridge_forecaster"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号。

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: RidgeForecasterConfig | None = None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model: Ridge | None = None
        self._input_mean: np.ndarray | None = None
        self._input_scale: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._num_features: int | None = None
        self._num_exog: int | None = None
        self._best_alpha: float | None = None
        self._chunk_ids: np.ndarray | None = None

    def set_chunk_ids(self, chunk_ids: np.ndarray | None) -> None:
        """设置时间连续段标识，用于构造不跨越时间断层的样本窗口。

        ``chunk_ids`` 应与 ``fit(x, y)`` 中的 ``x`` / ``y`` 行对齐。
        若传入 ``None``，则回退到原有行为（可能跨越时间断层）。
        """
        if chunk_ids is not None:
            chunk_ids = np.asarray(chunk_ids)
            if chunk_ids.ndim != 1:
                raise ValueError(f"chunk_ids 必须是 1-D 数组，当前维度 {chunk_ids.ndim}")
        self._chunk_ids = chunk_ids

    def _get_valid_anchor_indices(self, n_total: int) -> np.ndarray:
        """返回可完整构成 (seq_len, horizon) 窗口的 anchor 索引。

        若 ``self._chunk_ids`` 已设置，则 anchor 不会跨越时间断层；
        否则按整个序列连续采样。
        """
        cfg = self.config
        if self._chunk_ids is not None:
            if len(self._chunk_ids) != n_total:
                raise ValueError(
                    f"chunk_ids 长度 {len(self._chunk_ids)} 与 x 长度 {n_total} 不一致"
                )
            valid_indices = []
            for cid in np.unique(self._chunk_ids):
                indices = np.flatnonzero(self._chunk_ids == cid)
                if len(indices) <= cfg.seq_len + cfg.horizon:
                    continue
                first_anchor = indices[0] + cfg.seq_len - 1
                last_anchor = indices[-1] - cfg.horizon
                valid_indices.extend(range(first_anchor, last_anchor + 1))
            if not valid_indices:
                raise ValueError(
                    f"没有可用样本：所有 chunk 的长度都小于 "
                    f"seq_len + horizon = {cfg.seq_len + cfg.horizon}"
                )
            return np.asarray(valid_indices, dtype=int)

        n_samples = n_total - cfg.seq_len - cfg.horizon + 1
        if n_samples <= 0:
            raise ValueError(
                f"时间序列长度 {n_total} 不足以构造窗口 "
                f"(seq_len={cfg.seq_len}, horizon={cfg.horizon})"
            )
        return np.arange(cfg.seq_len - 1, n_total - cfg.horizon, dtype=int)

    def _split_train_val(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按时间顺序切分 train / val 索引（与 HBHD 原脚本一致）。"""
        cfg = self.config
        n_total = len(indices)
        train_end = int(n_total * cfg.train_ratio)
        val_end = int(n_total * (cfg.train_ratio + cfg.val_ratio))
        if train_end <= 0 or train_end >= n_total:
            raise ValueError(
                f"训练集划分结果无效: n_total={n_total}, train_end={train_end}, "
                f"train_ratio={cfg.train_ratio}"
            )
        if val_end <= train_end or val_end > n_total:
            raise ValueError(
                f"验证集划分结果无效: n_total={n_total}, train_end={train_end}, "
                f"val_end={val_end}, val_ratio={cfg.val_ratio}"
            )
        return indices[:train_end], indices[train_end:val_end]

    def _downsample_anchors(self, train_anchors: np.ndarray, val_anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对 train / val anchor 按配置步长降采样。"""
        cfg = self.config
        train_step = cfg.train_sample_step if cfg.train_sample_step is not None else max(1, cfg.seq_len // 10)
        val_step = cfg.val_sample_step
        return train_anchors[::train_step], val_anchors[::val_step]

    def _build_samples(
        self,
        target: np.ndarray,
        exog: np.ndarray,
        anchors: np.ndarray,
        input_mean: np.ndarray | None,
        input_scale: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造 Ridge 训练/验证样本。

        Args:
            target: 目标序列，形状 ``(timesteps,)``
            exog: 外生变量，形状 ``(timesteps, num_exog)``
            anchors: anchor 索引数组
            input_mean: 输入标准化均值，None 表示不做标准化
            input_scale: 输入标准化标准差，None 表示不做标准化

        Returns:
            tuple: ``(X, y, bases)``
                - X: ``(n_samples, seq_len * (num_exog + 1))``
                - y: ``(n_samples,)`` 相对锚点的增量
                - bases: ``(n_samples,)`` 锚点处的目标值
        """
        cfg = self.config
        n_features = cfg.seq_len * (exog.shape[1] + 1)
        n_samples = len(anchors)

        X = np.empty((n_samples, n_features), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.float32)
        bases = np.empty(n_samples, dtype=np.float32)

        delta_y = np.diff(target, prepend=target[0])

        for row, anchor in enumerate(anchors):
            start = anchor - cfg.seq_len + 1
            end = anchor + 1
            window_exog = exog[start:end]
            window_dy = delta_y[start:end]

            if input_mean is not None and input_scale is not None:
                window_exog_scaled = (window_exog - input_mean) / input_scale
            else:
                window_exog_scaled = window_exog

            features = np.column_stack([window_dy, window_exog_scaled]).ravel()
            X[row] = features.astype(np.float32, copy=False)
            bases[row] = target[anchor]
            y[row] = target[anchor + cfg.horizon] - bases[row]

        return X, y, bases

    def _standardize_features(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """在训练集上 fit 特征标准化器并 transform 训练/验证集。"""
        feature_mean = X_train.mean(axis=0, dtype=np.float64)
        feature_scale = X_train.std(axis=0, dtype=np.float64) + 1e-8
        X_train_scaled = ((X_train - feature_mean) / feature_scale).astype(np.float32)
        X_val_scaled = ((X_val - feature_mean) / feature_scale).astype(np.float32)
        return X_train_scaled, X_val_scaled, feature_mean, feature_scale

    def _train_and_select_alpha(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_bases: np.ndarray,
    ) -> tuple[float, Ridge]:
        """遍历候选 alpha，在验证集上选最优 Ridge 模型。

        返回最优 alpha 及其对应的、已在训练集上拟合好的 Ridge 模型。
        """
        cfg = self.config
        best_rmse = float('inf')
        best_alpha = cfg.alphas[0]
        best_model: Ridge | None = None

        for alpha in cfg.alphas:
            model = Ridge(alpha=alpha, fit_intercept=cfg.fit_intercept, solver=cfg.solver)
            model.fit(X_train, y_train)

            pred_abs = val_bases + model.predict(X_val)
            true_abs = val_bases + y_val
            rmse = float(np.sqrt(np.mean((pred_abs - true_abs) ** 2)))

            logger.debug(
                f"[{self.oid}] Ridge candidate alpha={alpha:g}: val_rmse={rmse:.6g}"
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_model = model

        if best_model is None:
            raise RuntimeError("Ridge alpha 选优失败，未产生有效模型")

        logger.info(
            f"[{self.oid}] 最优 Ridge alpha={best_alpha:g} (val_rmse={best_rmse:.6g})"
        )
        return best_alpha, best_model

    def _fit_data(self, x: np.ndarray, y: np.ndarray, *, params: None) -> None:
        cfg = self.config
        np.random.seed(cfg.seed)

        # ---- 输入校验 ----
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != 1:
            raise ValueError(
                f"RidgeForecaster 仅支持单目标预测，y 列数应为 1，但当前为 {y.shape[1]}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x 与 y 的时间步数不一致: {x.shape[0]} != {y.shape[0]}"
            )
        if cfg.target_idx >= x.shape[1]:
            raise ValueError(
                f"target_idx {cfg.target_idx} 超出 x 的列数 {x.shape[1]}"
            )
        if not np.allclose(y[:, 0], x[:, cfg.target_idx]):
            raise ValueError(
                f"y[:, 0] 与 x[:, target_idx={cfg.target_idx}] 不一致，"
                "请确保传入的 y 与 x 中的目标列对应同一变量"
            )

        n_total = x.shape[0]
        if n_total < cfg.seq_len + cfg.horizon:
            raise ValueError(
                f"时间序列长度 {n_total} 不足以构造窗口 "
                f"(seq_len={cfg.seq_len}, horizon={cfg.horizon})"
            )

        # ---- 目标与外生变量 ----
        target = x[:, cfg.target_idx].astype(np.float64, copy=False)
        exog = np.delete(x, cfg.target_idx, axis=1).astype(np.float64, copy=False)
        self._num_features = x.shape[1]
        self._num_exog = exog.shape[1]

        # ---- 构造 train / val anchor ----
        all_anchors = self._get_valid_anchor_indices(n_total)
        train_anchors, val_anchors = self._split_train_val(all_anchors)

        if len(train_anchors) == 0:
            raise ValueError("训练集 anchor 为空，无法训练")
        if len(val_anchors) == 0:
            raise ValueError("验证集 anchor 为空，无法选优")

        # ---- 输入标准化：在降采样前的 train anchor 覆盖范围内拟合（与 HBHD 一致） ----
        if cfg.standardize:
            max_train_row = int(train_anchors[-1] + cfg.horizon)
            self._input_mean = exog[:max_train_row + 1].mean(axis=0, dtype=np.float64)
            self._input_scale = exog[:max_train_row + 1].std(axis=0, dtype=np.float64) + 1e-12
        else:
            self._input_mean = None
            self._input_scale = None

        # ---- 降采样 ----
        train_anchors, val_anchors = self._downsample_anchors(train_anchors, val_anchors)

        # ---- 构造样本 ----
        X_train, y_train, train_bases = self._build_samples(
            target, exog, train_anchors, self._input_mean, self._input_scale
        )
        X_val, y_val, val_bases = self._build_samples(
            target, exog, val_anchors, self._input_mean, self._input_scale
        )

        # ---- 特征标准化 ----
        if cfg.standardize:
            X_train_scaled, X_val_scaled, self._feature_mean, self._feature_scale = \
                self._standardize_features(X_train, X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            self._feature_mean = None
            self._feature_scale = None

        # ---- alpha 网格选优 ----
        self._best_alpha, _ = self._train_and_select_alpha(
            X_train_scaled, y_train, X_val_scaled, y_val, val_bases
        )

        # ---- 最终模型：只用 train 训练 ----
        final_model = Ridge(
            alpha=self._best_alpha,
            fit_intercept=cfg.fit_intercept,
            solver=cfg.solver,
        )
        final_model.fit(X_train_scaled, y_train)
        self._model = final_model

    def _run_data(self, x: np.ndarray, *, params: None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型尚未训练，无法执行推理")

        cfg = self.config
        if self._num_features is None or self._num_exog is None:
            raise RuntimeError("算子内部状态缺失，请先训练或加载模型")

        batched = x.ndim == 3
        if not batched:
            if x.shape != (cfg.seq_len, self._num_features):
                raise ValueError(
                    f"推理输入形状应为 (seq_len={cfg.seq_len}, num_features={self._num_features}), "
                    f"但当前为 {x.shape}"
                )
            x = x[np.newaxis, ...]  # (1, seq_len, num_features)
        else:
            if x.shape[1:] != (cfg.seq_len, self._num_features):
                raise ValueError(
                    f"批量推理输入形状应为 (batch, seq_len={cfg.seq_len}, num_features={self._num_features}), "
                    f"但当前为 {x.shape}"
                )

        batch_size = x.shape[0]
        x = x.astype(np.float64, copy=False)

        # 提取目标与外生变量
        target = x[:, :, cfg.target_idx]  # (batch, seq_len)
        exog = np.delete(x, cfg.target_idx, axis=2)  # (batch, seq_len, num_exog)

        delta_y = np.diff(target, prepend=target[:, :1], axis=1)  # (batch, seq_len)

        # 输入标准化
        if self._input_mean is not None and self._input_scale is not None:
            exog_scaled = (exog - self._input_mean) / self._input_scale
        else:
            exog_scaled = exog

        # 构造展平特征
        features = np.concatenate([delta_y[..., np.newaxis], exog_scaled], axis=2)  # (batch, seq_len, num_exog+1)
        features = features.reshape(batch_size, -1)  # (batch, seq_len * (num_exog+1))

        # 特征标准化
        if self._feature_mean is not None and self._feature_scale is not None:
            features_scaled = (features - self._feature_mean) / self._feature_scale
        else:
            features_scaled = features

        # 预测增量并转换为绝对值
        pred_increment = self._model.predict(features_scaled)  # (batch,)
        base = target[:, -1]  # (batch,)
        pred_abs = base + pred_increment

        pred = pred_abs.reshape(batch_size, 1, 1)  # (batch, 1, 1)
        return pred if batched else pred[0]

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        if self._model is None:
            return

        np.savez(
            path / self._MODEL_FILE,
            coef=self._model.coef_.astype(np.float32),
            intercept=np.asarray([self._model.intercept_], dtype=np.float32),
        )

        np.savez(
            path / self._STATE_FILE,
            input_mean=self._input_mean if self._input_mean is not None else np.array([]),
            input_scale=self._input_scale if self._input_scale is not None else np.array([]),
            feature_mean=self._feature_mean if self._feature_mean is not None else np.array([]),
            feature_scale=self._feature_scale if self._feature_scale is not None else np.array([]),
            seq_len=np.asarray([self.config.seq_len]),
            horizon=np.asarray([self.config.horizon]),
            target_idx=np.asarray([self.config.target_idx]),
            num_features=np.asarray([self._num_features]),
            num_exog=np.asarray([self._num_exog]),
            best_alpha=np.asarray([self._best_alpha]),
            standardize=np.asarray([self.config.standardize]),
        )

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)

        state = np.load(path / self._STATE_FILE)
        cfg = self.config
        cfg.seq_len = int(state['seq_len'][0])
        cfg.horizon = int(state['horizon'][0])
        cfg.target_idx = int(state['target_idx'][0])
        cfg.standardize = bool(state['standardize'][0])
        self._num_features = int(state['num_features'][0])
        self._num_exog = int(state['num_exog'][0])
        self._best_alpha = float(state['best_alpha'][0])

        if cfg.standardize:
            self._input_mean = state['input_mean']
            self._input_scale = state['input_scale']
            self._feature_mean = state['feature_mean']
            self._feature_scale = state['feature_scale']
        else:
            self._input_mean = None
            self._input_scale = None
            self._feature_mean = None
            self._feature_scale = None

        model_data = np.load(path / self._MODEL_FILE)
        self._model = Ridge(
            alpha=self._best_alpha,
            fit_intercept=cfg.fit_intercept,
            solver=cfg.solver,
        )
        self._model.coef_ = model_data['coef'].astype(np.float64)
        self._model.intercept_ = float(model_data['intercept'][0])

        self._fitted = True
