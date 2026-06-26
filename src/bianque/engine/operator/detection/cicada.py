# -*- coding: utf-8 -*-

"""
CICADA 异常检测算子

基于 CICADA（Continual Learning via Incremental Component Adaptive Architecture）
重构误差的异常检测。CICADA 采用 Mixture-of-Experts + MAML 元学习 + 动态架构扩展，
支持多种异构专家编码器（GradPCA, GradKPCA, GradSFA, MLP 等）的自适应组合。

包含:
    - CICADAPredictor: CICADA 重构型预测器，训练后输出重构值

示例用法::

    # 训练 + 推理
    predictor = CICADAPredictor(experts=["MLP"], win_size=10, num_channels=3, epochs=5)
    predictor.fit(train_data)
    recon = predictor.run(test_data)
"""

import json
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bianque.engine.operator.base import UnsupervisedNumericOperatorMixin
from bianque.engine.operator.detection.base import BasePredictor

__all__ = [
    'CICADAPredictorConfig',
    'CICADAPredictor',
]


class CICADAPredictorConfig(BaseModel):
    """
    CICADA 预测器实例参数

    覆盖 CICADA 构造函数的全部参数。num_channels 为 None 时自动从训练数据推断。

    Attributes:
        experts: 专家模型名称列表（对应 bq_cicada CICADA 的 experts 参数）
        win_size: 滑动窗口长度
        stride: 训练滑动步长
        num_channels: 输入特征维度，None 时自动推断
        batch_size: 训练批大小
        epochs: 训练轮数
        latent_space_size: 隐空间维度
        n_components: 降维分量数，"auto" 为自动选择
        normalization: 归一化方式
        ar_order: 自回归阶数
        attn_bucket_heads: 桶注意力头数
        decoder_all_heads: 全局注意力头数
        forward_expansion: FFN 扩展因子
        train_init_meta_lr: 训练初始元学习率
        test_meta_lr: 测试元学习率
        meta_split_threshold: 专家分裂阈值
        lr_split_factor: 分裂学习率因子
        lambda_self: self_loss 权重（旧 ml_lambda）
        lambda_recon: recon_loss 权重
        lambda_mse: mse_loss 权重（仅 semi 使用）
        lambda_lr: lr_loss 权重（旧 penalty_rate）
        lr: 基础学习率
        ttlr: 测试时学习率
        gamma: 衰减因子
        adaptive_add: 是否动态扩展专家
        epoch_add: 扩展检查间隔
        close_epochs: 停止扩展提前量
        valid_size: 验证集比例，None 表示不划分
        shuffle: 是否打乱训练数据
        infer_mode: 推理模式
        th: 异常检测百分位阈值
    """

    model_config = ConfigDict(frozen=True)

    # -- 专家配置 --
    experts: list[str] = Field(
        default=["GradPCA", "GradKPCA", "GradFreKPCA", "GradSubPCA"],
        description="专家模型名称列表",
    )

    # -- 窗口 / 数据形状 --
    win_size: int = Field(default=5, gt=0, description="滑动窗口长度")
    stride: int = Field(default=1, gt=0, description="训练滑动步长")
    num_channels: int | None = Field(
        default=None,
        description="输入特征维度；None 时从训练数据自动推断",
    )

    # -- 训练 --
    batch_size: int = Field(default=256, gt=0, description="训练批大小")
    epochs: int = Field(default=60, gt=0, description="训练轮数")

    # -- 编码器架构 --
    latent_space_size: int = Field(default=128, gt=0, description="隐空间维度")
    n_components: str | int = Field(
        default="auto",
        description="降维分量数，'auto' 为自动选择",
    )
    normalization: str = Field(default="None", description="归一化方式")
    ar_order: int = Field(default=2, gt=0, description="自回归阶数")

    # -- 注意力 --
    attn_bucket_heads: int = Field(default=4, gt=0, description="桶注意力头数")
    decoder_all_heads: int = Field(default=8, gt=0, description="全局注意力头数")
    forward_expansion: int = Field(default=4, gt=0, description="FFN 扩展因子")

    # -- 元学习 --
    train_init_meta_lr: float = Field(default=1e-4, gt=0.0, description="训练初始元学习率")
    test_meta_lr: float = Field(default=1e-3, gt=0.0, description="测试元学习率")
    meta_split_threshold: float = Field(default=5e-4, gt=0.0, description="专家分裂阈值")
    lr_split_factor: float = Field(default=1.414, gt=1.0, description="分裂学习率因子")
    lambda_self: float = Field(default=10.0, gt=0.0, description="self_loss 权重")
    lambda_recon: float = Field(default=1.0, gt=0.0, description="recon_loss 权重")
    lambda_mse: float = Field(default=1.0, gt=0.0, description="mse_loss 权重（仅 semi）")
    lambda_lr: float = Field(default=0.1, ge=0.0, description="lr_loss 权重")

    # -- 优化器 --
    lr: float = Field(default=1e-3, gt=0.0, description="基础学习率")
    ttlr: float = Field(default=1e-3, gt=0.0, description="测试时学习率")
    gamma: float = Field(default=0.99, gt=0.0, lt=1.0, description="衰减因子")

    # -- 动态扩展 --
    adaptive_add: bool = Field(default=True, description="是否动态扩展专家")
    epoch_add: int = Field(default=10, gt=0, description="扩展检查间隔（轮数）")
    close_epochs: int = Field(default=20, ge=0, description="停止扩展提前量（轮数）")

    # -- 其他 --
    valid_size: float | None = Field(
        default=None, ge=0.0, lt=1.0,
        description="验证集比例；None 表示不划分",
    )
    shuffle: bool = Field(default=False, description="是否打乱训练数据")

    # -- 推理 --
    infer_mode: Literal["offline", "online"] = Field(default="offline", description="推理模式")
    th: float = Field(default=0.98, gt=0.0, le=1.0, description="异常检测百分位阈值")


class CICADAPredictor(UnsupervisedNumericOperatorMixin[None],
                      BasePredictor[None, CICADAPredictorConfig, None]):
    """
    CICADA 重构型预测器

    基于 CICADA 算法的重构型预测器。训练阶段通过 Mixture-of-Experts + MAML 元学习
    学习数据的重构表示，推理阶段输出与输入同维度的重构值。

    核心逻辑:
        - ``_fit_data``: 创建并训练 CICADA 模型，自动推断 num_channels
        - ``_run_data``: 调用 CICADA reconstruct 返回重构值

    注意:
        - bq_cicada 包为延迟导入，使用前需确保已安装 bq_cicada
        - num_channels 在 Config 中为 None 时自动从训练数据推断
        - 输入数据长度需 >= win_size，否则无法创建滑动窗口

    泛型参数:
        - EO: None（无附加输出）
        - C: CICADAPredictorConfig
        - RP: None（无运行参数）
        - FP: None（无训练参数）
    """

    _MODEL_FILE = "cicada_model.pt"
    _META_FILE = "cicada_meta.json"

    @classmethod
    def name(cls) -> str:
        return "cicada_predictor"

    def __init__(self, *, oid: str | None = None, config: CICADAPredictorConfig | None = None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model = None
        self._num_channels_detected: int | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate_ndarray_input(self, x: np.ndarray, params) -> None:
        if x.ndim != 2:
            raise ValueError(f"CICADAPredictor 要求 2D 输入，收到 {x.ndim}D")
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"CICADAPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行"
            )

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        try:
            from bq_cicada import CICADA  # noqa: lazy import
        except ImportError:
            raise RuntimeError("请先安装 bq-cicada 包") from None

        # 校验输入维度（fit 管线不做此检查，需在 _fit_data 内部校验）
        if x.ndim != 2:
            raise ValueError(f"CICADAPredictor 要求 2D 输入，收到 {x.ndim}D")
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"CICADAPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行"
            )

        # 推断 num_channels
        num_channels = (
            self.config.num_channels
            if self.config.num_channels is not None
            else x.shape[1]
        )
        self._num_channels_detected = num_channels

        self._model = CICADA(
            experts=list(self.config.experts),
            win_size=self.config.win_size,
            stride=self.config.stride,
            num_channels=num_channels,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            latent_space_size=self.config.latent_space_size,
            n_components=self.config.n_components,
            train_init_meta_lr=self.config.train_init_meta_lr,
            test_meta_lr=self.config.test_meta_lr,
            meta_split_threshold=self.config.meta_split_threshold,
            gamma=self.config.gamma,
            normalization=self.config.normalization,
            ar_order=self.config.ar_order,
            attn_bucket_heads=self.config.attn_bucket_heads,
            decoder_all_heads=self.config.decoder_all_heads,
            forward_expansion=self.config.forward_expansion,
            lr_split_factor=self.config.lr_split_factor,
            lr=self.config.lr,
            ttlr=self.config.ttlr,
            lambda_self=self.config.lambda_self,
            lambda_recon=self.config.lambda_recon,
            lambda_mse=self.config.lambda_mse,
            lambda_lr=self.config.lambda_lr,
            adaptive_add=self.config.adaptive_add,
            epoch_add=self.config.epoch_add,
            close_epochs=self.config.close_epochs,
            valid_size=self.config.valid_size,
            shuffle=self.config.shuffle,
            infer_mode=self.config.infer_mode,
            th=self.config.th,
        )

        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        self._model.fit(x_float32)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        return self._model.reconstruct(x_float32)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import torch  # noqa: lazy import

        path = Path(path)
        super().save(path)

        if self._model is not None:
            torch.save(self._model, path / self._MODEL_FILE)

        meta = {"num_channels_detected": self._num_channels_detected}
        (path / self._META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        import torch  # noqa: lazy import

        path = Path(path)
        instance = super().load(path, oid=oid)

        model_file = path / cls._MODEL_FILE
        if model_file.exists():
            instance._model = torch.load(model_file, weights_only=False)

        meta_file = path / cls._META_FILE
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            instance._num_channels_detected = meta.get("num_channels_detected")

        if instance._model is not None:
            instance._fitted = True

        return instance
