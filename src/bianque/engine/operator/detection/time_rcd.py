# -*- coding: utf-8 -*-

"""
Time-RCD 零样本异常检测算子（Scorer / 第 2 层）

将 ``bq_rcd`` 的 `TimeRCDPretrainTester.zero_shot` 接口封装为 TSA-Suite 算子，
属于 SingleScorer 路径：输出形状 ``(N,)`` 的逐时刻异常分数。

与 cicada 的本质差异：
    - Time-RCD 是预训练零样本模型，``fit()`` **不更新权重、不学统计量**，
      仅完成"加载预训练 checkpoint + 推断特征维度"的资源准备。
    - ``run()`` 每次对当次输入做一次 Z-score（走 ``zero_shot(ndarray)`` 默认路径），
      不持久化任何归一化参数，与原版 ``TimeRCDDataset(normalize=True)`` 行为一致。
    - ``save/load`` 仅持久化少量元信息（特征维度、checkpoint 路径），
      模型权重靠 HuggingFace Hub 缓存恢复。

包含:
    - TimeRCDScorerConfig: 实例参数
    - TimeRCDScorer: 单输出评分器（SingleScorerMixin）

示例用法::

    scorer = TimeRCDScorer(win_size=200, batch_size=2)
    scorer.fit(train_data)        # 加载 checkpoint + 推断 num_features
    scores = scorer.run(test_data)  # 形状 (N,) 的异常分数
"""

import json
import warnings
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bianque.engine.operator.base import (
    NumericOperator,
    UnsupervisedNumericOperatorMixin,
)
from bianque.engine.operator.detection.base import SingleScorerMixin

__all__ = ["TimeRCDScorerConfig", "TimeRCDScorer"]


class TimeRCDScorerConfig(BaseModel):
    """Time-RCD 评分器实例参数

    Attributes:
        win_size: 滑动窗口长度（推理用非重叠窗口拼分数）
        batch_size: 推理批大小
        patch_size: patch 大小，须与 checkpoint 匹配（HF 上的官方权重为 16）
        num_features: 输入特征通道数；None 时由 fit 阶段自动从数据推断
        checkpoint: 本地 checkpoint 路径；None 时自动从 HF Hub 下载并缓存
        score_form: 输出分数形式，``"prob"`` 为 softmax 后的异常概率，
            ``"logit"`` 为 logit_1 - logit_0
    """

    model_config = ConfigDict(frozen=True)

    win_size: int = Field(default=5000, gt=0, description="滑动窗口长度")
    batch_size: int = Field(default=64, gt=0, description="推理批大小")
    patch_size: int = Field(default=16, gt=0, description="patch 大小，须匹配 checkpoint")
    num_features: int | None = Field(
        default=None,
        description="输入特征通道数；None 时 fit 阶段自动推断",
    )
    checkpoint: str | None = Field(
        default=None,
        description="checkpoint 路径；None 时自动从 HF Hub 下载",
    )
    score_form: Literal["prob", "logit"] = Field(
        default="prob",
        description="输出分数形式",
    )


class TimeRCDScorer(
    SingleScorerMixin[None],
    UnsupervisedNumericOperatorMixin[None],
    NumericOperator[None, TimeRCDScorerConfig, None],
):
    """Time-RCD 零样本异常分数评分器

    基于 ``bq_rcd.time_rcd.TimeRCDPretrainTester`` 的零样本异常检测：

    - ``_fit_data``: 推断 num_features（None 时取 ``x.shape[1]``）→ 构造 tester
      （触发 HF Hub 下载或本地 checkpoint 解析）。**不更新权重、不学统计量**。
    - ``_run_data``: 直接将 ndarray 传给 ``tester.zero_shot``，由其内部对当次
      输入做一次 Z-score。

    输出:
        - 异常分数 ndarray，形状 ``(N,)``，其中 N == 输入长度。
        - DataFrame 输入时，输出列名为 ``["score"]``。

    泛型参数:
        - EO: None（无附加输出）
        - C: TimeRCDScorerConfig
        - RP: None（无运行参数）
    """

    _META_FILE = "time_rcd_meta.json"

    @classmethod
    def name(cls) -> str:
        return "time_rcd_scorer"

    def __init__(
        self,
        *,
        oid: str | None = None,
        config: TimeRCDScorerConfig | None = None,
        **kwargs,
    ) -> None:
        """初始化 Time-RCD 评分器。

        Args:
            oid: 算子实例唯一标识后缀
            config: 类型化实例参数
            **kwargs: 透传给 Config 的字段（win_size、batch_size、num_features 等）
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._tester = None
        self._num_features_detected: int | None = None
        self._checkpoint_path_resolved: str | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate_ndarray_input(self, x: np.ndarray, params) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDScorer 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDScorer 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

    # ------------------------------------------------------------------
    # 训练（实为加载预训练 checkpoint + 推断维度，不更新权重）
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """加载预训练权重 + 推断特征维度。

        本方法名为 ``_fit_data`` 仅为满足 TSA-Suite 框架"必须先 fit 再 run"的契约。
        Time-RCD 是预训练零样本模型，**不会**在此处更新模型权重，
        也**不会**学习任何归一化统计量。

        Args:
            x: 训练/校准数据，形状 (n_samples, n_features)
            params: 无训练参数
        """
        from bq_rcd.time_rcd import TimeRCDPretrainTester
        from bq_rcd.time_rcd.time_rcd_config import TimeRCDConfig, TimeSeriesConfig

        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDScorer 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDScorer 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

        num_features = (
            self.config.num_features
            if self.config.num_features is not None
            else x.shape[1]
        )
        self._num_features_detected = num_features

        ts_config = TimeSeriesConfig(
            patch_size=self.config.patch_size,
            num_features=num_features,
        )
        rcd_config = TimeRCDConfig(
            ts_config=ts_config,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
        )

        self._tester = TimeRCDPretrainTester(
            checkpoint_path=self.config.checkpoint,
            config=rcd_config,
        )
        self._checkpoint_path_resolved = self._tester.checkpoint_path

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(
        self,
        x: np.ndarray,
        params: None,
        idx: pd.Index | None = None,
    ) -> np.ndarray:
        """对输入做零样本异常分数推理。

        每次调用都会让 ``zero_shot`` 内部对当次输入做一次 Z-score，
        不复用任何"训练时"的统计量。

        Args:
            x: 输入数据，形状 (n_samples, n_features)
            params: 无运行参数
            idx: DataFrame 输入时的行索引（此处未使用）

        Returns:
            np.ndarray: 异常分数，形状 (n_samples,)
        """
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        with warnings.catch_warnings():
            # zero_shot 对 ndarray 输入会发 UserWarning 提示"将临时 fit 一组 Z-score"，
            # 这正是我们想要的零样本默认行为，无需冒泡到用户。
            warnings.simplefilter("ignore", UserWarning)
            scores = self._tester.zero_shot(
                x_float32,
                score_form=self.config.score_form,
            )
        return np.asarray(scores)

    # ------------------------------------------------------------------
    # 持久化（仅落元信息，模型权重靠 HF 缓存恢复）
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """保存算子状态到目录（仅 config + 元信息）。

        Args:
            path: 目标目录路径
        """
        path = Path(path)
        super().save(path)

        meta = {
            "num_features_detected": self._num_features_detected,
            "checkpoint_path_resolved": self._checkpoint_path_resolved,
        }
        (path / self._META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """从目录恢复算子（重建 tester，HF 命中缓存即秒级返回）。

        Args:
            path: 源目录路径
            oid: 算子实例唯一标识后缀

        Returns:
            恢复后的 TimeRCDScorer 实例
        """
        path = Path(path)
        instance = super().load(path, oid=oid)

        meta_file = path / cls._META_FILE
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            instance._num_features_detected = meta.get("num_features_detected")
            instance._checkpoint_path_resolved = meta.get("checkpoint_path_resolved")

        if instance._num_features_detected is not None:
            from bq_rcd.time_rcd import TimeRCDPretrainTester
            from bq_rcd.time_rcd.time_rcd_config import (
                TimeRCDConfig,
                TimeSeriesConfig,
            )

            ts_config = TimeSeriesConfig(
                patch_size=instance.config.patch_size,
                num_features=instance._num_features_detected,
            )
            rcd_config = TimeRCDConfig(
                ts_config=ts_config,
                batch_size=instance.config.batch_size,
                win_size=instance.config.win_size,
            )
            instance._tester = TimeRCDPretrainTester(
                checkpoint_path=(
                    instance.config.checkpoint
                    or instance._checkpoint_path_resolved
                ),
                config=rcd_config,
            )
            instance._fitted = True

        return instance
