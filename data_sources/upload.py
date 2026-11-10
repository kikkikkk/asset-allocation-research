# -*- coding: utf-8 -*-
"""P1 上传数据层：复用原版 asset_data_loader.py 的稳健解析。

- 列名自适应（时间 / 收盘价(元)）、表头漂移校验、日期合法性、价格 > 0、≥3 条有效；
- 不自动公开用户上传数据：仅在会话内存中处理，不落盘、不写入部署包；
- 上传资产的收益口径 / 分红处理 / 授权状态由用户在 UI 显式声明（默认"未知"并警示）。
"""
from __future__ import annotations

import pandas as pd

from v1_reference.asset_data_loader import load_ifind_prices  # noqa: F401  (真实复用)


def parse_uploaded_prices(file_like) -> tuple[pd.Series, dict]:
    """解析上传的 CSV/Excel，返回 (价格序列, 解析元信息)。"""
    ser, meta = load_ifind_prices(file_like, return_meta=True)
    return ser, meta


def upload_template_frame() -> pd.DataFrame:
    """页面展示用：上传文件所需的最小列结构示例（不是数据、不生成模拟行情）。"""
    return pd.DataFrame(
        {
            "时间": ["YYYY-MM-DD", "YYYY-MM-DD"],
            "收盘价(元)": ["正数点位/净值", "正数点位/净值"],
            "说明": ["两列必需，其余列自动忽略", "日期升序、无重复、无空值"],
        }
    )
