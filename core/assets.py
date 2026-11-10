# -*- coding: utf-8 -*-
"""P1 通用资产层：资产元数据、N 资产注册表、收益口径分类与可比性检查。

设计原则
--------
- 不写死任何具体资产：模型与回测只依赖 asset_id 列表与价格矩阵列。
- 每个资产携带完整元数据（含收益口径、分红处理、再分发许可）。
- 内置资产只是**默认示例**（沿用原版两个资产 + 若干中证官方全收益指数），
  用户可勾选/取消、上传自有资产、或按注册表扩展。
- 口径不同（含息 vs 不含息）时生成显式警告，绝不静默视为可比。

收益口径说明（重要）
--------------------
- 前复权 / 后复权是**价格调整方式**，用于消除分红除权造成的价格跳空；
  后复权价格序列可近似还原含息总回报，但不等同于所有类型的总收益指数
  （税收、再投资时点、成分调整等差异仍存在）。
- **累计单位净值**（单位净值 + 历史分红累计）并非严格的复权净值：
  它未考虑分红的再投资复利，不能直接当作总回报序列使用。
- 债券净价指数不含应计利息；全价指数含应计利息；财富指数额外计入票息再投资。
- 不同口径的数据不可直接视为经济含义可比；混用时页面必须提示。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

# ---------------- 口径常量 ----------------
TOTAL_RETURN_BASES = {
    "全收益指数", "财富指数", "后复权价格", "复权单位净值",
}
PRICE_ONLY_BASES = {
    "价格指数", "净价指数", "全价指数", "未复权价格", "单位净值", "累计单位净值",
}
KNOWN_BASES = TOTAL_RETURN_BASES | PRICE_ONLY_BASES | {"未知", "其他"}

CALIBER_NOTES = {
    "全收益指数": "含成分股现金股息再投资的总回报指数（如中证 H00300 系列）",
    "财富指数": "含票息再投资的债券总回报指数（如中债 CBA06501）",
    "价格指数": "仅反映价格变动，**不含**现金股息再投资",
    "净价指数": "不含应计利息，不能与财富指数混用",
    "全价指数": "含应计利息，但仍未计入票息再投资复利",
    "后复权价格": "以上市首日为基准向后复权，可近似总回报；与总收益指数仍有细节差异",
    "前复权价格": "以当前时点向前调整历史价格，历史上会随新分红发生变化，不适合长周期回测",
    "复权单位净值": "基金分红再投资后的净值序列，优先于累计单位净值",
    "单位净值": "不含分红再投资，分红日会出现净值跳空",
    "累计单位净值": "仅为单位净值 + 历史分红累加，**未计复利**，不等于复权净值",
    "未知": "上传数据未声明口径，需用户自行确认",
}


@dataclass(frozen=True)
class AssetMeta:
    """单个资产的完整元数据（不假设所有资产字段相同，未知字段显式标注）。"""

    asset_id: str
    name: str
    code: str
    asset_class: str = "其他"                 # 股票 / 债券 / 商品 / 基金 / REITs / 其他
    instrument_type: str = "指数"             # 指数 / ETF / 公募基金 / 现货 / 其他
    currency: str = "CNY"
    source: str = "用户上传"
    source_url: str = ""
    frequency: str = "D"                      # D 日频 / W 周频 / M 月频
    return_basis: str = "未知"                # 见 KNOWN_BASES
    dividend_treatment: str = "未知"          # 不含股息 / 股息再投资 / 票息再投资 / 未知
    start_date: str | None = None
    end_date: str | None = None
    last_updated: str | None = None
    data_quality_status: str = "unknown"      # unknown / ok / warning / error
    redistribution_permission: str = "未知"   # 公开可再分发 / 仅运行时获取 / 用户提供需自审 / 未知
    fetch_key: str = ""                       # 内置公开源的获取标识；上传资产为空
    default_enabled: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_total_return(self) -> bool:
        return self.return_basis in TOTAL_RETURN_BASES

    @property
    def is_price_only(self) -> bool:
        return self.return_basis in PRICE_ONLY_BASES

    @property
    def label(self) -> str:
        return f"{self.name}（{self.code}）"


# ---------------- 内置资产注册表（默认示例，可替换/扩展） ----------------
_CSI = "中证指数有限公司官网（csindex.com.cn）"
_CB = "中国债券信息网（yield.chinabond.com.cn）"


def _builtin_registry() -> dict[str, AssetMeta]:
    items = [
        AssetMeta(
            asset_id="HS300_TR", name="沪深300全收益指数", code="H00300",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="全收益指数", dividend_treatment="股息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:H00300", default_enabled=True,
            notes="原版 P0 修复后默认股票口径；与中债财富指数同为含息总回报",
        ),
        AssetMeta(
            asset_id="CB7_10_WEALTH", name="中债-7-10年国债财富指数", code="CBA06501",
            asset_class="债券", instrument_type="指数", source=_CB,
            source_url="https://yield.chinabond.com.cn/",
            return_basis="财富指数", dividend_treatment="票息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="chinabond:CBA06501", default_enabled=True,
            notes="官方编制说明：基准日 2006-12-31、中债估值取价、市值加权、月度调仓、现金再投资周期=月度；不是无风险利率",
        ),
        AssetMeta(
            asset_id="CSI500_TR", name="中证小盘500全收益指数", code="H00905",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="全收益指数", dividend_treatment="股息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:H00905",
            notes="用于把股票侧扩展到多资产；与沪深300高度相关",
        ),
        AssetMeta(
            asset_id="CSI1000_TR", name="中证1000全收益指数", code="H00852",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="全收益指数", dividend_treatment="股息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:H00852", notes="小盘股票全收益口径",
        ),
        AssetMeta(
            asset_id="SSE50_TR", name="上证50全收益指数", code="H00016",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="全收益指数", dividend_treatment="股息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:H00016", notes="大盘股票全收益口径",
        ),
        AssetMeta(
            asset_id="CSI800_TR", name="中证800全收益指数", code="H00906",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="全收益指数", dividend_treatment="股息再投资",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:H00906", notes="大中盘股票全收益口径",
        ),
        AssetMeta(
            asset_id="HS300_PR", name="沪深300价格指数（对照）", code="000300",
            asset_class="股票", instrument_type="指数", source=_CSI,
            source_url="https://www.csindex.com.cn/",
            return_basis="价格指数", dividend_treatment="不含股息",
            redistribution_permission="仅运行时获取（不得随网页再分发原始数据）",
            fetch_key="csindex:000300", default_enabled=False,
            notes="⚠️ 不含股息再投资（历史股息贡献约 2.4~2.5pp/年），与财富指数口径不对称；仅作对照，默认不启用",
        ),
    ]
    return {a.asset_id: a for a in items}


class AssetRegistry:
    """N 资产注册表：内置资产 + 用户上传资产；支持增删改查与口径检查。"""

    def __init__(self, builtins: Iterable[AssetMeta] | None = None):
        base = _builtin_registry() if builtins is None else {a.asset_id: a for a in builtins}
        self._items: dict[str, AssetMeta] = dict(base)

    def all(self) -> list[AssetMeta]:
        return list(self._items.values())

    def get(self, asset_id: str) -> AssetMeta:
        if asset_id not in self._items:
            raise KeyError(f"资产 {asset_id} 不在注册表中")
        return self._items[asset_id]

    def subset(self, asset_ids: Iterable[str]) -> list[AssetMeta]:
        return [self.get(a) for a in asset_ids]

    def defaults(self) -> list[str]:
        return [a.asset_id for a in self._items.values() if a.default_enabled]

    def add_upload(self, meta: AssetMeta) -> None:
        if meta.asset_id in self._items:
            raise ValueError(f"asset_id 已存在：{meta.asset_id}")
        self._items[meta.asset_id] = meta

    def remove(self, asset_id: str) -> None:
        self._items.pop(asset_id, None)

    def update_quality(self, asset_id: str, status: str, start=None, end=None, updated=None) -> None:
        """生成替换后的元数据（frozen dataclass 不可变）。"""
        old = self.get(asset_id)
        self._items[asset_id] = AssetMeta(**{
            **old.to_dict(),
            "data_quality_status": status,
            "start_date": str(start) if start is not None else old.start_date,
            "end_date": str(end) if end is not None else old.end_date,
            "last_updated": str(updated) if updated is not None else old.last_updated,
        })

    def caliber_warnings(self, asset_ids: Iterable[str]) -> list[str]:
        """混用口径、未知口径、对照资产的显式提示（不阻断，但必须展示）。"""
        metas = self.subset(asset_ids)
        warns: list[str] = []
        tr = [m for m in metas if m.is_total_return]
        po = [m for m in metas if m.is_price_only]
        unknown = [m for m in metas if m.return_basis not in TOTAL_RETURN_BASES | PRICE_ONLY_BASES]
        if tr and po:
            warns.append(
                "⚠️ **收益口径不对称**："
                + "、".join(m.name for m in tr)
                + " 为含息总回报口径；"
                + "、".join(m.name for m in po)
                + " 不含股息/票息再投资。两者经济含义不可比，组合收益会系统性低估不含息资产，"
                  "结论仅供对照，不可直接用于模型优劣排名。"
            )
        for m in unknown:
            warns.append(f"⚠️ **{m.name}** 收益口径未声明（{m.return_basis}），请在「资产与数据」页补录口径。")
        for m in metas:
            if m.return_basis == "累计单位净值":
                warns.append(
                    f"⚠️ **{m.name}** 使用累计单位净值：该口径未计分红再投资复利，不能直接当作复权净值；"
                    "请改用复权单位净值或确认误差可接受。"
                )
            if m.return_basis == "前复权价格":
                warns.append(
                    f"⚠️ **{m.name}** 使用前复权价格：历史价格会随新分红变化，不适合长周期回测；"
                    "建议改用后复权或全收益指数。"
                )
            if m.redistribution_permission.startswith("用户提供"):
                warns.append(
                    f"ℹ️ **{m.name}** 为用户上传资产：请自行确认数据使用与再分发授权；"
                    "平台不会公开上传数据，也不会将其写入部署包。"
                )
        return warns

    def info_frame(self, asset_ids: Iterable[str] | None = None) -> "pd.DataFrame":
        metas = self.subset(asset_ids) if asset_ids is not None else self.all()
        return pd.DataFrame([m.to_dict() for m in metas]).set_index("asset_id")


def make_upload_meta(
    asset_id: str,
    name: str,
    code: str = "",
    asset_class: str = "其他",
    instrument_type: str = "其他",
    currency: str = "CNY",
    frequency: str = "D",
    return_basis: str = "未知",
    dividend_treatment: str = "未知",
    notes: str = "",
) -> AssetMeta:
    """为用户上传文件构造元数据（授权状态默认"用户提供需自审"）。"""
    if return_basis not in KNOWN_BASES:
        return_basis = "未知"
    return AssetMeta(
        asset_id=asset_id,
        name=name or asset_id,
        code=code or asset_id,
        asset_class=asset_class,
        instrument_type=instrument_type,
        currency=currency,
        source="用户上传（CSV/Excel）",
        source_url="",
        frequency=frequency,
        return_basis=return_basis,
        dividend_treatment=dividend_treatment,
        redistribution_permission="用户提供需自审（不随部署包分发）",
        fetch_key="",
        default_enabled=False,
        notes=notes or "上传文件请自行确认收益口径与分红处理，避免重复计入分红",
    )
