# -*- coding: utf-8 -*-
"""V2.3 真实行情持久化快照缓存（只保存已通过质量校验的合法公开数据）。

设计要点
--------
- **逐资产独立快照**：一个资产更新失败不会污染另一个资产，
  也不会把不同截止日期的数据强行拼接（对齐仍由 align_price_frame 取交集）。
- **原子写入 + 文件锁**：临时文件 → os.replace，配合 fcntl.flock，
  多会话/多页面同时刷新不会写出半截文件。
- **身份 + 内容校验**：读取时核对 asset_id / code / fetch_key / 收益口径 /
  sha256 校验和 / 日期升序 / 正值 / 最小行数 / 质量状态；
  任一不符即拒绝使用，绝不静默修补。
- **绝不生成数据**：本模块只持久化接口返回的真实价格，不插值、不前向填充、
  不生成模拟行情。
- 缓存目录位于项目内 `data_cache/`，由 .gitignore 排除，不进 Git、不上传。
- **云端兼容**：Streamlit Cloud 等环境的磁盘可能只读或随时被清空，
  因此所有持久化写入都在失败时优雅降级（保留本次接口获取的真实数据，
  仅依赖进程内 `st.cache_data`），绝不因写盘失败使整个网页不可用；
  也不把服务器磁盘当作永久数据库——重启丢失后重新从公开接口获取。
- **时区统一**：指数与交易日历按北京时间（Asia/Shanghai，UTC+8）判断，
  不依赖服务器本地时区（云端通常为 UTC）。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Windows 无 fcntl；缺失时退化为仅进程内线程锁（云端 Linux 仍用文件锁）
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
SNAPSHOT_VERSION = 1
MIN_OBS = 60

# 北京时间固定偏移：不依赖 tzdata，云端 UTC 服务器同样正确
BEIJING_TZ = timezone(timedelta(hours=8))

_thread_lock = threading.RLock()
_lock_handles: dict[Path, object] = {}
# 磁盘只读时的进程内状态/快照回退（避免 rerun 丢数据或重复请求接口）
# 云端重启后内存自然清空，届时重新从公开接口获取，不把服务器磁盘当永久数据库。
_mem_state: dict[str, dict] = {}
_mem_snapshots: dict[tuple[str, str], dict] = {}
_readonly_dirs: set[str] = set()


def beijing_now(now: datetime | None = None) -> datetime:
    """返回北京时间 datetime；naive 输入按北京时间解释（测试可控时间保持一致）。"""
    if now is None:
        return datetime.now(BEIJING_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=BEIJING_TZ)
    return now.astimezone(BEIJING_TZ)


def _now_iso(now: datetime | None = None) -> str:
    return beijing_now(now).isoformat(timespec="seconds")


def _pairs_from_series(series: pd.Series) -> list[list]:
    s = pd.Series(series).dropna().sort_index()
    return [[pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)] for d, v in s.items()]


def compute_checksum(pairs: list[list]) -> str:
    raw = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def cache_lock(cache_dir: Path | None = None):
    """进程内线程锁 + 跨进程文件锁，保证读写原子性。

    同一进程复用同一个文件句柄：flock 与 RLock 都可重入，
    允许 read_state / write_state 在 claim 临界区内嵌套调用而不死锁。

    云端磁盘只读 / 无写权限 / 无 fcntl 时降级为「仅线程锁」：
    读操作仍可执行（若目录里恰好有快照），写操作由调用方捕获 OSError，
    绝不因锁文件创建失败而让整个数据加载崩溃。
    """
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    with _thread_lock:
        locked = False
        fh = None
        try:
            d.mkdir(parents=True, exist_ok=True)
            if fcntl is not None:
                fh = _lock_handles.get(d)
                if fh is None or getattr(fh, "closed", False):
                    fh = (d / ".lock").open("a+")
                    _lock_handles[d] = fh
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                locked = True
        except OSError:
            _readonly_dirs.add(str(d))
        try:
            yield
        finally:
            if locked and fh is not None and fcntl is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _snapshot_path(asset_id: str, cache_dir: Path) -> Path:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in asset_id)
    return cache_dir / f"snapshot_{safe}.json"


def state_path(cache_dir: Path | None = None) -> Path:
    return (Path(cache_dir) if cache_dir else CACHE_DIR) / "_state.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_snapshot(
    meta: dict,
    series: pd.Series,
    quality: dict,
    *,
    now: datetime | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """保存一个**已通过质量校验**的真实行情快照。

    meta 至少包含 asset_id / name / code / source / source_url / fetch_key /
    return_basis / dividend_treatment。任一身份字段缺失即拒绝保存。
    """
    required = ("asset_id", "name", "code", "source", "fetch_key",
                "return_basis", "dividend_treatment")
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise ValueError(f"快照缺少身份字段，拒绝保存：{missing}")
    s = pd.Series(series).dropna().sort_index()
    if len(s) < MIN_OBS:
        raise ValueError(f"有效观测 {len(s)} < {MIN_OBS}，拒绝保存")
    vals = s.to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals <= 0).any():
        raise ValueError("价格含非有限值或非正值，拒绝保存")
    if s.index.has_duplicates:
        raise ValueError("日期存在重复，拒绝保存")
    if not bool(quality.get("ok")):
        raise ValueError("质量校验未通过，拒绝保存")

    pairs = _pairs_from_series(s)
    payload = {
        "cache_version": SNAPSHOT_VERSION,
        "asset_id": meta["asset_id"],
        "name": meta["name"],
        "code": meta["code"],
        "source": meta["source"],
        "source_url": meta.get("source_url", ""),
        "fetch_key": meta["fetch_key"],
        "return_basis": meta["return_basis"],
        "dividend_treatment": meta["dividend_treatment"],
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
        "fetched_at": _now_iso(now),
        "rows": int(len(s)),
        "quality_ok": True,
        "quality_summary": {
            k: quality.get(k) for k in
            ("rows_valid", "duplicate_dates", "missing_values",
             "invalid_prices", "extreme_returns", "warnings")
            if k in quality
        },
        "checksum": compute_checksum(pairs),
        "prices": pairs,
    }
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    result = {k: v for k, v in payload.items() if k != "prices"}
    # 进程内始终保留一份：云端磁盘只读/被清空时，rerun 仍可恢复本次真实数据
    _mem_snapshots[(str(d), meta["asset_id"])] = payload
    try:
        with cache_lock(d):
            _atomic_write_json(_snapshot_path(meta["asset_id"], d), payload)
        result["persisted"] = True
    except OSError as exc:  # 云端只读磁盘：保留本次真实数据，仅不落盘
        _readonly_dirs.add(str(d))
        result["persisted"] = False
        result["persist_error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_snapshot(
    asset_id: str,
    expected_meta: dict,
    *,
    cache_dir: Path | None = None,
    min_obs: int = MIN_OBS,
) -> tuple[pd.Series | None, dict | None, str]:
    """读取并严格校验快照。返回 (序列, 元数据, 错误原因)。

    任何校验失败都返回 (None, None, 原因)，调用方必须保留原有健康状态，
    不得用无效数据覆盖，也不得把损坏缓存当作成功。
    """
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    path = _snapshot_path(asset_id, d)
    if path.is_file():
        try:
            with cache_lock(d):
                raw = path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return None, None, f"快照读取/解析失败：{type(exc).__name__}: {exc}"
    else:
        # 磁盘无文件（云端只读/重启后磁盘被清空）：用本次进程内已校验的快照
        doc = _mem_snapshots.get((str(d), asset_id))
        if doc is None:
            return None, None, "无本地快照"

    if doc.get("cache_version") != SNAPSHOT_VERSION:
        return None, None, "快照版本不兼容（需重新获取）"
    for key in ("asset_id", "code", "fetch_key", "return_basis", "dividend_treatment"):
        exp = str(expected_meta.get(key, ""))
        got = str(doc.get(key, ""))
        if exp and got != exp:
            return None, None, f"快照身份不一致：{key} 期望 {exp}，实际 {got}"
    if doc.get("asset_id") != asset_id:
        return None, None, "快照 asset_id 不匹配"
    if not doc.get("quality_ok"):
        return None, None, "快照质量校验未通过"
    pairs = doc.get("prices")
    if not isinstance(pairs, list) or len(pairs) < min_obs:
        return None, None, f"快照有效观测不足（{len(pairs) if isinstance(pairs, list) else 0} 条）"
    if compute_checksum(pairs) != doc.get("checksum"):
        return None, None, "快照校验和不匹配（内容损坏或被修改）"
    try:
        idx = pd.to_datetime([p[0] for p in pairs])
        vals = np.array([float(p[1]) for p in pairs], dtype=float)
        s = pd.Series(vals, index=pd.DatetimeIndex(idx))
    except Exception as exc:  # noqa: BLE001
        return None, None, f"快照数据结构异常：{type(exc).__name__}: {exc}"
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        return None, None, "快照日期重复或未升序"
    if not np.isfinite(vals).all() or (vals <= 0).any():
        return None, None, "快照含非有限或非正价格"
    meta = {k: v for k, v in doc.items() if k != "prices"}
    return s.rename(asset_id), meta, ""


def read_state(cache_dir: Path | None = None) -> dict:
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    p = state_path(d)
    if not p.is_file():
        return dict(_mem_state.get(str(d), {}))
    try:
        with cache_lock(d):
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 状态文件损坏/不可读，回退进程内状态
        return dict(_mem_state.get(str(d), {}))


def write_state(state: dict, cache_dir: Path | None = None) -> None:
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    key = str(d)
    _mem_state[key] = dict(state)   # 磁盘失败时进程内仍有一致状态
    try:
        with cache_lock(d):
            _atomic_write_json(state_path(d), state)
        _readonly_dirs.discard(key)
    except OSError:  # 不抛异常：云端只读不影响接口获取与本次结果
        _readonly_dirs.add(key)


def storage_is_persistent(cache_dir: Path | None = None) -> bool:
    """供页面展示：本次快照是否成功落盘（云端磁盘可能只读/非永久）。"""
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    return str(d) not in _readonly_dirs


def snapshot_summary(cache_dir: Path | None = None) -> dict[str, dict]:
    """列出所有资产快照的简要信息（不含行情值），供页面展示。"""
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    out: dict[str, dict] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("snapshot_*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            aid = doc.get("asset_id", p.stem)
            out[aid] = {k: doc.get(k) for k in
                        ("name", "code", "source", "start", "end",
                         "fetched_at", "rows", "quality_ok", "checksum")}
        except Exception:  # noqa: BLE001
            continue
    return out
