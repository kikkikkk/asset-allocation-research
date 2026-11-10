# -*- coding: utf-8 -*-
"""公开数据源获取模块（云端部署版专用，多源回退）

用途：在运行时从公开、免费的数据渠道获取两个指数的日度收盘数据，
使部署包无需包含任何真实行情文件，也不再分发任何受授权限制的数据。

数据来源与口径（重要）：
1. 沪深300 默认使用**全收益指数** H00300（官方全称"沪深300全收益指数"，简称"300收益"，
   CSI 300 Total Return Index，含样本股现金股息再投资的**含息总回报**口径）：
   - 数据源：中证指数有限公司官网（csindex.com.cn，沪深300官方编制方）。
     经实测腾讯、新浪公开行情接口**均不提供** H00300，故该口径只有官方唯一渠道；
     官网不可用时明确报错（提示改用上传模式），**绝不静默降级为价格指数**。
2. 沪深300 **价格指数**（000300.SH，不含股息再投资）作为可选对照口径——三个备用渠道：
   a) 腾讯财经公开行情接口（web.ifzq.gtimg.cn，主源）；
   b) 中证指数有限公司官网；
   c) 新浪财经公开行情接口（money.finance.sina.com.cn）。
   三个渠道返回的均为同一指数（000300）同一口径（日收盘点位），已抽样交叉核验
   （2025-09-01：中证官网 4523.71 = 新浪 4523.711，差异仅为小数位）。
3. 中债-7-10年国债指数 财富指数（CBA06501，含票息再投资的**含息总回报**口径）：
   中国债券信息网（yield.chinabond.com.cn，中债估值中心官方公开网站）。
   该指数是债券价格表现（含息），不是"无风险利率"，界面上不得作为利率口径展示。

⚠️ 口径提示：默认"沪深300全收益（H00300）+ 中债财富（CBA06501）"两者均为含息总回报，
口径对称；若切换为价格指数对照，股票侧不含股息（历史上约 2.4~2.5pp/年），
与债券财富指数口径不对称，页面必须明确提示。

回退规则：
- 价格指数口径按 a → b → c 顺序尝试，任一渠道成功即采用，并在页面展示实际使用的渠道；
- 全收益指数仅中证官网一个渠道（其他公开渠道无此指数）；
- 中债官网为该指数唯一公开权威源，无备用渠道；
- 全部渠道失败时抛出异常，由调用方明确提示（建议改用上传模式），
  **绝不静默切换为模拟数据、绝不静默更换指数口径**。

不做任何鉴权、不使用任何凭据。
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests
from requests.exceptions import (ConnectionError as ReqConnectionError,
                                 HTTPError, JSONDecodeError, ProxyError,
                                 SSLError, Timeout)

# ---------------- 数据获取错误分类（保留底层原因，不吞异常） ----------------
CAT_PROXY = "代理故障"
CAT_NETWORK = "网络故障"
CAT_FORMAT = "接口格式变化"
CAT_VALIDATION = "数据校验失败"
CAT_UNKNOWN = "未知错误"


def classify_fetch_error(exc: BaseException) -> str:
    """把底层异常归入可操作类别，便于页面区分「网络/代理」与「接口/数据」问题。

    分类优先级：代理 > 网络 > 格式 > 校验 > 未知。
    ProxyError 是 ConnectionError 的子类，必须先判断。
    """
    if isinstance(exc, ProxyError) or "proxyerror" in type(exc).__name__.lower() \
            or "unable to connect to proxy" in str(exc).lower():
        return CAT_PROXY
    if isinstance(exc, (ReqConnectionError, Timeout, SSLError, HTTPError)):
        return CAT_NETWORK
    msg = str(exc).lower()
    # V2.3：日志里传递的是 `f"{type(exc).__name__}: {exc}"` 文本，
    # 已从真实异常变成 RuntimeError，isinstance 无法识别。补上异常类型名/典型消息
    # 的文本匹配，避免 ReadTimeout / ConnectionError 被误归为「未知错误」。
    if any(k in msg for k in (
            "readtimeout", "connecttimeout", "timeouterror", "read timed out",
            "connectionerror", "connection reset", "connection refused",
            "connection aborted", "sslerror", "httperror", "remote end closed",
            "name or service not known", "temporary failure in name resolution",
            "max retries exceeded", "failed to establish a new connection")):
        return CAT_NETWORK
    if isinstance(exc, JSONDecodeError) or "expecting value" in msg or "json" in msg \
            or "code=" in msg or "msg=" in msg:
        return CAT_FORMAT
    if any(k in msg for k in ("数据过少", "收盘价字段为空", "有效数据过少",
                              "返回数据过少", "未返回数据", "校验")):
        return CAT_VALIDATION
    if isinstance(exc, (KeyError, IndexError, TypeError)):
        return CAT_FORMAT
    return CAT_UNKNOWN


def describe_fetch_error(exc: BaseException) -> str:
    """格式化错误：分类标签 + 异常类型 + 原始消息（保留真实底层原因，不泄露凭据）。"""
    text = str(exc)
    # 兜底：任何情况下都不把 URL 中可能携带的认证参数带进日志
    for secret_key in ("access_token=", "refresh_token=", "token=", "password="):
        if secret_key in text.lower():
            text = text.split("?")[0] + "?[已脱敏]"
            break
    return f"[{classify_fetch_error(exc)}] {type(exc).__name__}: {text}"


def raise_classified(context: str, exc: BaseException, extra: str = "") -> None:
    """以统一格式抛 RuntimeError，并把底层异常挂到 __cause__。"""
    tail = f"；{extra}" if extra else ""
    raise RuntimeError(f"{context}：{describe_fetch_error(exc)}{tail}") from exc


_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# ---------------- 沪深300 口径常量 ----------------
HS300_CODE = "000300.SH"
HS300_TR_CODE = "H00300"
HS300_TR_NAME = "沪深300全收益指数（300收益，CSI 300 Total Return Index）"

# 口径常量：total_return = 全收益（默认，含股息再投资）；price = 价格指数（对照）
CALIBER_TR = "total_return"
CALIBER_PRICE = "price"

# 中债指数树上「中债-7-10年国债指数」的节点 ID（官网懒加载树的叶子节点），
# 同时保留按名称动态查找作为兜底，防止官网调整 ID。
_CB_NODE_ID = "8a8b2c8f5a492a01015a4ac986480043"
_CB_TREE_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/queryTree"
_CB_QUERY_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult"
_CB_INDEX_NAME = "中债-7-10年国债指数"

# 腾讯接口单次最多返回约 640 根K线，按约3年一段循环取全区间（2016年至今约10年）
_HS300_CHUNKS = [
    ("2016-01-04", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2024-12-31"),
    ("2025-01-01", "2030-12-31"),
]


def _series_from_pairs(pairs, name: str) -> pd.Series:
    """把 (date_str, close) 列表转成按日期升序去重的 Series。"""
    alld = pd.DataFrame(pairs, columns=["date", "close"]).drop_duplicates("date")
    ser = pd.Series(
        alld["close"].astype(float).to_numpy(),
        index=pd.to_datetime(alld["date"]),
        name=name,
    )
    return ser[ser.index.notna()].sort_index()


# ---------------- 沪深300 源1：腾讯（主源） ----------------
def _fetch_hs300_tencent() -> pd.Series:
    frames = []
    for beg, end in _HS300_CHUNKS:
        r = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"sh000300,day,{beg},{end},800,"},
            headers=_UA,
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
        data = (j.get("data") or {}).get("sh000300") or {}
        kl = data.get("day") or data.get("qfqday") or []
        if not kl:
            raise ValueError(
                f"腾讯行情接口 {beg}~{end} 段未返回数据（code={j.get('code')} msg={j.get('msg')}）。"
            )
        # kline 行格式: [date, open, close, high, low, volume, ...]，取收盘价
        frames.extend((x[0], x[2]) for x in kl)
        time.sleep(0.3)
    return _series_from_pairs(frames, "沪深300")


# ---------------- 沪深300 源2：中证指数官网（官方权威） ----------------
def _fetch_csindex_index(index_code: str, name: str,
                         start_date: str | None = None) -> pd.Series:
    """中证指数有限公司官网，沪深300官方编制方，单次返回日行情。

    index_code 为中证官网 6 位代码（000300 价格指数 / H00300 全收益指数）。
    `start_date`（YYYY-MM-DD）为 V2.3 增量更新用的可选起点；默认 None
    表示保持既有的 2016 起全区间获取。增量时最少行数阈值降为 2，
    但仍由 data_provider 的重叠一致性校验保证数据身份未变。
    """
    if start_date:
        start = pd.Timestamp(start_date).strftime("%Y%m%d")
        min_rows = 2
    else:
        start = "20160101"
        min_rows = 100
    end = (date.today() + timedelta(days=365)).strftime("%Y%m%d")
    last_err = None
    for attempt in range(2):  # 官网偶发超时，重试一次
        try:
            r = requests.get(
                "https://www.csindex.com.cn/csindex-home/perf/index-perf",
                params={"indexCode": index_code, "startDate": start, "endDate": end},
                headers=_UA,
                timeout=60,
            )
            r.raise_for_status()
            j = r.json()
            if str(j.get("code")) != "200":
                raise ValueError(f"中证指数官网接口返回异常：code={j.get('code')} msg={j.get('msg')}")
            rows = j.get("data") or []
            if len(rows) < min_rows:
                raise ValueError(f"中证指数官网返回数据过少（{len(rows)} 条）。")
            pairs = []
            for x in rows:
                close = x.get("close")
                if close in (None, ""):
                    continue
                pairs.append((str(x["tradeDate"]), close))
            if len(pairs) < min_rows:
                raise ValueError("中证指数官网返回的收盘价字段为空。")
            return _series_from_pairs(pairs, name)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0)
    # 保留真实底层异常（含代理/网络/格式分类），不再二次包装成笼统消息
    raise_classified(
        f"中证指数官网获取 {index_code} 失败（已重试 2 次）",
        last_err,
        extra="该错误已按代理/网络/接口格式/数据校验分类，原始异常保留在 __cause__",
    )


def _fetch_hs300_csindex() -> pd.Series:
    return _fetch_csindex_index("000300", "沪深300")


def _fetch_hs300_total_return_csindex(start_date: str | None = None) -> pd.Series:
    """沪深300全收益指数 H00300（中证官网唯一公开渠道）。

    官方名称：沪深300全收益指数（简称"300收益"，CSI 300 Total Return Index），
    含样本股现金股息再投资。`start_date` 仅用于增量更新，默认全区间。
    """
    return _fetch_csindex_index("H00300", "沪深300全收益", start_date=start_date)


# ---------------- 沪深300 源3：新浪财经 ----------------
def _fetch_hs300_sina() -> pd.Series:
    """新浪财经日K（scale=240 为日线），datalen 从最新往前取，2800 根覆盖 2016 年至今。"""
    r = requests.get(
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData",
        params={"symbol": "sh000300", "scale": "240", "ma": "no", "datalen": "2800"},
        headers=_UA,
        timeout=40,
    )
    r.raise_for_status()
    kl = r.json()
    if not isinstance(kl, list) or len(kl) < 100:
        raise ValueError(f"新浪行情接口返回数据过少（{len(kl) if isinstance(kl, list) else 0} 条）。")
    return _series_from_pairs([(x["day"], x["close"]) for x in kl], "沪深300")


_HS300_SOURCES = [
    ("腾讯财经行情接口（主源）", "https://web.ifzq.gtimg.cn", "_fetch_hs300_tencent"),
    ("中证指数官网（官方编制方）", "https://www.csindex.com.cn", "_fetch_hs300_csindex"),
    ("新浪财经行情接口", "https://money.finance.sina.com.cn", "_fetch_hs300_sina"),
]


def fetch_hs300(caliber: str = CALIBER_TR,
                start_date: str | None = None) -> tuple[pd.Series, dict]:
    """获取沪深300日收盘数据（指定口径），任一渠道成功即返回。

    caliber = "total_return"（默认）：H00300 沪深300全收益指数（含股息再投资），
        仅中证官网一个公开渠道（腾讯/新浪均不提供该指数）；失败即抛错。
    caliber = "price"：000300.SH 价格指数（不含股息，仅作对照口径），
        按腾讯 → 中证官网 → 新浪 顺序回退。

    返回 (Series, 来源信息)；全部失败时抛出 RuntimeError（含各渠道错误摘要）。
    渠道函数在运行时按名称解析（globals()[func_name]），便于单元测试替换。
    """
    if caliber == CALIBER_TR:
        try:
            # 无 start_date 时保持原有单参数调用形式（兼容既有调用方与测试桩）
            ser = (_fetch_hs300_total_return_csindex(start_date=start_date)
                   if start_date else _fetch_hs300_total_return_csindex())
            if len(ser) >= (2 if start_date else 100):
                return ser, {
                    "source": "中证指数官网（官方编制方，唯一公开渠道）",
                    "host": "https://www.csindex.com.cn",
                    "code": HS300_TR_CODE,
                    "name": HS300_TR_NAME,
                    "caliber": CALIBER_TR,
                    "rows": int(len(ser)),
                    "start": ser.index[0],
                    "end": ser.index[-1],
                }
            raise ValueError(f"全收益指数返回数据过少（{len(ser)} 条）")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[{classify_fetch_error(exc)}] 沪深300全收益指数（{HS300_TR_CODE}）获取失败"
                "——该指数仅中证指数官网一个公开渠道（腾讯/新浪均不提供），不做任何静默降级。\n"
                f"错误信息：{describe_fetch_error(exc)}\n"
                "请稍后重试，或在左侧「数据来源」改为上传真实数据（须为全收益口径）。"
            ) from exc

    errors = []
    for label, host, func_name in _HS300_SOURCES:
        fn = globals()[func_name]
        try:
            ser = fn()
            if len(ser) >= 100:
                return ser, {
                    "source": label,
                    "host": host,
                    "code": HS300_CODE,
                    "name": "沪深300指数（价格指数，不含股息）",
                    "caliber": CALIBER_PRICE,
                    "rows": int(len(ser)),
                    "start": ser.index[0],
                    "end": ser.index[-1],
                }
            errors.append(f"{label}：{describe_fetch_error(ValueError(f'返回数据过少（{len(ser)} 条）'))}")
        except Exception as exc:  # noqa: BLE001 —— 单渠道失败记录后继续尝试下一渠道
            errors.append(f"{label}（{host}）：{describe_fetch_error(exc)}")
    joined = "；".join(errors)
    if CAT_PROXY in joined:
        category = CAT_PROXY
    elif CAT_NETWORK in joined:
        category = CAT_NETWORK
    elif CAT_FORMAT in joined:
        category = CAT_FORMAT
    elif CAT_VALIDATION in joined:
        category = CAT_VALIDATION
    else:
        category = CAT_UNKNOWN
    raise RuntimeError(
        f"[{category}] 沪深300所有公开数据渠道均获取失败：\n- " + "\n- ".join(errors)
    )


# ---------------- 中债-7-10年国债指数（财富指数 CBA06501） ----------------
def fetch_bond_wealth() -> tuple[pd.Series, dict]:
    """中国债券信息网公开接口获取中债-7-10年国债指数财富指数日度数据（2007年至今）。

    口径：财富指数（CFZS，含票息再投资），即 CBA06501 的含息总回报口径，
    是债券价格表现而非"无风险利率"。
    """
    s = requests.Session()
    s.headers.update({**_UA, "X-Requested-With": "XMLHttpRequest"})
    node_id = _resolve_node_id(s)

    try:
        r = s.post(
            _CB_QUERY_URL,
            params={
                "indexid": node_id,
                "qxlxt": "00",
                "ltcslx": "",
                "zslxt": "CFZS",  # CFZS = 财富指数（含票息再投资）
                "zslxt1": "",
                "lx": "1",
                "locale": "zh_CN",
            },
            timeout=40,
        )
        r.raise_for_status()
        j = r.json()
    except Exception as exc:  # noqa: BLE001 —— 保留底层代理/网络/格式原因
        raise_classified(f"中债财富指数（{_CB_INDEX_NAME}）请求失败", exc)
    if "emsg" in j:
        raise ValueError(f"[{CAT_FORMAT}] 中国债券信息网接口返回错误：{j['emsg']}")
    raw = j.get("CFZS_00") or {}
    if len(raw) < 10:
        raise ValueError(f"[{CAT_VALIDATION}] 中国债券信息网返回数据过少（{len(raw)} 条）。")

    # 官网时间戳为北京时间零点的 UTC 表示（差 8 小时），+8h 后归一到交易日日期。
    idx = pd.to_datetime([int(t) for t in raw], unit="ms") + pd.Timedelta(hours=8)
    ser = (
        pd.Series(
            pd.to_numeric(list(raw.values()), errors="coerce").astype(float),
            index=idx.normalize(),
            name=_CB_INDEX_NAME,
        )
        .sort_index()
    )
    ser = ser[ser.index.notna() & ser.notna()]
    if len(ser) < 10:
        raise ValueError("中国债券信息网返回的有效数据过少。")
    meta = {
        "source": "中国债券信息网（中债估值中心官方公开网站）",
        "host": "yield.chinabond.com.cn",
        "code": "CBA06501（财富指数口径）",
        "name": f"{_CB_INDEX_NAME}（财富指数，含票息再投资）",
        "rows": int(len(ser)),
        "start": ser.index[0],
        "end": ser.index[-1],
    }
    return ser, meta


def _resolve_node_id(session) -> str:
    """先按硬编码 ID，失败则从官网指数树按名称动态查找。

    独立审查修复：保留每一步的真实底层异常，按代理/网络/格式/校验分类后上报，
    不再把网络故障统一伪装成「接口可能已调整」。
    """
    errors: list[str] = []
    ok, reason = _probe_node(session, _CB_NODE_ID)
    if ok:
        return _CB_NODE_ID
    errors.append(f"硬编码节点探测失败：{reason}")
    try:
        resp = session.get(_CB_TREE_URL, params={"locale": "zh_CN"}, timeout=30)
        resp.raise_for_status()
        tree = resp.json()
        nodes = tree if isinstance(tree, list) else tree.get("data") or tree.get("rows") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(f"[{CAT_FORMAT}] 指数树返回结构异常（节点为空或非列表）")

        def walk(items):
            for n in items if isinstance(items, list) else []:
                if n.get("name") == _CB_INDEX_NAME and n.get("isParent") != "true":
                    return n.get("id")
                found = walk(n.get("children") or n.get("nodes") or [])
                if found:
                    return found
            return None

        nid = walk(nodes)
        if not nid:
            raise ValueError(f"[{CAT_FORMAT}] 指数树中未找到「{_CB_INDEX_NAME}」节点，接口结构可能已调整")
        ok2, reason2 = _probe_node(session, nid)
        if ok2:
            return nid
        errors.append(f"动态节点 {nid} 探测失败：{reason2}")
    except Exception as exc:  # noqa: BLE001
        errors.append(describe_fetch_error(exc))

    # 若任一步是代理/网络故障，明确归因为连接问题，而不是接口改版
    joined = " | ".join(errors)
    if CAT_PROXY in joined:
        category = CAT_PROXY
    elif CAT_NETWORK in joined:
        category = CAT_NETWORK
    elif CAT_FORMAT in joined:
        category = CAT_FORMAT
    else:
        category = CAT_UNKNOWN
    raise RuntimeError(
        f"[{category}] 未能从中国债券信息网定位「中债-7-10年国债指数」。"
        f"底层原因：{joined}"
    )


def _probe_node(session, node_id: str) -> tuple[bool, str]:
    """探测某个节点 ID 是否能返回财富指数数据。

    返回 (是否成功, 失败原因)；**不再吞掉网络异常**，原因带分类标签。
    """
    try:
        r = session.post(
            _CB_QUERY_URL,
            params={
                "indexid": node_id, "qxlxt": "00", "ltcslx": "",
                "zslxt": "CFZS", "zslxt1": "", "lx": "1", "locale": "zh_CN",
            },
            timeout=25,
        )
        r.raise_for_status()
        j = r.json()
        if "emsg" in j:
            return False, f"[{CAT_FORMAT}] 接口返回 emsg={j['emsg']}"
        if len(j.get("CFZS_00") or {}) < 10:
            return False, f"[{CAT_VALIDATION}] CFZS_00 数据量不足 10 条"
        return True, ""
    except Exception as exc:  # noqa: BLE001 —— 保留底层原因，不再静默返回 False
        return False, describe_fetch_error(exc)


def fetch_public_series(
    stock_caliber: str = CALIBER_TR,
) -> tuple[pd.Series, pd.Series, dict, dict]:
    """一次获取两个资产的公开数据（沪深300口径由 stock_caliber 决定）。

    返回 (沪深300, 中债-7-10年国债财富指数, 沪深300来源信息, 债券来源信息)。
    任一资产全部渠道失败时抛异常，由调用方明确提示，绝不静默降级。
    """
    stock, stock_meta = fetch_hs300(stock_caliber)
    bond, bond_meta = fetch_bond_wealth()
    return stock, bond, stock_meta, bond_meta
