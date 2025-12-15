#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance Spot (USDT quote) 全量扫描（支持多 base endpoint 兜底）：

- 永远不跳过交易对：任何错误/数据不足都输出一行，并标记 status/note
- MA7/30/100：数据不够则对应字段为缺失（None/NaN），但仍计算可计算的 MA
- 有效站上：连续2天“已收盘日线”收盘价 > 当天MA（MA缺失则有效站上=缺失）
- 放量：RVOL = 今日量 / 前N日均量（不含今日）；数据不够则缺失
- 24h quoteVolume 用于排序（只对 USDT quote 有意义）
- 429/418：按 Retry-After 退避重试；451：快速失败并切换 base endpoint
- 多 base endpoint：自动尝试
    https://api-gcp.binance.com
    https://api.binance.com
    https://api1.binance.com
    https://api2.binance.com
    https://api3.binance.com
    https://api4.binance.com

可选环境变量：
- BINANCE_BASE_URLS: 逗号分隔，覆盖/追加 base endpoint 优先顺序
  例：BINANCE_BASE_URLS="https://api-gcp.binance.com,https://api.binance.com"
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


PATH_EXCHANGE_INFO = "/api/v3/exchangeInfo"
PATH_TICKER_24HR = "/api/v3/ticker/24hr"
PATH_KLINES = "/api/v3/klines"


# ---- multi-base endpoints ----
DEFAULT_BINANCE_BASE_URLS: List[str] = [
    "https://api-gcp.binance.com",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]


def get_binance_base_urls() -> List[str]:
    env = os.environ.get("BINANCE_BASE_URLS", "").strip()
    env_list = [x.strip() for x in env.split(",") if x.strip()]
    merged = env_list + DEFAULT_BINANCE_BASE_URLS

    seen = set()
    out: List[str] = []
    for u in merged:
        if u and u not in seen:
            seen.add(u)
            out.append(u.rstrip("/"))
    return out


@dataclass
class RowOut:
    coin: str                 # baseAsset
    symbol: str               # e.g. BTCUSDT
    quote_asset: str          # USDT
    close: Optional[float]

    ma7: Optional[float]
    ma30: Optional[float]
    ma100: Optional[float]

    above_ma7: Optional[bool]
    above_ma30: Optional[bool]
    above_ma100: Optional[bool]

    valid_above_ma7: Optional[bool]
    valid_above_ma30: Optional[bool]
    valid_above_ma100: Optional[bool]

    rvol: Optional[float]
    volume_spike: Optional[bool]
    quote_volume_24h: Optional[float]

    bars_closed: int
    status: str               # OK / DATA_INSUFFICIENT / RATE_LIMITED / HTTP_ERROR / PARSE_ERROR
    note: str                 # 具体原因


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def request_json_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    timeout: int = 15,
    max_retries: int = 5,
    base_sleep: float = 0.8,
) -> Tuple[Optional[Any], str, str]:
    """
    返回 (json, status, note)
    status: OK / RATE_LIMITED / HTTP_ERROR / PARSE_ERROR
    note:   详细原因

    429/418：读取 Retry-After 秒数退避；没有就指数退避。
    451：快速返回 HTTP_ERROR（让外层切换 base endpoint）。
    """
    last_note = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)

            # 451：切换 base endpoint（同域名重试意义不大）
            if resp.status_code == 451:
                return None, "HTTP_ERROR", f"HTTP 451 (Unavailable for Legal Reasons) at {url}"

            if resp.status_code in (429, 418):
                retry_after = resp.headers.get("Retry-After")
                wait_s = None
                if retry_after is not None:
                    try:
                        wait_s = float(retry_after)
                    except Exception:
                        wait_s = None

                if wait_s is None:
                    wait_s = base_sleep * (2 ** (attempt - 1))

                last_note = (
                    f"HTTP {resp.status_code} rate limited; Retry-After={retry_after}; "
                    f"wait {wait_s:.2f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(wait_s)
                continue

            resp.raise_for_status()

            try:
                return resp.json(), "OK", ""
            except Exception as e:
                return None, "PARSE_ERROR", f"json parse failed: {e}"

        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            last_note = f"HTTPError {code}: {e}"
            # 非 429/418 不重试太多，给 2 次机会
            if attempt < min(max_retries, 2):
                time.sleep(base_sleep * attempt)
                continue
            return None, "HTTP_ERROR", last_note
        except requests.RequestException as e:
            last_note = f"RequestException: {e}"
            if attempt < max_retries:
                time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            return None, "HTTP_ERROR", last_note

    return None, "RATE_LIMITED", last_note or "rate limited"


class BinanceMultiBaseClient:
    """
    维护一个“当前可用 base_url”，失败时自动切换到下一个。
    """
    def __init__(self, session: requests.Session, base_urls: List[str]):
        self.session = session
        self.base_urls = base_urls[:] if base_urls else DEFAULT_BINANCE_BASE_URLS[:]
        self.current_base: Optional[str] = None

    def _iter_bases(self) -> List[str]:
        if self.current_base and self.current_base in self.base_urls:
            # 当前 base 优先，其余做 fallback
            others = [b for b in self.base_urls if b != self.current_base]
            return [self.current_base] + others
        return self.base_urls

    def get_json(
        self,
        path: str,
        params: Optional[dict] = None,
        timeout: int = 15,
        max_retries: int = 5,
        base_sleep: float = 0.8,
    ) -> Tuple[Optional[Any], str, str, str]:
        """
        返回 (json, status, note, used_base)
        """
        last_note = ""
        last_status = "HTTP_ERROR"
        used = ""

        for base in self._iter_bases():
            url = base + path
            js, st, note = request_json_with_retry(
                session=self.session,
                url=url,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                base_sleep=base_sleep,
            )

            if st == "OK" and js is not None:
                self.current_base = base
                return js, "OK", "", base

            # 记录失败原因，继续尝试下一个 base
            last_status = st
            last_note = (note or f"failed at {url}").strip()
            used = base

        return None, last_status, last_note, (used or "")


def fetch_exchange_info_spot(client: BinanceMultiBaseClient) -> Tuple[Optional[dict], str, str]:
    params = {"permissions": "SPOT", "symbolStatus": "TRADING"}
    js, st, note, _ = client.get_json(PATH_EXCHANGE_INFO, params=params)
    return js, st, note


def fetch_all_tickers_24hr(client: BinanceMultiBaseClient) -> Tuple[Optional[List[dict]], str, str]:
    js, st, note, _ = client.get_json(PATH_TICKER_24HR, params=None)
    return js, st, note


def fetch_klines(client: BinanceMultiBaseClient, symbol: str, interval: str, limit: int) -> Tuple[Optional[List[List[Any]]], str, str]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    js, st, note, _ = client.get_json(PATH_KLINES, params=params)
    return js, st, note


def klines_to_df(klines: List[List[Any]]) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    for c in ["open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df = df.dropna(subset=["close", "volume", "close_time"]).reset_index(drop=True)
    return df


def last_two_closed_rows(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], int]:
    now_ms = int(time.time() * 1000)
    closed = df[df["close_time"] < now_ms]
    if len(closed) == 0:
        return None, None, 0
    if len(closed) == 1:
        return None, closed.iloc[-1], 1
    return closed.iloc[-2], closed.iloc[-1], len(closed)


def compute_indicators_partial(
    df: pd.DataFrame,
    vol_lookback: int,
    rvol_threshold: float,
    require_vol_gt_yesterday: bool,
) -> Dict[str, Any]:
    """
    返回 dict，字段可能为 None（表示缺失）
    """
    df = df.copy()
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma100"] = df["close"].rolling(100).mean()

    prev_row, last_row, bars_closed = last_two_closed_rows(df)
    out: Dict[str, Any] = {"bars_closed": bars_closed}

    if last_row is None:
        out.update({
            "close": None,
            "ma7": None, "ma30": None, "ma100": None,
            "above_ma7": None, "above_ma30": None, "above_ma100": None,
            "valid_above_ma7": None, "valid_above_ma30": None, "valid_above_ma100": None,
            "rvol": None, "volume_spike": None,
        })
        return out

    close_last = safe_float(last_row["close"])
    ma7_last = safe_float(last_row["ma7"])
    ma30_last = safe_float(last_row["ma30"])
    ma100_last = safe_float(last_row["ma100"])

    out["close"] = close_last
    out["ma7"] = ma7_last
    out["ma30"] = ma30_last
    out["ma100"] = ma100_last

    def above(close: Optional[float], ma: Optional[float]) -> Optional[bool]:
        if close is None or ma is None:
            return None
        return bool(close > ma)

    out["above_ma7"] = above(close_last, ma7_last)
    out["above_ma30"] = above(close_last, ma30_last)
    out["above_ma100"] = above(close_last, ma100_last)

    # 有效站上：连续2天收盘 > 当天MA（两天都必须是“已收盘日线”且 MA 不缺失）
    def valid_above(prev: Optional[pd.Series], last: pd.Series, key_ma: str) -> Optional[bool]:
        if prev is None:
            return None
        c0 = safe_float(prev["close"])
        c1 = safe_float(last["close"])
        m0 = safe_float(prev[key_ma])
        m1 = safe_float(last[key_ma])
        if c0 is None or c1 is None or m0 is None or m1 is None:
            return None
        return bool((c0 > m0) and (c1 > m1))

    out["valid_above_ma7"] = valid_above(prev_row, last_row, "ma7")
    out["valid_above_ma30"] = valid_above(prev_row, last_row, "ma30")
    out["valid_above_ma100"] = valid_above(prev_row, last_row, "ma100")

    # 放量：RVOL = 今日量 / 前N日均量（不含今日）
    # 需要至少 N+1 根已收盘日线
    if bars_closed is None or bars_closed < vol_lookback + 1:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    # 取最后一根已收盘在 df 中的 index：用 close_time < now_ms 的最后 index
    now_ms = int(time.time() * 1000)
    closed_idx = df.index[df["close_time"] < now_ms]
    i_last = int(closed_idx[-1])

    vol_today = safe_float(df.loc[i_last, "volume"])
    if vol_today is None:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    vols_prev_n = df.loc[i_last - vol_lookback: i_last - 1, "volume"]
    vol_avg_n = safe_float(vols_prev_n.mean())
    if vol_avg_n is None or vol_avg_n <= 0:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    rvol = vol_today / vol_avg_n
    out["rvol"] = float(rvol)

    volume_spike: Optional[bool] = bool(rvol >= rvol_threshold)
    if require_vol_gt_yesterday and prev_row is not None:
        vol_yesterday = safe_float(prev_row["volume"])
        if vol_yesterday is not None:
            volume_spike = bool(volume_spike and (vol_today > vol_yesterday))
        else:
            volume_spike = None  # 昨日量缺失就标缺失
    out["volume_spike"] = volume_spike
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quote", default="USDT", help="只扫描某个计价资产（默认 USDT）")
    ap.add_argument("--top", type=int, default=0, help="只扫按24h quoteVolume排序的前N个；0=全部")
    ap.add_argument("--kline-limit", type=int, default=300, help="每个交易对拉取的日线数量（<=1000）")
    ap.add_argument("--vol-lookback", type=int, default=20, help="RVOL 均量回看天数N")
    ap.add_argument("--rvol", type=float, default=1.5, help="放量阈值：RVOL>=该值")
    ap.add_argument("--no-vol-gt-yesterday", action="store_true", help="不要求今日量>昨日量（更宽松）")
    ap.add_argument("--sleep", type=float, default=0.08, help="每次K线请求之间sleep秒数（防429）")
    ap.add_argument("--out", default="spot_usdt_scan.xlsx", help="输出Excel文件名")
    args = ap.parse_args()

    session = requests.Session()
    base_urls = get_binance_base_urls()
    client = BinanceMultiBaseClient(session=session, base_urls=base_urls)

    ex, ex_status, ex_note = fetch_exchange_info_spot(client)
    if ex_status != "OK" or not ex:
        raise SystemExit(
            f"exchangeInfo failed: {ex_status} {ex_note} "
            f"(tried bases: {', '.join(base_urls)})"
        )

    # 只保留 quoteAsset=USDT 的现货交易对
    symbol_to_base: Dict[str, str] = {}
    symbol_to_quote: Dict[str, str] = {}
    for s in ex.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != args.quote:
            continue
        sym = s.get("symbol")
        if not sym:
            continue
        symbol_to_base[sym] = s.get("baseAsset", sym)
        symbol_to_quote[sym] = s.get("quoteAsset", args.quote)

    symbols_all = list(symbol_to_base.keys())
    if not symbols_all:
        raise SystemExit(f"No TRADING SPOT symbols found for quote={args.quote}")

    # 24h ticker：拿 quoteVolume 排序（只对 USDT quote 可比）
    tickers, t_status, t_note = fetch_all_tickers_24hr(client)
    qv_map: Dict[str, float] = {sym: 0.0 for sym in symbols_all}
    if t_status == "OK" and isinstance(tickers, list):
        for t in tickers:
            sym = t.get("symbol")
            if sym in qv_map:
                try:
                    qv_map[sym] = float(t.get("quoteVolume", 0.0))
                except Exception:
                    qv_map[sym] = 0.0

    symbols_sorted = sorted(symbols_all, key=lambda s: qv_map.get(s, 0.0), reverse=True)
    if args.top and args.top > 0:
        symbols_sorted = symbols_sorted[: args.top]

    rows: List[RowOut] = []

    global_ticker_note = ""
    if t_status != "OK":
        global_ticker_note = f"ticker/24hr failed: {t_status} {t_note}"

    for sym in symbols_sorted:
        base = symbol_to_base.get(sym, sym)
        quote = symbol_to_quote.get(sym, args.quote)
        qv = qv_map.get(sym, 0.0)

        kl, k_status, k_note = fetch_klines(
            client,
            sym,
            interval="1d",
            limit=min(int(args.kline_limit), 1000),
        )
        if k_status != "OK" or not isinstance(kl, list):
            note = (k_note or "").strip()
            if global_ticker_note:
                note = (note + " | " + global_ticker_note).strip(" |")
            rows.append(RowOut(
                coin=base, symbol=sym, quote_asset=quote, close=None,
                ma7=None, ma30=None, ma100=None,
                above_ma7=None, above_ma30=None, above_ma100=None,
                valid_above_ma7=None, valid_above_ma30=None, valid_above_ma100=None,
                rvol=None, volume_spike=None,
                quote_volume_24h=float(qv),
                bars_closed=0,
                status=("RATE_LIMITED" if k_status == "RATE_LIMITED" else k_status),
                note=note
            ))
            time.sleep(max(0.0, args.sleep))
            continue

        try:
            df = klines_to_df(kl)
            ind = compute_indicators_partial(
                df=df,
                vol_lookback=int(args.vol_lookback),
                rvol_threshold=float(args.rvol),
                require_vol_gt_yesterday=(not args.no_vol_gt_yesterday),
            )

            status = "OK"
            note = ""
            if ind.get("bars_closed", 0) < 2:
                status = "DATA_INSUFFICIENT"
                note = "Not enough closed daily candles"
            elif ind.get("ma7") is None and ind.get("ma30") is None and ind.get("ma100") is None:
                status = "DATA_INSUFFICIENT"
                note = "Not enough history for MA windows"

            if global_ticker_note:
                note = (note + " | " + global_ticker_note).strip(" |")

            rows.append(RowOut(
                coin=base, symbol=sym, quote_asset=quote,
                close=ind.get("close"),

                ma7=ind.get("ma7"),
                ma30=ind.get("ma30"),
                ma100=ind.get("ma100"),

                above_ma7=ind.get("above_ma7"),
                above_ma30=ind.get("above_ma30"),
                above_ma100=ind.get("above_ma100"),

                valid_above_ma7=ind.get("valid_above_ma7"),
                valid_above_ma30=ind.get("valid_above_ma30"),
                valid_above_ma100=ind.get("valid_above_ma100"),

                rvol=ind.get("rvol"),
                volume_spike=ind.get("volume_spike"),
                quote_volume_24h=float(qv),

                bars_closed=int(ind.get("bars_closed", 0)),
                status=status,
                note=note
            ))
        except Exception as e:
            note = f"compute failed: {e}"
            if global_ticker_note:
                note = (note + " | " + global_ticker_note).strip(" |")
            rows.append(RowOut(
                coin=base, symbol=sym, quote_asset=quote, close=None,
                ma7=None, ma30=None, ma100=None,
                above_ma7=None, above_ma30=None, above_ma100=None,
                valid_above_ma7=None, valid_above_ma30=None, valid_above_ma100=None,
                rvol=None, volume_spike=None,
                quote_volume_24h=float(qv),
                bars_closed=0,
                status="PARSE_ERROR",
                note=note
            ))

        time.sleep(max(0.0, args.sleep))

    df_out = pd.DataFrame([asdict(r) for r in rows])

    cols = [
        "coin",
        "symbol",
        "close",
        "above_ma7",
        "above_ma30",
        "above_ma100",
        "valid_above_ma7",
        "valid_above_ma30",
        "valid_above_ma100",
        "volume_spike",
        "rvol",
        "ma7",
        "ma30",
        "ma100",
        "quote_volume_24h",
        "bars_closed",
        "status",
        "note",
    ]
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = None

    df_out = df_out[cols].sort_values(by=["quote_volume_24h", "symbol"], ascending=[False, True])
    df_out.to_excel(args.out, index=False)
    print(f"Saved: {args.out}  rows={len(df_out)}")


if __name__ == "__main__":
    main()
