#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OKX Spot USDT scanner (Binance-like CLI):

CLI (same as your Binance script):
  --quote USDT
  --top 0
  --kline-limit 600
  --vol-lookback 20
  --rvol 1.5
  --no-vol-gt-yesterday
  --sleep 0.12
  --out all_usdt.xlsx

Data sources (OKX v5):
- Instruments: GET /api/v5/public/instruments?instType=SPOT  (filter quoteCcy=USDT, state=live) :contentReference[oaicite:4]{index=4}
- Tickers:     GET /api/v5/market/tickers?instType=SPOT      (use volCcy24h as 24h quote volume for sorting) :contentReference[oaicite:5]{index=5}
- Candles:     GET /api/v5/market/candles?instId=...&bar=1Dutc&limit=...
               Candle array layout: [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]  :contentReference[oaicite:6]{index=6}
               We ONLY use confirm=1 candles (closed).
               limit per request is commonly 300 for /market/candles; we paginate if kline-limit > 300. :contentReference[oaicite:7]{index=7}

Rate-limit:
- OKX may return API error code 50011 for rate limit; also possible HTTP 429. We backoff and continue. :contentReference[oaicite:8]{index=8}
"""

from __future__ import annotations

import argparse
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd


OKX_REST = "https://www.okx.com"

PATH_INSTRUMENTS = "/api/v5/public/instruments"
PATH_TICKERS = "/api/v5/market/tickers"
PATH_CANDLES = "/api/v5/market/candles"


# ----------------------------
# Output row (match Binance-like columns)
# ----------------------------
@dataclass
class RowOut:
    coin: str
    symbol: str
    quote_asset: str
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
    status: str
    note: str


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        v = float(s)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


# ----------------------------
# HTTP helper with retry/backoff for OKX
# - handles HTTP 429
# - handles OKX JSON code 50011 (rate limit)
# ----------------------------
class OKXClient:
    def __init__(self, base_url: str = OKX_REST, timeout: int = 12):
        self.base_url = base_url.rstrip("/")
        self.sess = requests.Session()
        self.timeout = timeout

    def get_json_with_retry(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 6,
        base_sleep: float = 0.8,
    ) -> Tuple[Optional[dict], str, str]:
        """
        returns: (json, status, note)
        status: OK / RATE_LIMITED / HTTP_ERROR / PARSE_ERROR / API_ERROR
        note: detail
        """
        url = f"{self.base_url}{path}"
        params = params or {}
        last_note = ""

        for attempt in range(1, max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_note = f"RequestException: {e}"
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2)
                continue

            # HTTP 429
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_s = None
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except Exception:
                        wait_s = None
                if wait_s is None:
                    wait_s = base_sleep * (2 ** (attempt - 1))
                last_note = f"HTTP 429; Retry-After={retry_after}; wait={wait_s:.2f}s (attempt {attempt}/{max_retries})"
                time.sleep(wait_s + random.random() * 0.2)
                if attempt == max_retries:
                    return None, "RATE_LIMITED", last_note
                continue

            if r.status_code != 200:
                last_note = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2)
                if attempt == max_retries:
                    return None, "HTTP_ERROR", last_note
                continue

            try:
                js = r.json()
            except Exception as e:
                last_note = f"JSON parse failed: {e}"
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2)
                if attempt == max_retries:
                    return None, "PARSE_ERROR", last_note
                continue

            # OKX standard: {"code":"0","msg":"","data":[...]}
            code = js.get("code") if isinstance(js, dict) else None
            if code not in (None, "0", 0):
                msg = js.get("msg", "")
                # rate limit code 50011 is documented as "Rate limit reached" :contentReference[oaicite:9]{index=9}
                if str(code) == "50011":
                    wait_s = base_sleep * (2 ** (attempt - 1))
                    last_note = f"API rate limit code=50011 msg={msg}; wait={wait_s:.2f}s (attempt {attempt}/{max_retries})"
                    time.sleep(wait_s + random.random() * 0.2)
                    if attempt == max_retries:
                        return None, "RATE_LIMITED", last_note
                    continue

                last_note = f"API error code={code} msg={msg}"
                time.sleep(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.2)
                if attempt == max_retries:
                    return None, "API_ERROR", last_note
                continue

            return js, "OK", ""

        return None, "HTTP_ERROR", last_note or "unknown error"


# ----------------------------
# OKX data fetching
# ----------------------------
def fetch_spot_instruments_usdt(client: OKXClient, quote: str) -> Tuple[Dict[str, Dict[str, str]], str]:
    """
    returns:
      instruments_map: instId -> {"base": baseCcy, "quote": quoteCcy}
      note (global)
    """
    js, st, note = client.get_json_with_retry(PATH_INSTRUMENTS, {"instType": "SPOT"})
    if st != "OK" or not js:
        raise RuntimeError(f"instruments failed: {st} {note}")

    items = js.get("data", [])
    out: Dict[str, Dict[str, str]] = {}
    for it in items:
        if it.get("state") != "live":
            continue
        if it.get("quoteCcy") != quote:
            continue
        inst_id = it.get("instId")
        base = it.get("baseCcy")
        quote_ccy = it.get("quoteCcy")
        if inst_id and base and quote_ccy:
            out[inst_id] = {"base": base, "quote": quote_ccy}
    return out, ""


def fetch_all_spot_tickers(client: OKXClient) -> Tuple[Dict[str, dict], str]:
    """
    returns tickers_map: instId -> ticker_obj
    We'll use volCcy24h (24h trading volume in quote ccy) for sorting for USDT pairs. :contentReference[oaicite:10]{index=10}
    """
    js, st, note = client.get_json_with_retry(PATH_TICKERS, {"instType": "SPOT"})
    if st != "OK" or not js:
        return {}, f"tickers failed: {st} {note}"

    items = js.get("data", [])
    mp = {}
    for it in items:
        inst_id = it.get("instId")
        if inst_id:
            mp[inst_id] = it
    return mp, ""


def fetch_closed_daily_candles_paginated(
    client: OKXClient,
    inst_id: str,
    want: int,
    sleep_s: float,
    bar: str = "1Dutc",
) -> Tuple[Optional[List[dict]], str, str]:
    """
    Fetch CLOSED daily candles (confirm=1), in ascending time.
    OKX candle array layout includes 'confirm' at index 8. :contentReference[oaicite:11]{index=11}

    /api/v5/market/candles returns newest-first. We'll paginate with 'after' to get older data.
    Single request limit is commonly up to 300 for market candles; we cap each call to 300 and loop. :contentReference[oaicite:12]{index=12}
    """
    per_call = min(300, max(1, want))  # cap to 300
    all_rows: List[dict] = []
    after: Optional[int] = None  # timestamp in ms

    # limit total loops
    max_loops = 10 if want <= 3000 else 20

    for _ in range(max_loops):
        params: Dict[str, Any] = {"instId": inst_id, "bar": bar, "limit": str(per_call)}
        if after is not None:
            params["after"] = str(after)

        js, st, note = client.get_json_with_retry(PATH_CANDLES, params=params)
        if st != "OK" or not js:
            return None, ("RATE_LIMITED" if st == "RATE_LIMITED" else st), note

        rows = js.get("data", [])
        if not rows:
            break

        parsed_batch: List[dict] = []
        for arr in rows:
            if not isinstance(arr, list) or len(arr) < 9:
                continue
            confirm = arr[8]
            if str(confirm) != "1":
                continue

            ts = int(arr[0])
            o = safe_float(arr[1])
            h = safe_float(arr[2])
            l = safe_float(arr[3])
            c = safe_float(arr[4])
            vol_ccy_quote = safe_float(arr[7])  # quote turnover (best for USDT pairs)
            vol_ccy = safe_float(arr[6])

            parsed_batch.append(
                {
                    "ts": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    # prefer quote turnover for volume computations
                    "vol_quote": vol_ccy_quote if vol_ccy_quote is not None else vol_ccy,
                }
            )

        if not parsed_batch:
            break

        # OKX returns newest-first; make ascending and append
        parsed_batch.sort(key=lambda x: x["ts"])
        all_rows.extend(parsed_batch)

        # stop if enough
        all_rows = sorted({r["ts"]: r for r in all_rows}.values(), key=lambda x: x["ts"])  # de-dup by ts
        if len(all_rows) >= want:
            all_rows = all_rows[-want:]  # keep latest 'want' candles
            return all_rows, "OK", ""

        # prepare next page: 'after' should request older data (earlier timestamps)
        # Use the oldest ts in current response as cursor
        oldest_ts_in_response = min(int(arr[0]) for arr in rows if isinstance(arr, list) and len(arr) >= 1)
        after = oldest_ts_in_response  # move backward

        time.sleep(max(0.0, sleep_s))

    # return whatever we have (may be insufficient but not an error)
    all_rows = sorted({r["ts"]: r for r in all_rows}.values(), key=lambda x: x["ts"])
    return all_rows, "OK", ""


# ----------------------------
# Indicators (partial MA + 2-day valid + RVOL)
# ----------------------------
def last_two_rows(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if len(df) == 0:
        return None, None
    if len(df) == 1:
        return None, df.iloc[-1]
    return df.iloc[-2], df.iloc[-1]


def compute_indicators_partial(
    candles: List[dict],
    vol_lookback: int,
    rvol_threshold: float,
    require_vol_gt_yesterday: bool,
) -> Dict[str, Any]:
    df = pd.DataFrame(candles)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    out: Dict[str, Any] = {"bars_closed": int(len(df))}

    prev_row, last_row = last_two_rows(df)
    if last_row is None:
        out.update(
            {
                "close": None,
                "ma7": None,
                "ma30": None,
                "ma100": None,
                "above_ma7": None,
                "above_ma30": None,
                "above_ma100": None,
                "valid_above_ma7": None,
                "valid_above_ma30": None,
                "valid_above_ma100": None,
                "rvol": None,
                "volume_spike": None,
            }
        )
        return out

    # rolling MAs if enough history
    df["ma7"] = df["close"].rolling(7).mean() if len(df) >= 7 else pd.NA
    df["ma30"] = df["close"].rolling(30).mean() if len(df) >= 30 else pd.NA
    df["ma100"] = df["close"].rolling(100).mean() if len(df) >= 100 else pd.NA

    # refresh prev/last after MA columns exist
    prev_row, last_row = last_two_rows(df)

    close_last = safe_float(last_row["close"])
    ma7_last = safe_float(last_row["ma7"]) if "ma7" in last_row else None
    ma30_last = safe_float(last_row["ma30"]) if "ma30" in last_row else None
    ma100_last = safe_float(last_row["ma100"]) if "ma100" in last_row else None

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

    # valid above: last 2 closes > their day's MA
    def valid_above(prev: Optional[pd.Series], last: pd.Series, key_ma: str) -> Optional[bool]:
        if prev is None:
            return None
        c0 = safe_float(prev["close"])
        c1 = safe_float(last["close"])
        m0 = safe_float(prev.get(key_ma))
        m1 = safe_float(last.get(key_ma))
        if c0 is None or c1 is None or m0 is None or m1 is None:
            return None
        return bool((c0 > m0) and (c1 > m1))

    out["valid_above_ma7"] = valid_above(prev_row, last_row, "ma7")
    out["valid_above_ma30"] = valid_above(prev_row, last_row, "ma30")
    out["valid_above_ma100"] = valid_above(prev_row, last_row, "ma100")

    # RVOL on quote turnover (vol_quote)
    if "vol_quote" not in df.columns:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    dfv = df.dropna(subset=["vol_quote"]).reset_index(drop=True)
    if len(dfv) < vol_lookback + 1:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    vol_today = safe_float(dfv.iloc[-1]["vol_quote"])
    vol_avg = safe_float(dfv["vol_quote"].iloc[-(vol_lookback + 1) : -1].mean())
    if vol_today is None or vol_avg is None or vol_avg <= 0:
        out["rvol"] = None
        out["volume_spike"] = None
        return out

    rvol = vol_today / vol_avg
    out["rvol"] = float(rvol)

    spike = bool(rvol >= rvol_threshold)
    if require_vol_gt_yesterday and prev_row is not None:
        v_y = safe_float(prev_row.get("vol_quote"))
        if v_y is None:
            out["volume_spike"] = None
            return out
        spike = spike and (vol_today > v_y)

    out["volume_spike"] = bool(spike)
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="OKX Spot USDT MA+Volume scanner (Binance-like CLI)")
    ap.add_argument("--quote", default="USDT", help="Only scan quote currency (default USDT)")
    ap.add_argument("--top", type=int, default=0, help="Only scan top N by 24h quote volume; 0=all")
    ap.add_argument("--kline-limit", type=int, default=300, help="Daily candles to fetch per symbol (supports >300 via pagination)")
    ap.add_argument("--vol-lookback", type=int, default=20, help="RVOL lookback N (avg of prev N days)")
    ap.add_argument("--rvol", type=float, default=1.5, help="Volume spike threshold: RVOL >= this")
    ap.add_argument("--no-vol-gt-yesterday", action="store_true", help="Disable 'today volume > yesterday volume' filter")
    ap.add_argument("--sleep", type=float, default=0.12, help="Sleep between requests (seconds)")
    ap.add_argument("--out", default="okx_spot_usdt_scan.xlsx", help="Output Excel filename")
    args = ap.parse_args()

    client = OKXClient()

    instruments_map, _ = fetch_spot_instruments_usdt(client, args.quote)
    tickers_map, tickers_note = fetch_all_spot_tickers(client)

    inst_ids = list(instruments_map.keys())
    if not inst_ids:
        raise SystemExit(f"No OKX SPOT instruments found for quote={args.quote}")

    # build 24h quote volume map for sorting
    qv_map: Dict[str, float] = {inst_id: 0.0 for inst_id in inst_ids}
    for inst_id in inst_ids:
        t = tickers_map.get(inst_id, {})
        qv = safe_float(t.get("volCcy24h"))  # usually quote volume for spot tickers
        if qv is not None:
            qv_map[inst_id] = float(qv)

    inst_ids_sorted = sorted(inst_ids, key=lambda s: qv_map.get(s, 0.0), reverse=True)
    if args.top and args.top > 0:
        inst_ids_sorted = inst_ids_sorted[: args.top]

    rows: List[RowOut] = []

    global_note = tickers_note.strip()

    for inst_id in inst_ids_sorted:
        base = instruments_map[inst_id]["base"]
        quote = instruments_map[inst_id]["quote"]
        qv24 = qv_map.get(inst_id, 0.0)

        candles, st, note = fetch_closed_daily_candles_paginated(
            client=client,
            inst_id=inst_id,
            want=max(1, int(args.kline_limit)),
            sleep_s=max(0.0, float(args.sleep)),
            bar="1Dutc",
        )

        if st != "OK" or candles is None:
            # still output row (do not skip)
            full_note = (note or "").strip()
            if global_note:
                full_note = (full_note + " | " + global_note).strip(" |")
            rows.append(
                RowOut(
                    coin=base,
                    symbol=inst_id,
                    quote_asset=quote,
                    close=None,
                    ma7=None,
                    ma30=None,
                    ma100=None,
                    above_ma7=None,
                    above_ma30=None,
                    above_ma100=None,
                    valid_above_ma7=None,
                    valid_above_ma30=None,
                    valid_above_ma100=None,
                    rvol=None,
                    volume_spike=None,
                    quote_volume_24h=float(qv24),
                    bars_closed=0,
                    status=("RATE_LIMITED" if st == "RATE_LIMITED" else st),
                    note=full_note,
                )
            )
            time.sleep(max(0.0, float(args.sleep)))
            continue

        try:
            ind = compute_indicators_partial(
                candles=candles,
                vol_lookback=int(args.vol_lookback),
                rvol_threshold=float(args.rvol),
                require_vol_gt_yesterday=(not args.no_vol_gt_yesterday),
            )

            status = "OK"
            note2 = ""
            if ind.get("bars_closed", 0) < 2:
                status = "DATA_INSUFFICIENT"
                note2 = "Not enough closed daily candles"
            elif ind.get("ma7") is None and ind.get("ma30") is None and ind.get("ma100") is None:
                status = "DATA_INSUFFICIENT"
                note2 = "Not enough history for MA windows"

            if global_note:
                note2 = (note2 + " | " + global_note).strip(" |")

            rows.append(
                RowOut(
                    coin=base,
                    symbol=inst_id,
                    quote_asset=quote,
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
                    quote_volume_24h=float(qv24),
                    bars_closed=int(ind.get("bars_closed", 0)),
                    status=status,
                    note=note2,
                )
            )
        except Exception as e:
            full_note = f"compute failed: {e}"
            if global_note:
                full_note = (full_note + " | " + global_note).strip(" |")
            rows.append(
                RowOut(
                    coin=base,
                    symbol=inst_id,
                    quote_asset=quote,
                    close=None,
                    ma7=None,
                    ma30=None,
                    ma100=None,
                    above_ma7=None,
                    above_ma30=None,
                    above_ma100=None,
                    valid_above_ma7=None,
                    valid_above_ma30=None,
                    valid_above_ma100=None,
                    rvol=None,
                    volume_spike=None,
                    quote_volume_24h=float(qv24),
                    bars_closed=0,
                    status="PARSE_ERROR",
                    note=full_note,
                )
            )

        time.sleep(max(0.0, float(args.sleep)))

    df_out = pd.DataFrame([asdict(r) for r in rows])

    cols = [
        "coin",
        "symbol",
        "quote_asset",
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
