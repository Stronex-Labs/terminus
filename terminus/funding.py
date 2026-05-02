"""BTC perpetual funding rate fetcher for Terminus.

Pulls historical 8h funding rates from Binance futures and caches
them to Parquet under ~/.terminus/cache/funding/.

Usage
-----
    from terminus.funding import load_or_fetch_funding

    funding_series = await load_or_fetch_funding("BTCUSDT", days=365)
    # Returns pd.Series with timestamp index (UTC ms) and float values
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("terminus.funding")

_CACHE_DIR = Path.home() / ".terminus" / "cache" / "funding"
_BASE_URL = "https://fapi.binance.com"
_LIMIT_PER_REQ = 1000  # Binance max per call


def _cache_path(pair: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{pair}.parquet"


async def _fetch_funding_page(
    client: httpx.AsyncClient,
    pair: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[dict]:
    """Fetch one page of funding rate history."""
    params: dict = {"symbol": pair, "limit": _LIMIT_PER_REQ}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    resp = await client.get("/fapi/v1/fundingRate", params=params)
    resp.raise_for_status()
    return resp.json()


async def fetch_funding_history(pair: str, days: int = 365) -> pd.Series:
    """Fetch full funding rate history for a pair.

    Returns a pd.Series indexed by open_time (UTC milliseconds),
    values are float funding rates (e.g. 0.0001 = 0.01%/8h).
    """
    import time as _time

    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - int(days * 86_400 * 1000)

    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=_BASE_URL, timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            page = await _fetch_funding_page(client, pair, start_ms=cursor, end_ms=end_ms)
            if not page:
                break
            rows.extend(page)
            cursor = int(page[-1]["fundingTime"]) + 1
            if len(page) < _LIMIT_PER_REQ:
                break
            await asyncio.sleep(0.1)  # rate limit courtesy

    if not rows:
        logger.warning(f"No funding data returned for {pair}")
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_numeric(df["fundingTime"])
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    series = df.set_index("fundingTime")["fundingRate"].sort_index()
    series.index.name = "ts_ms"
    return series


async def load_or_fetch_funding(pair: str, days: int = 365) -> pd.Series:
    """Load funding from Parquet cache; fetch from Binance if stale/missing.

    Cache strategy: if cached file covers the required date range, return it.
    Otherwise fetch and overwrite the cache.
    """
    cache = _cache_path(pair)

    if cache.exists():
        try:
            cached = pd.read_parquet(cache)
            # Check if cache is recent enough (last entry within 9h — one funding period)
            import time as _time
            now_ms = int(_time.time() * 1000)
            required_start = now_ms - int(days * 86_400 * 1000)
            if (
                not cached.empty
                and int(cached.index[0]) <= required_start + 86_400_000  # 1-day tolerance
                and (now_ms - int(cached.index[-1])) < 9 * 3_600_000     # fresh enough
            ):
                logger.debug(f"Funding cache hit for {pair} ({len(cached)} rows)")
                return cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        except Exception as e:
            logger.warning(f"Funding cache read failed for {pair}: {e}")

    logger.info(f"Fetching {days}d funding history for {pair}...")
    series = await fetch_funding_history(pair, days=days)
    if not series.empty:
        series.to_frame("funding_rate").to_parquet(cache)
        logger.info(f"Cached {len(series)} funding rows for {pair}")
    return series


def attach_funding(df: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Forward-fill funding rate onto a kline DataFrame.

    Args:
        df:      kline DataFrame with open_time column (UTC ms integers)
        funding: funding rate Series indexed by ts_ms

    Returns:
        df with new column 'funding_8h' (float, forward-filled)
    """
    if funding.empty:
        df["funding_8h"] = 0.0
        return df

    merged = pd.merge_asof(
        df.sort_values("open_time"),
        funding.rename("funding_8h").reset_index().rename(columns={"ts_ms": "open_time"}),
        on="open_time",
        direction="backward",
    )
    merged["funding_8h"] = merged["funding_8h"].fillna(0.0)
    return merged
