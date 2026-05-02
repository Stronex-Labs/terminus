"""Fear & Greed Index fetcher for Terminus.

Pulls historical daily Fear & Greed values from alternative.me and caches
to Parquet under ~/.terminus/cache/sentiment/.

Values: 0 = Extreme Fear, 100 = Extreme Greed.

Usage
-----
    from terminus.sentiment import load_or_fetch_fng

    fng_series = await load_or_fetch_fng(days=365)
    # Returns pd.Series indexed by date (UTC midnight ms), values 0-100
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("terminus.sentiment")

_CACHE_DIR = Path.home() / ".terminus" / "cache" / "sentiment"
_FNG_URL = "https://api.alternative.me/fng/"
_CACHE_FILE = _CACHE_DIR / "fng.parquet"


async def fetch_fng(days: int = 365) -> pd.Series:
    """Fetch Fear & Greed historical data from alternative.me.

    Returns pd.Series indexed by UTC midnight timestamp (ms), values 0-100.
    Older than 2018 not available.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(_FNG_URL, params={"limit": max(days + 30, 2000)})
        resp.raise_for_status()
        data = resp.json().get("data", [])

    if not data:
        logger.warning("Fear & Greed API returned empty data")
        return pd.Series(dtype=float)

    records = [
        {"ts_ms": int(row["timestamp"]) * 1000, "fng": int(row["value"])}
        for row in data
    ]
    df = pd.DataFrame(records).sort_values("ts_ms")
    series = df.set_index("ts_ms")["fng"]
    series.index.name = "ts_ms"
    return series


async def load_or_fetch_fng(days: int = 365) -> pd.Series:
    """Load F&G from Parquet cache; refresh if older than 24h."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _CACHE_FILE.exists():
        try:
            cached = pd.read_parquet(_CACHE_FILE)
            import time as _time
            now_ms = int(_time.time() * 1000)
            if (
                not cached.empty
                and (now_ms - int(cached.index[-1])) < 25 * 3_600_000  # <25h old
            ):
                logger.debug(f"F&G cache hit ({len(cached)} rows)")
                s = cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
                return s
        except Exception as e:
            logger.warning(f"F&G cache read failed: {e}")

    logger.info(f"Fetching {days}d Fear & Greed history...")
    series = await fetch_fng(days=days)
    if not series.empty:
        series.to_frame("fng").to_parquet(_CACHE_FILE)
        logger.info(f"Cached {len(series)} F&G rows")
    return series


def attach_fear_greed(df: pd.DataFrame, fng: pd.Series) -> pd.DataFrame:
    """Forward-fill F&G index onto a kline DataFrame.

    Args:
        df:  kline DataFrame with open_time column (UTC ms integers)
        fng: F&G Series indexed by ts_ms (daily, UTC midnight)

    Returns:
        df with new column 'fng' (int 0-100, forward-filled)
    """
    if fng.empty:
        df["fng"] = 50  # neutral fallback
        return df

    merged = pd.merge_asof(
        df.sort_values("open_time"),
        fng.rename("fng").reset_index().rename(columns={"ts_ms": "open_time"}),
        on="open_time",
        direction="backward",
    )
    merged["fng"] = merged["fng"].fillna(50).astype(int)
    return merged
