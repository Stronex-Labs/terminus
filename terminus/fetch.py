"""Kline fetcher — Binance-first, pluggable.

Each fetcher returns a DataFrame with columns: open_time (int ms), open,
high, low, close, volume. All prices float. Open time is the candle open
in UTC milliseconds.

Built-in:
    BinanceFetcher — spot klines via public API, no auth required for
                     historical data

Pluggable: instantiate any object with the method signature
    async def fetch(pair, interval, days) -> DataFrame | None
and pass it to the cache layer.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd


logger = logging.getLogger("terminus.fetch")


TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


class Fetcher(Protocol):
    async def fetch(self, pair: str, interval: str, days: int
                    ) -> pd.DataFrame | None: ...


class BinanceFetcher:
    """Spot klines via https://api.binance.com/api/v3/klines (no auth)."""

    def __init__(self, *, timeout: float = 30.0,
                 retries: int = 4, initial_backoff: float = 5.0):
        self._timeout = timeout
        self._retries = retries
        self._initial_backoff = initial_backoff
        self._client = httpx.AsyncClient(
            base_url="https://api.binance.com",
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def _get_chunk(self, pair: str, interval: str,
                         start_ms: int, end_ms: int) -> list:
        params = {
            "symbol": pair.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        delay = self._initial_backoff
        for attempt in range(self._retries):
            try:
                r = await self._client.get("/api/v3/klines", params=params)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning("fetch attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
        return []

    async def fetch(self, pair: str, interval: str, days: int
                    ) -> pd.DataFrame | None:
        if interval not in TF_MS:
            raise ValueError(f"Unknown interval: {interval}")
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - days * 86_400_000
        step_ms = TF_MS[interval] * 1000  # 1000 bars per request

        all_rows = []
        cursor = start_ms
        while cursor < end_ms:
            chunk = await self._get_chunk(pair, interval, cursor,
                                           min(cursor + step_ms, end_ms))
            if not chunk:
                break
            all_rows.extend(chunk)
            last_open = int(chunk[-1][0])
            # Advance to 1ms past last bar so we don't re-fetch it
            cursor = last_open + TF_MS[interval]
            if len(chunk) < 1000:
                # Done — exchange returned partial batch
                break

        if not all_rows:
            return None

        df = pd.DataFrame({
            "open_time": [int(k[0]) for k in all_rows],
            "open":   [float(k[1]) for k in all_rows],
            "high":   [float(k[2]) for k in all_rows],
            "low":    [float(k[3]) for k in all_rows],
            "close":  [float(k[4]) for k in all_rows],
            "volume": [float(k[5]) for k in all_rows],
        }).drop_duplicates(subset="open_time").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df


# --- CSV cache -----------------------------------------------------------

_CSV_DTYPES = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}


def cache_path(pair: str, tf: str, days: int,
               cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = Path.home() / ".terminus" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{pair}_{tf}_{days}d.csv"


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=_CSV_DTYPES)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def is_fresh(path: Path, tf: str) -> bool:
    """Fresh if last-row timestamp is within 2 bar intervals of now."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=["open_time"], dtype={"open_time": "int64"})
    except Exception:
        return False
    if df.empty:
        return False
    last_ms = int(df["open_time"].iloc[-1])
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    age_ms = now_ms - last_ms
    return age_ms < 2 * TF_MS.get(tf, TF_MS["1h"])


async def load_or_fetch(
    fetcher: Fetcher, pair: str, tf: str, days: int,
    *, cache_dir: Path | None = None, force: bool = False,
) -> pd.DataFrame | None:
    path = cache_path(pair, tf, days, cache_dir)
    if not force and is_fresh(path, tf):
        try:
            return _read_cache(path)
        except Exception as e:
            logger.warning("cache read failed for %s, refetching: %s", path, e)

    df = await fetcher.fetch(pair, tf, days)
    if df is None or len(df) < 100:
        return df
    try:
        df.drop(columns=["ts"], errors="ignore").to_csv(path, index=False)
    except Exception as e:
        logger.warning("cache write failed for %s: %s", path, e)
    return df
