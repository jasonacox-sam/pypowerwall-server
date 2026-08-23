"""
Time-Series API Endpoints

REST API for daily energy statistics and raw sample access backed by the
TimeSeriesStore (SQLite, see app/core/timeseries.py).
All routes are prefixed with /api/timeseries (configured in main.py).

Routes:
    - GET /api/timeseries/daily   -> Daily kWh totals per gateway/category
    - GET /api/timeseries/today   -> Today's running totals for all gateways
    - GET /api/timeseries/trend   -> Bucketed kW + battery level (charting)
    - GET /api/timeseries/samples -> Raw samples (troubleshooting)
    - GET /api/timeseries/status  -> Subsystem status and DB sizing

Query Parameters:
    daily:   days (int, default 7)  — number of days back from today
             gateway (str, optional) — restrict to a single gateway ID
    today:   none — returns every configured gateway's running totals
    trend:   hours (int, default 24, max 168) — window length; gateway
             (str, optional) restricts to one gateway. start (unix ts,
             optional) overrides hours with an explicit window start;
             end (unix ts, optional) explicit window end (default now);
             fit (bool, optional, default false) uses all retained raw
             data. Returns ~360 bucket-averaged points: solar/home/
             battery/grid kW plus mean battery level (%) per bucket.
    samples: gateway (str, optional), start (unix ts), end (unix ts),
             limit (int, default 500, max 10000)

All endpoints respond with {"enabled": false, ...} when the subsystem is
disabled (PW_TIMESERIES_RETENTION=-1) so clients can hide the UI panel
instead of erroring.

Design Notes:
    - All reads go through the store's single worker thread — they never
      block or contend with the poll loop's writes (WAL mode).
    - Endpoints return immediately; missing data yields empty lists, not
      errors, matching the degraded-gracefully style of the other APIs.
"""

import time
from typing import Optional

from fastapi import APIRouter, Query

from app.core.gateway_manager import gateway_manager
from app.core.timeseries import get_timeseries_store

router = APIRouter()


@router.get("/daily")
async def get_daily_energy(
    days: int = Query(default=7, ge=1, le=366),
    gateway: Optional[str] = Query(default=None),
):
    """Daily energy totals (kWh) per gateway, most recent day first.

    Each day entry contains one row per gateway that reported samples that
    day, with directional kWh categories (solar, home, battery charge/
    discharge, grid import/export).
    """
    return await get_timeseries_store().get_daily_energy(days=days, gateway=gateway)


@router.get("/today")
async def get_today():
    """Today's running kWh totals for every configured gateway.

    Uses each gateway's own timezone to determine 'today'. Gateways without
    samples yet are omitted. Handy for the UI panel and quick checks:

        {"enabled": true, "gateways": {"gw1": {...kWh...}}}
    """
    store = get_timeseries_store()
    if not store.enabled:
        return {"enabled": False, "gateways": {}}
    out = {}
    for gateway_id, gateway in gateway_manager.gateways.items():
        row = await store.get_today(gateway_id, timezone=gateway.timezone)
        if row is not None:
            out[gateway_id] = row
    return {
        "enabled": True,
        "gateways": out,
        "server_time": time.time(),
    }


@router.get("/trend")
async def get_trend(
    hours: int = Query(default=24, ge=1, le=168),
    gateway: Optional[str] = Query(default=None),
    start: Optional[float] = Query(default=None),
    end: Optional[float] = Query(default=None),
    fit: bool = Query(default=False),
):
    """Bucketed time series of power (kW) and battery level (%) for charts.

    Averages raw samples into ~4-minute buckets over the requested window
    so a 24h view is ~360 points rather than ~17k rows. Battery kW is
    positive = discharging, grid kW positive = importing; ``battery_level``
    is mean state of charge per bucket (raw-sample only, never persisted
    into daily aggregates).
    """
    return await get_timeseries_store().get_trend(
        hours=hours, gateway=gateway, start=start, end=end, fit=fit
    )


@router.get("/samples")
async def get_samples(
    gateway: Optional[str] = Query(default=None),
    start: Optional[float] = Query(default=None),
    end: Optional[float] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=10000),
):
    """Raw power samples, ascending by time (troubleshooting)."""
    return await get_timeseries_store().get_samples(
        gateway=gateway, start=start, end=end, limit=limit
    )


@router.get("/status")
async def get_status():
    """Time-series subsystem status: retention settings, DB size, row counts."""
    return await get_timeseries_store().status()
