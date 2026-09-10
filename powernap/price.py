from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

import requests

from .model import PriceContext


@dataclass(frozen=True)
class PricePoint:
    start: datetime
    end: datetime
    sek_kwh: float
    provider: str


BASES = {
    "elpris_eu": "https://se.elpris.eu/api/v1/prices",
    "elprisetjustnu": "https://www.elprisetjustnu.se/api/v1/prices",
}


class PriceService:
    def __init__(
        self,
        area: str,
        timezone: str,
        timeout: float,
        provider: str,
        fallback: str | None = None,
        cache_hours: float = 36.0,
        session: requests.Session | None = None,
    ):
        if provider not in BASES:
            raise ValueError(f"Unknown price provider: {provider}")
        if fallback and fallback not in BASES:
            raise ValueError(f"Unknown fallback price provider: {fallback}")
        self.area = area
        self.tz = ZoneInfo(timezone)
        self.timeout = timeout
        self.provider = provider
        self.fallback = fallback if fallback != provider else None
        self.cache_hours = cache_hours
        self.session = session or requests.Session()
        self.cache: dict[date, tuple[float, list[PricePoint]]] = {}

    def fetch_day(self, day: date, provider: str) -> list[PricePoint]:
        url = f"{BASES[provider]}/{day:%Y/%m-%d}_{self.area}.json"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Price response is not a list")
        points = []
        rejected = 0
        for item in payload:
            try:
                start = datetime.fromisoformat(str(item["time_start"]).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(item["time_end"]).replace("Z", "+00:00"))
                price = float(item["SEK_per_kWh"])
                if start.tzinfo is None or end.tzinfo is None or end <= start:
                    raise ValueError("invalid interval")
                points.append(PricePoint(start, end, price, provider))
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected += 1
        points.sort(key=lambda point: point.start)
        unique = []
        seen = set()
        for point in points:
            key = (point.start, point.end)
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        for previous, current in zip(unique, unique[1:]):
            if current.start < previous.end:
                raise ValueError("Price response contains overlapping intervals")
        if not unique:
            raise ValueError("Price response contained no valid intervals")
        if rejected:
            logging.warning("Rejected %d invalid price intervals from %s", rejected, provider)
        return unique

    def ensure(self, day: date, now_epoch: float | None = None) -> None:
        now_epoch = time.time() if now_epoch is None else now_epoch
        cached = self.cache.get(day)
        if cached and now_epoch - cached[0] <= self.cache_hours * 3600:
            return
        errors = []
        for provider in (self.provider, self.fallback):
            if not provider:
                continue
            try:
                self.cache[day] = (now_epoch, self.fetch_day(day, provider))
                return
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
        if cached:
            logging.warning("Price refresh failed; using stale cached data for %s: %s", day, "; ".join(errors))
            return
        raise RuntimeError("; ".join(errors))

    @staticmethod
    def _percentile_rank(value: float, values: list[float]) -> float:
        if len(values) <= 1 or max(values) == min(values):
            return 0.5
        lower = sum(item < value for item in values)
        equal = sum(item == value for item in values)
        return max(0.0, min(1.0, (lower + max(0, equal - 1) / 2) / (len(values) - 1)))

    def context(self, now: datetime | None = None, lookahead_hours: int = 3) -> PriceContext:
        local_now = (now or datetime.now(self.tz)).astimezone(self.tz)
        end = local_now + timedelta(hours=lookahead_hours)
        required_days = {local_now.date(), end.date()}
        for day in required_days:
            try:
                self.ensure(day)
            except RuntimeError as exc:
                logging.warning("Price data unavailable for %s: %s", day, exc)
        points = sorted(
            (point for day in required_days for point in self.cache.get(day, (0, []))[1]),
            key=lambda point: point.start,
        )
        current = next((point for point in points if point.start <= local_now < point.end), None)
        day_points = [point for point in points if point.start.astimezone(self.tz).date() == local_now.date()]
        if current is None or not day_points:
            return PriceContext()
        values = [point.sek_kwh for point in day_points]
        rank = self._percentile_rank(current.sek_kwh, values)
        future = [point.sek_kwh for point in points if local_now < point.start < end]
        future_rank = self._percentile_rank(mean(future), values) if future else None
        return PriceContext(
            current_sek_kwh=current.sek_kwh,
            rank=rank,
            future_rank=future_rank,
            trend=None if future_rank is None else future_rank - rank,
            fresh=time.time() - self.cache[local_now.date()][0] <= self.cache_hours * 3600,
            provider=current.provider,
        )
