from __future__ import annotations

import logging
import math
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
        repository=None,
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
        self._owns_session = session is None
        self.session = session or requests.Session()
        self.repository = repository
        self.cache: dict[date, tuple[float, list[PricePoint]]] = {}
        self._load_persisted()

    def close(self) -> None:
        if self._owns_session:
            self.session.close()
            self._owns_session = False

    def _load_persisted(self) -> None:
        if self.repository is None:
            return
        try:
            rows = self.repository.load_prices(self.area)
        except Exception as exc:
            logging.warning("Unable to load persisted prices: %s", exc)
            return
        grouped: dict[date, list[PricePoint]] = {}
        fetched: dict[date, float] = {}
        for row in rows:
            try:
                start = datetime.fromisoformat(row["start"])
                end = datetime.fromisoformat(row["end"])
                point = PricePoint(start, end, float(row["sek_kwh"]), row["provider"])
                day = start.astimezone(self.tz).date()
                grouped.setdefault(day, []).append(point)
                fetched[day] = max(fetched.get(day, 0), row["fetched_ms"] / 1000.0)
            except (KeyError, TypeError, ValueError):
                continue
        for day, points in grouped.items():
            self.cache[day] = (fetched[day], sorted(points, key=lambda point: point.start))

    def _persist(self, points: list[PricePoint]) -> None:
        if self.repository is None:
            return
        try:
            self.repository.store_prices(self.area, points)
        except Exception as exc:
            logging.warning("Unable to persist prices: %s", exc)

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
                if not math.isfinite(price):
                    raise ValueError("non-finite price")
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
                points = self.fetch_day(day, provider)
                self.cache[day] = (now_epoch, points)
                self._persist(points)
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

    def _day_bounds(self, day: date) -> tuple[datetime, datetime]:
        start = datetime.combine(day, datetime.min.time(), self.tz)
        end = datetime.combine(day + timedelta(days=1), datetime.min.time(), self.tz)
        return start, end

    @staticmethod
    def _coverage(points: list[PricePoint], start: datetime, end: datetime) -> tuple[float, int]:
        utc = ZoneInfo("UTC")
        start_utc = start.astimezone(utc)
        end_utc = end.astimezone(utc)
        relevant = sorted(
            (point for point in points if point.end.astimezone(utc) > start_utc and point.start.astimezone(utc) < end_utc),
            key=lambda point: point.start.astimezone(utc),
        )
        cursor = start_utc
        covered = 0.0
        gaps = 0
        for point in relevant:
            clipped_start = max(start_utc, point.start.astimezone(utc))
            clipped_end = min(end_utc, point.end.astimezone(utc))
            if clipped_start > cursor:
                gaps += 1
            effective_start = max(cursor, clipped_start)
            if clipped_end > effective_start:
                covered += (clipped_end - effective_start).total_seconds()
                cursor = clipped_end
        if cursor < end_utc:
            gaps += 1
        total = max(1.0, (end_utc - start_utc).total_seconds())
        return min(1.0, covered / total), gaps

    @staticmethod
    def _duration_weighted_mean(points: list[PricePoint], start: datetime, end: datetime) -> float | None:
        weighted = duration = 0.0
        for point in points:
            overlap = max(0.0, (min(end, point.end) - max(start, point.start)).total_seconds())
            if overlap:
                weighted += point.sek_kwh * overlap
                duration += overlap
        return weighted / duration if duration else None

    def context(self, now: datetime | None = None, lookahead_hours: int = 3) -> PriceContext:
        local_now = (now or datetime.now(self.tz)).astimezone(self.tz)
        end = local_now + timedelta(hours=lookahead_hours)
        days = []
        day = local_now.date()
        while day <= end.date():
            days.append(day)
            day += timedelta(days=1)
        for required_day in days:
            try:
                self.ensure(required_day)
            except RuntimeError as exc:
                logging.warning("Price data unavailable for %s: %s", required_day, exc)
        all_points = sorted(
            (point for required_day in days for point in self.cache.get(required_day, (0, []))[1]),
            key=lambda point: point.start,
        )
        current = next((point for point in all_points if point.start <= local_now < point.end), None)
        if current is None:
            return PriceContext()
        provider = current.provider
        points = [point for point in all_points if point.provider == provider]
        day_start, day_end = self._day_bounds(local_now.date())
        day_points = [point for point in points if point.end > day_start and point.start < day_end]
        if not day_points:
            return PriceContext()
        values = [point.sek_kwh for point in day_points]
        rank = self._percentile_rank(current.sek_kwh, values)
        future_start = current.end
        future_mean = self._duration_weighted_mean(points, future_start, end)
        future_rank = self._percentile_rank(future_mean, values) if future_mean is not None else None
        coverage_ratio, gap_count = self._coverage(points, local_now, end)
        day_coverage_ratio, day_gap_count = self._coverage(day_points, day_start, day_end)
        current_day_complete = day_coverage_ratio >= 0.999 and day_gap_count == 0
        fetched = [self.cache[required_day][0] for required_day in days if required_day in self.cache]
        cache_age = max(0.0, time.time() - min(fetched)) if fetched else None
        fresh = cache_age is not None and cache_age <= self.cache_hours * 3600
        complete = coverage_ratio >= 0.999 and gap_count == 0
        quality = "fresh" if fresh and complete else "incomplete" if fresh else "stale"
        return PriceContext(
            current_sek_kwh=current.sek_kwh,
            rank=rank,
            future_rank=future_rank,
            trend=None if future_rank is None else future_rank - rank,
            fresh=fresh and complete,
            provider=provider,
            cache_age_seconds=cache_age,
            quality=quality,
            complete=complete,
            coverage_ratio=coverage_ratio,
            gap_count=gap_count,
            current_day_complete=current_day_complete,
            current_day_coverage_ratio=day_coverage_ratio,
            current_day_gap_count=day_gap_count,
            expected_day_seconds=int((day_end.astimezone(ZoneInfo("UTC")) - day_start.astimezone(ZoneInfo("UTC"))).total_seconds()),
        )

