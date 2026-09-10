from datetime import date, datetime, timedelta, timezone

import pytest

from powernap.price import PricePoint, PriceService


class Response:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []
    def get(self, url, timeout):
        self.urls.append((url, timeout))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return Response(result)


def row(start, end, price):
    return {"time_start": start.isoformat(), "time_end": end.isoformat(), "SEK_per_kWh": price}


def service(session=None):
    return PriceService("SE3", "Europe/Stockholm", 5, "elpris_eu", "elprisetjustnu", 36, session)


def test_fetch_builds_documented_path_and_deduplicates():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    payload = [row(start, start + timedelta(minutes=15), 1.0)] * 2
    session = Session([payload])
    points = service(session).fetch_day(date(2026, 9, 10), "elpris_eu")
    assert len(points) == 1
    assert session.urls[0][0].endswith("/2026/09-10_SE3.json")


def test_fetch_rejects_overlap():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    payload = [row(start, start + timedelta(hours=1), 1), row(start + timedelta(minutes=30), start + timedelta(hours=2), 2)]
    with pytest.raises(ValueError, match="overlapping"):
        service(Session([payload])).fetch_day(start.date(), "elpris_eu")


def test_fallback_provider_is_used():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    s = service(Session([RuntimeError("primary down"), [row(start, start + timedelta(hours=1), 1)]]))
    s.ensure(start.date(), now_epoch=100)
    assert s.cache[start.date()][1][0].provider == "elprisetjustnu"


def test_stale_cache_survives_refresh_failure():
    day = date(2026, 9, 10)
    point = PricePoint(datetime(2026, 9, 10, tzinfo=timezone.utc), datetime(2026, 9, 10, 1, tzinfo=timezone.utc), 1, "cached")
    s = service(Session([RuntimeError("down"), RuntimeError("down")]))
    s.cache[day] = (0, [point])
    s.ensure(day, now_epoch=200000)
    assert s.cache[day][1] == [point]


def test_flat_prices_have_neutral_rank():
    assert PriceService._percentile_rank(1.0, [1.0] * 96) == 0.5


def test_context_crosses_midnight_without_assuming_interval_count(monkeypatch):
    s = service()
    now = datetime(2026, 9, 10, 23, 30, tzinfo=timezone(timedelta(hours=2)))
    starts = [now.replace(hour=23, minute=0), now.replace(hour=23, minute=30), now.replace(day=11, hour=0, minute=0), now.replace(day=11, hour=0, minute=30)]
    points = [PricePoint(x, x + timedelta(minutes=30), float(i + 1), "test") for i, x in enumerate(starts)]
    s.cache[now.date()] = (10**20, points[:2])
    s.cache[(now + timedelta(days=1)).date()] = (10**20, points[2:])
    monkeypatch.setattr(s, "ensure", lambda day: None)
    context = s.context(now, 2)
    assert context.current_sek_kwh == 2.0
    assert context.future_rank is not None
