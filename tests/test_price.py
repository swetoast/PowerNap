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


def test_non_finite_prices_are_rejected():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="no valid intervals"):
        service(Session([[row(start, start + timedelta(hours=1), "nan")]])).fetch_day(start.date(), "elpris_eu")


def test_persisted_prices_are_loaded_on_restart(tmp_path, monkeypatch):
    from powernap.database import Repository
    repo = Repository(tmp_path / "prices.db")
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    repo.store_prices("SE3", [PricePoint(start, start + timedelta(hours=1), 1.5, "cached")])
    s = PriceService("SE3", "Europe/Stockholm", 5, "elpris_eu", "elprisetjustnu", 36, repository=repo)
    assert any(point.sek_kwh == 1.5 for _, points in s.cache.values() for point in points)
    repo.close()


def test_price_context_reports_fresh_cache_age_and_quality(monkeypatch):
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    svc = service(Session([[row(start, start + timedelta(hours=1), 1.0)]]))
    monkeypatch.setattr("powernap.price.time.time", lambda: 1000.0)
    svc.cache[start.date()] = (900.0, [PricePoint(start, start + timedelta(hours=1), 1.0, "test")])
    context = svc.context(now=start + timedelta(minutes=30), lookahead_hours=1)
    assert context.fresh is False
    assert context.quality == "incomplete"
    assert context.cache_age_seconds == 100.0
    assert context.complete is False


def test_price_context_reports_stale_cache_explicitly(monkeypatch):
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    svc = service(Session([]))
    svc.cache_hours = 1
    svc.cache[start.date()] = (0.0, [PricePoint(start, start + timedelta(hours=1), 1.0, "persisted")])
    monkeypatch.setattr("powernap.price.time.time", lambda: 7200.0)
    context = svc.context(now=start + timedelta(minutes=30), lookahead_hours=1)
    assert context.fresh is False
    assert context.quality == "stale"
    assert context.cache_age_seconds == 7200.0


def test_context_fetches_every_intermediate_date(monkeypatch):
    s = service()
    now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    called = []
    monkeypatch.setattr(s, "ensure", lambda day: called.append(day))
    s.context(now, 60)
    assert called == [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 12), date(2026, 9, 13)]


def test_context_does_not_mix_providers(monkeypatch):
    s = service()
    now = datetime(2026, 9, 10, 0, 30, tzinfo=timezone.utc)
    primary = PricePoint(now.replace(minute=0), now.replace(minute=0)+timedelta(hours=1), 1.0, "primary")
    fallback = PricePoint(now.replace(minute=0)+timedelta(hours=1), now.replace(minute=0)+timedelta(hours=2), 9.0, "fallback")
    s.cache[now.date()] = (10**20, [primary, fallback])
    monkeypatch.setattr(s, "ensure", lambda day: None)
    context = s.context(now, 2)
    assert context.provider == "primary"
    assert context.future_rank is None
    assert context.complete is False
    assert context.gap_count > 0


def test_context_duration_weights_future_intervals(monkeypatch):
    s = service()
    now = datetime(2026, 9, 10, 0, 15, tzinfo=timezone.utc)
    points = [
        PricePoint(now.replace(minute=0), now.replace(minute=0)+timedelta(hours=1), 1.0, "p"),
        PricePoint(now.replace(minute=0)+timedelta(hours=1), now.replace(minute=0)+timedelta(hours=2), 4.0, "p"),
        PricePoint(now.replace(minute=0)+timedelta(hours=2), now.replace(minute=0)+timedelta(hours=2, minutes=15), 16.0, "p"),
    ]
    s.cache[now.date()] = (10**20, points)
    monkeypatch.setattr(s, "ensure", lambda day: None)
    context = s.context(now, 2)
    assert context.future_rank is not None
    assert context.coverage_ratio == 1.0
    assert context.complete is True


def test_context_marks_gap_as_incomplete(monkeypatch):
    s = service()
    now = datetime(2026, 9, 10, 0, 15, tzinfo=timezone.utc)
    points = [PricePoint(now.replace(minute=0), now.replace(minute=0)+timedelta(minutes=30), 1.0, "p")]
    s.cache[now.date()] = (10**20, points)
    monkeypatch.setattr(s, "ensure", lambda day: None)
    context = s.context(now, 1)
    assert context.quality == "incomplete"
    assert context.fresh is False
    assert context.coverage_ratio < 1.0


def local_day_points(service, day, minutes, provider="test"):
    start, end = service._day_bounds(day)
    points = []
    cursor = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    step = timedelta(minutes=minutes)
    while cursor < end_utc:
        points.append(PricePoint(cursor, min(cursor + step, end_utc), float(len(points)), provider))
        cursor += step
    return points


def test_spring_dst_day_has_23_hourly_intervals_and_is_complete(monkeypatch):
    s = service()
    day = date(2026, 3, 29)
    points = local_day_points(s, day, 60)
    assert len(points) == 23
    s.cache[day] = (10**20, points)
    monkeypatch.setattr(s, "ensure", lambda requested: None)
    context = s.context(datetime(2026, 3, 29, 12, tzinfo=timezone.utc), 1)
    assert context.current_day_complete is True
    assert context.current_day_coverage_ratio == 1.0
    assert context.expected_day_seconds == 23 * 3600


def test_autumn_dst_day_has_25_hourly_intervals_and_is_complete(monkeypatch):
    s = service()
    day = date(2026, 10, 25)
    points = local_day_points(s, day, 60)
    assert len(points) == 25
    s.cache[day] = (10**20, points)
    monkeypatch.setattr(s, "ensure", lambda requested: None)
    context = s.context(datetime(2026, 10, 25, 12, tzinfo=timezone.utc), 1)
    assert context.current_day_complete is True
    assert context.expected_day_seconds == 25 * 3600


def test_spring_dst_quarter_hour_day_has_92_intervals():
    s = service()
    assert len(local_day_points(s, date(2026, 3, 29), 15)) == 92


def test_autumn_dst_quarter_hour_day_has_100_intervals():
    s = service()
    assert len(local_day_points(s, date(2026, 10, 25), 15)) == 100


def test_incomplete_day_is_reported_even_when_current_lookahead_is_covered(monkeypatch):
    s = service()
    day = date(2026, 9, 10)
    start, _ = s._day_bounds(day)
    points = [PricePoint(start + timedelta(hours=12), start + timedelta(hours=14), 1.0, "test")]
    s.cache[day] = (10**20, points)
    monkeypatch.setattr(s, "ensure", lambda requested: None)
    context = s.context(start + timedelta(hours=12, minutes=30), 1)
    assert context.complete is True
    assert context.current_day_complete is False
    assert context.current_day_coverage_ratio < 1.0
    assert context.current_day_gap_count > 0


def test_fetch_accepts_negative_zero_and_extreme_finite_prices():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    payload = [
        row(start, start + timedelta(hours=1), -1.0),
        row(start + timedelta(hours=1), start + timedelta(hours=2), 0.0),
        row(start + timedelta(hours=2), start + timedelta(hours=3), 1000000.0),
    ]
    points = service(Session([payload])).fetch_day(start.date(), "elpris_eu")
    assert [point.sek_kwh for point in points] == [-1.0, 0.0, 1000000.0]


def test_fetch_sorts_out_of_order_intervals():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    payload = [
        row(start + timedelta(hours=1), start + timedelta(hours=2), 2.0),
        row(start, start + timedelta(hours=1), 1.0),
    ]
    points = service(Session([payload])).fetch_day(start.date(), "elpris_eu")
    assert [point.sek_kwh for point in points] == [1.0, 2.0]


def test_owned_http_session_is_closed_once(monkeypatch):
    closed = []
    class OwnedSession:
        def close(self): closed.append(True)
    monkeypatch.setattr("powernap.price.requests.Session", OwnedSession)
    s = service()
    s.close()
    s.close()
    assert closed == [True]


def test_injected_http_session_is_not_closed():
    class BorrowedSession:
        def __init__(self): self.closed = False
        def close(self): self.closed = True
    session = BorrowedSession()
    s = service(session)
    s.close()
    assert session.closed is False
