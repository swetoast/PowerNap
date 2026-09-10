from powernap.database import Repository
from powernap.model import CPUState,Decision,PriceContext,Profile,SystemState,ThermalState

def test_database_cycle(tmp_path):
    r=Repository(tmp_path/"x.db");s=SystemState("now",1,CPUState(1,2,0,0,0,0,30,"cpu"),(),PriceContext());d=Decision(1,0,Profile.ECO,Profile.BALANCED,Profile.MAXIMUM,Profile.BALANCED,ThermalState.NORMAL,"ok");r.record_cycle(s,d,[]);assert r.report()["decisions"][0]["reason"]=="ok";r.close()


def test_multiple_cycles_in_same_millisecond_are_preserved(tmp_path, monkeypatch):
    from powernap import database
    r = Repository(tmp_path / "x.db")
    monkeypatch.setattr(database.time, "time_ns", lambda: 1_000_000)
    s = SystemState("now", 1, CPUState(1,2,0,0,0,0,30,"cpu"),(),PriceContext())
    d = Decision(1,0,Profile.ECO,Profile.BALANCED,Profile.MAXIMUM,Profile.BALANCED,ThermalState.NORMAL,"ok")
    r.record_cycle(s,d,[]); r.record_cycle(s,d,[])
    assert len(r.report()["decisions"]) == 2
    r.close()


def test_legacy_090_schema_is_migrated(tmp_path):
    import sqlite3
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript('''
      CREATE TABLE samples(ts INTEGER PRIMARY KEY,timestamp TEXT,payload TEXT);
      CREATE TABLE decisions(ts INTEGER PRIMARY KEY,recommended TEXT,reason TEXT,payload TEXT);
      CREATE TABLE controls(id INTEGER PRIMARY KEY,ts INTEGER,adapter TEXT,target TEXT,result TEXT,payload TEXT);
      INSERT INTO samples VALUES(1,'now','{}');
      INSERT INTO decisions VALUES(1,'balanced','legacy','{}');
    ''')
    conn.commit(); conn.close()
    r = Repository(path)
    assert r.report()["decisions"][0]["reason"] == "legacy"
    columns = {row[1] for row in r.conn.execute("PRAGMA table_info(samples)")}
    assert {"id", "ts_ms"}.issubset(columns)
    r.close()
