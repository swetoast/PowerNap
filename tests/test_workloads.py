import psutil
from powernap.workloads import protected_active

class Process:
    def __init__(self, info): self.info = info

def test_protected_process_matches_executable_basename(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [Process({"name":"python", "exe":"/usr/bin/ffmpeg", "cmdline":[]})])
    assert protected_active(("ffmpeg",)) == (True, ("ffmpeg",))

def test_no_rules_avoids_process_scan(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: (_ for _ in ()).throw(AssertionError()))
    assert protected_active(()) == (False, ())
