from __future__ import annotations

import psutil


def protected_active(names: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    wanted = {item.casefold() for item in names}
    found = set()
    if not wanted:
        return False, ()
    for process in psutil.process_iter(["name", "exe", "cmdline"]):
        try:
            values = [process.info.get("name") or "", process.info.get("exe") or ""]
            values.extend(process.info.get("cmdline") or [])
            for value in values:
                basename = value.rsplit("/", 1)[-1].casefold()
                if basename in wanted:
                    found.add(basename)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return bool(found), tuple(sorted(found))
