# Contributing to PowerNap

Thank you for helping improve PowerNap.

## Before opening a change

- Search existing issues and pull requests.
- Keep the change focused on one clear problem.
- Do not add undocumented hardware writes.
- Do not assume that one vendor, driver, governor set, sensor layout, or sysfs topology represents every Linux system.

## Development setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,nvidia]'
pytest
```

## Requirements for hardware adapters

Every writable adapter must:

1. Discover support at runtime.
2. Read hardware or driver limits instead of guessing them.
3. Produce a complete dry-run plan.
4. Verify every applied value through read-back.
5. Report partial and failed application accurately.
6. Restore previous values where safe and supported.
7. Include fixture-based automated tests.
8. Document real-hardware validation separately.

## Pull requests

A pull request should include:

- A concise problem statement
- The behavior before and after the change
- Tests added or updated
- Hardware and driver details when relevant
- Dry-run output for new controls
- Any behavior that could not be verified

Do not describe hardware support as verified unless the final control path and restoration behavior were tested on relevant hardware.

## Documentation

Write documentation in natural, direct English. Keep visible labels in sentence case. Avoid promotional claims that the code or hardware path has not demonstrated.
