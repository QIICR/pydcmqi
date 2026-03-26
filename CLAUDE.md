# CLAUDE.md

## Project

pydcmqi — Python API wrapper for dcmqi (DICOM quantitative imaging) CLI tools.

## Development

- Python 3.10+
- Uses `hatchling` + `hatch-vcs` for build/versioning
- Install for development: `.venv/bin/pip install -e ".[test]"`
- Pre-commit hooks configured (ruff, mypy, prettier, codespell, etc.)

## Before committing

Always run pre-commit hooks before committing:

```bash
.venv/bin/pre-commit run --all-files
```

Fix any failures before creating the commit. Do not skip hooks with `--no-verify`.

## Testing

```bash
.venv/bin/pytest tests/
```

Integration tests (TestSegimageRead, TestSegimageWrite) require network access to download DICOM data from IDC.

## Linting

```bash
.venv/bin/pre-commit run --all-files   # all hooks including mypy
.venv/bin/ruff check src/ tests/       # ruff only
.venv/bin/mypy src/ tests/             # mypy only
```

## Project structure

```
src/pydcmqi/
├── __init__.py      # Public API exports
├── types.py         # TypedDicts, Literal types
├── exceptions.py    # DcmqiError
├── triplet.py       # Triplet class (DICOM code triplet)
├── segment.py       # SegmentData, Segment
├── segimage.py      # SegImageData, SegImageFiles, SegImage (orchestrator)
```
