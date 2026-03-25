# Plan: Make pydcmqi Robust and Usable

## Context

pydcmqi is a Python wrapper around the dcmqi C++ CLI tools (`segimage2itkimage`, `itkimage2segimage`) for reading/writing DICOM Segmentation objects. The project has excellent scaffolding (CI, linting, typing) but the core library code has bugs, missing API exports, fragile subprocess handling, incomplete type contracts, and no user-facing documentation. The dcmqi JSON schema (`seg-schema.json`) defines fields and constraints that pydcmqi doesn't fully reflect. Goal: make this a robust, usable library that someone can `pip install` and immediately understand how to use.

**Target: Python 3.10+** (drop 3.8/3.9 support).

---

## Phase 1: Python 3.10+ & Config Updates

### 1.1 Update pyproject.toml
- `requires-python = ">=3.10"`
- Remove classifiers for 3.8, 3.9
- `python_version = "3.10"` for mypy
- `py-version = "3.10"` for pylint

### 1.2 Update CI matrix
- **File**: `.github/workflows/ci.yml`
- Test matrix: `[3.10, 3.12]` (drop 3.8, keep or drop PyPy)

---

## Phase 2: Bug Fixes & Code Correctness

### 2.1 Fix `load()` return type
- **File**: `src/pydcmqi/segimage.py:594`
- Annotated `-> bool` but returns `None`. Change to `-> None`.

### 2.2 Fix `diable_sanity_check` typo
- **File**: `src/pydcmqi/segimage.py:367`
- Rename to `disable_sanity_check`. Project is "Planning" status, no downstream consumers.

### 2.3 Replace `assert` with proper exceptions (5 locations in src/)
- **Line 207** (`getConfigData`): → `raise ValueError("Segment data failed validation.")`
- **Line 280** (`_triplet_setter`): → `raise TypeError(...)`
- **Lines 391-392** (`setFile`): → `raise ValueError(...)` with descriptive messages
- **Line 742** (`getExportedConfiguration`): → `raise RuntimeError("No data loaded. Call load() first.")`

### 2.4 Fix `SegImageData._data` TypedDict violation
- **File**: `src/pydcmqi/types.py`
- `SegImageDict`: make `total=False` (matches dcmqi schema where all top-level fields are optional)
- `SegmentDict`: split into required base + optional via class inheritance:
  ```python
  class _SegmentDictRequired(TypedDict):
      labelID: int
      SegmentedPropertyCategoryCodeSequence: TripletDict
      SegmentedPropertyTypeCodeSequence: TripletDict
      SegmentAlgorithmType: str

  class SegmentDict(_SegmentDictRequired, total=False):
      SegmentLabel: str
      SegmentDescription: str
      ...
  ```

### 2.5 Fix duplicate docstrings
- `Segment`, `SegImageData`, `SegImageFiles` all have same docstring. Give each a distinct one.

### 2.6 Fix README broken links
- **File**: `README.md:17-19`
- `[text(url)]` → `[text](url)` for quantitative imaging and DICOM links.

---

## Phase 3: Replace Triplet with pydicom Code / highdicom CodedConcept Interop

### Analysis

Investigated `highdicom/src/highdicom/sr/coding.py` — `CodedConcept` is a `pydicom.dataset.Dataset` subclass with:
- Constructor: `CodedConcept(value, scheme_designator, meaning)`
- Properties: `.value`, `.scheme_designator`, `.meaning`
- `from_code(code)`: accepts pydicom `Code` (a NamedTuple)
- Handles long codes (LongCodeValue, URNCodeValue)
- Equality/hashing based on scheme + value

Also found `highdicom.seg.SegmentAlgorithmTypeValues` enum (`AUTOMATIC`, `MANUAL`, `SEMIAUTOMATIC`) and `highdicom.seg.SegmentDescription` which mirrors pydcmqi's `SegmentData`.

### Decision: Accept Code/CodedConcept, keep Triplet as thin adapter

**Rationale**: pydcmqi's core job is serializing to/from dcmqi's JSON format (`{"CodeValue": "...", "CodingSchemeDesignator": "...", "CodeMeaning": "..."}`). We need:
1. Mutable code objects (users set `.label`, `.code`, `.scheme` individually — see test line 537-539)
2. Dict serialization matching dcmqi JSON format
3. Interoperability with pydicom `Code` and highdicom `CodedConcept`

pydicom's `Code` is immutable (NamedTuple) → can't replace Triplet directly. CodedConcept is a Dataset subclass → too heavy for a 3-field data class. But we should interoperate.

### Implementation

**Keep `Triplet` but make it ecosystem-aware:**

1. **Accept pydicom Code and highdicom CodedConcept as input everywhere a Triplet is accepted:**
   ```python
   # In _triplet_setter and factory methods:
   CodeLike = Triplet | tuple[str, str, str] | TripletDict
   # When pydicom is available, also accept Code/CodedConcept via duck typing
   ```

2. **Add `Triplet.from_code()` class method** — accepts any object with `.value`, `.scheme_designator`, `.meaning` attributes (duck-typed, no hard dependency):
   ```python
   @staticmethod
   def from_code(code) -> Triplet:
       """Create from pydicom Code or highdicom CodedConcept."""
       return Triplet(code.meaning, code.value, code.scheme_designator)
   ```

3. **Add `Triplet.to_code()` method** — returns a pydicom `Code` if pydicom is installed (optional):
   ```python
   def to_code(self):
       """Convert to pydicom Code. Requires pydicom."""
       from pydicom.sr.coding import Code
       return Code(self.code, self.scheme, self.label)
   ```

4. **Update `_triplet_setter` in SegmentData** to accept the broader type:
   ```python
   def _triplet_setter(self, key: str, value: tuple[str, str, str] | Triplet):
       if isinstance(value, Triplet):
           pass  # already a Triplet
       elif isinstance(value, tuple):
           value = Triplet.fromTuple(value)
       elif hasattr(value, 'value') and hasattr(value, 'scheme_designator'):
           value = Triplet.from_code(value)
       else:
           raise TypeError(...)
       self._data[key] = value.asdict()
   ```

5. **Add `SegmentAlgorithmType` Literal** in types.py:
   ```python
   SegmentAlgorithmType = Literal["AUTOMATIC", "MANUAL", "SEMIAUTOMATIC"]
   ```
   Use this in `SegmentData.segmentAlgorithmType` setter for validation.

6. **Do NOT add pydicom/highdicom as hard dependencies** — keep them optional. The duck-typing approach means pydcmqi works standalone but interoperates when the ecosystem is present.

---

## Phase 4: Subprocess Robustness

### 4.1 Add dcmqi tool availability check
- The `dcmqi >= 0.2.0` pip dependency auto-installs the CLI binaries (`segimage2itkimage`, `itkimage2segimage`, etc.) via platform-specific wheels. Normal `pip install pydcmqi` already puts them on PATH.
- Still add a `shutil.which()` check as a safety net (e.g. user installed pydcmqi in a different env than dcmqi).
- Error: `RuntimeError("dcmqi tool 'segimage2itkimage' not found. Install dcmqi: pip install dcmqi")`

### 4.2 Capture subprocess output and wrap errors
- Change to `subprocess.run(cmd, capture_output=True, text=True, check=False)`
- On failure, raise `DcmqiError(cmd, returncode, stderr)` — custom exception with the dcmqi stderr output.
- dcmqi prints `"ERROR: ..."` to stderr on failure (confirmed from C++ source).
- On success with verbose, log stdout/stderr via Python logging.
- **New file**: `src/pydcmqi/exceptions.py`

### 4.3 Expose additional dcmqi CLI options
Per dcmqi XML parameter definitions (`segimage2itkimage.xml`, `itkimage2segimage.xml`):

**`load()` additions**:
- `merge_segments: bool = False` → `--mergeSegments`

**`write()` additions**:
- `skip_empty_slices: bool = True` → `--skip`
- `geometry_check: bool = True` → `--referencesGeometryCheck`
- `use_label_id_as_segment_number: bool = False` → `--useLabelIDAsSegmentNumber`

---

## Phase 5: Module Organization

Split `segimage.py` (807 lines, 6 classes) into focused modules.

### New structure:
```
src/pydcmqi/
├── __init__.py          # Public API exports
├── _version.pyi         # (unchanged)
├── py.typed             # (unchanged)
├── types.py             # TypedDicts + Literals (updated)
├── exceptions.py        # DcmqiError (new)
├── triplet.py           # Triplet class + _path() helper (extracted)
├── segment.py           # SegmentData, Segment + get_min_max_values() (extracted)
├── segimage.py          # SegImageData, SegImageFiles, SegImage (orchestrator)
```

### Migration:
- `Triplet` + `_path()` → `triplet.py`
- `SegmentData` + `Segment` + `get_min_max_values()` → `segment.py`
- `SegImageData`, `SegImageFiles`, `SegImage` stay in `segimage.py`
- Existing class/method names stay identical
- Each module imports from siblings

### Update `__init__.py` — expose public API:
```python
from pydcmqi._version import version as __version__
from pydcmqi.exceptions import DcmqiError
from pydcmqi.segimage import SegImage, SegImageData, SegImageFiles
from pydcmqi.segment import Segment, SegmentData
from pydcmqi.triplet import Triplet
from pydcmqi.types import SegImageDict, SegmentDict, TripletDict

__all__ = [
    "__version__",
    "DcmqiError",
    "SegImage", "SegImageData", "SegImageFiles",
    "Segment", "SegmentData",
    "Triplet",
    "SegImageDict", "SegmentDict", "TripletDict",
]
```

---

## Phase 6: Logging

- Add `logger = logging.getLogger(__name__)` to each module.
- Replace commented-out `print()` calls (lines 599, 714) with `logger.debug(...)`.
- When `verbose=True`, log subprocess commands at INFO level.

---

## Phase 7: Align Types with dcmqi Schema

### 7.1 Add missing fields to TypedDicts
Per `seg-schema.json` and `common-schema.json`:

**SegImageDict** — add as optional:
- `ContentLabel: str` (CS, max 16)
- `ContentDescription: str` (LO, max 64)

**SegmentDict** — add as optional:
- `TrackingIdentifier: str` (UT)
- `TrackingUniqueIdentifier: str` (UI)
- `RecommendedDisplayCIELabValue: list[int]` (3 ints)
- `SegmentedPropertyTypeModifierCodeSequence: TripletDict`
- `AnatomicRegionSequence: TripletDict`
- `AnatomicRegionModifierSequence: TripletDict`

These ensure round-trip fidelity — data from dcmqi won't be silently dropped.

---

## Phase 8: README & Documentation

### 8.1 Fix broken links (lines 17-19)
### 8.2 Add quickstart section:
```python
from pydcmqi import SegImage

# Load a DICOM segmentation
seg = SegImage()
seg.load("path/to/file.seg.dcm")

for segment in seg.segments:
    print(segment.data.label, segment.data.rgb)
    arr = segment.numpy  # numpy array

# Create and write
seg = SegImage()
seg.data.seriesDescription = "My Segmentation"
s = seg.new_segment()
s.data.label = "Liver"
s.data.segmentAlgorithmType = "AUTOMATIC"
s.data.segmentedPropertyCategory = ("Tissue", "85756007", "SCT")
s.data.segmentedPropertyType = ("Liver", "10200004", "SCT")
s.setFile("liver.nii.gz", labelID=1)
seg.write("output.seg.dcm", "dicom_dir/")
```

### 8.3 Document pydicom/highdicom interop:
```python
from pydicom.sr.coding import Code
from pydcmqi import Triplet

# Accept pydicom Code
liver_code = Code("10200004", "SCT", "Liver")
t = Triplet.from_code(liver_code)

# Convert back
code = t.to_code()
```

---

## Phase 9: Test Updates

### 9.1 Update imports
- Verify `from pydcmqi import SegImage` works alongside `from pydcmqi.segimage import SegImage`
- Add test for public API completeness

### 9.2 Add tests for new error paths
- `DcmqiError` raised when dcmqi tool missing
- `ValueError`/`TypeError` where `assert` was replaced
- `RuntimeError` for `getExportedConfiguration()` before loading

### 9.3 Update renamed parameter
- `diable_sanity_check` → `disable_sanity_check` in any test references

### 9.4 Add Triplet interop tests
- `Triplet.from_code()` with duck-typed object
- `Triplet.to_code()` returns pydicom Code
- `_triplet_setter` accepts Code-like objects

### 9.5 Modernize syntax
- Remove `from __future__ import annotations` from all files
- Native `X | Y` union syntax

---

## Execution Order

1. **Phase 1** — Python 3.10+ config
2. **Phase 2** — Bug fixes
3. **Phase 3** — Triplet/Code interop
4. **Phase 4** — Subprocess robustness + exceptions.py
5. **Phase 5** — Module split + public API
6. **Phase 6** — Logging
7. **Phase 7** — Schema alignment
8. **Phase 8** — README
9. **Phase 9** — Test updates + syntax modernization (final pass)

---

## Files Modified

| File | Action |
|------|--------|
| `pyproject.toml` | Python 3.10+, mypy/pylint targets |
| `.github/workflows/ci.yml` | Update test matrix |
| `src/pydcmqi/__init__.py` | Public API exports |
| `src/pydcmqi/types.py` | Fix TypedDicts, add fields, Literal types |
| `src/pydcmqi/exceptions.py` | **New** — DcmqiError |
| `src/pydcmqi/triplet.py` | **New** — Triplet + from_code/to_code + _path() |
| `src/pydcmqi/segment.py` | **New** — SegmentData, Segment, get_min_max_values() |
| `src/pydcmqi/segimage.py` | Slim to orchestrator, fix bugs, subprocess robustness |
| `README.md` | Fix links, add quickstart + interop docs |
| `tests/test_segimage.py` | Renamed param, error tests, interop tests |
| `tests/test_package.py` | Public API import test |

---

## Phase 10: Save Plan

- Save a copy of this plan as `PLAN.md` in the workspace root for reference.

---

## Verification

1. `nox -s lint` — all linting passes
2. `nox -s tests` — all tests pass (requires dcmqi + network for IDC)
3. `python -c "from pydcmqi import SegImage, Segment, Triplet"` — works
4. `mypy src/` — strict type checking passes
5. Test Code interop: `python -c "from pydcmqi import Triplet; t = Triplet('Liver', '10200004', 'SCT'); print(t.to_code())"`
6. README renders correctly
