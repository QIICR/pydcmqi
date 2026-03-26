# pydcmqi
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/QIICR/pydcmqi/main.svg)](https://results.pre-commit.ci/latest/github/QIICR/pydcmqi/main)

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

## About

`pydcmqi` is a Python API wrapper for the
[QIICR/dcmqi](https://github.com/QIICR/dcmqi) library and command line tool
collection for standardized communication of
[quantitative image analysis](http://journals.sagepub.com/doi/pdf/10.1177/0962280214537333)
research data using the [DICOM standard](https://en.wikipedia.org/wiki/DICOM).

This package is in its early development stages. Its functionality and API will
change. Please share feedback by opening issues in this repository.

## Installation

```bash
pip install pydcmqi
```

This installs pydcmqi and the dcmqi CLI binaries automatically.

## Quick Start

### Load a DICOM Segmentation

```python
from pydcmqi import SegImage

seg = SegImage()
seg.load("path/to/file.seg.dcm")

for segment in seg.segments:
    print(segment.data.label, segment.data.rgb)
    arr = segment.numpy  # get as numpy array
```

### Create and Write a DICOM Segmentation

```python
from pydcmqi import SegImage

seg = SegImage()
seg.data.seriesDescription = "My Segmentation"
seg.data.contentCreatorName = "Researcher"

s = seg.new_segment()
s.data.label = "Liver"
s.data.segmentAlgorithmType = "AUTOMATIC"
s.data.segmentAlgorithmName = "MyModel"
s.data.segmentedPropertyCategory = ("Tissue", "85756007", "SCT")
s.data.segmentedPropertyType = ("Liver", "10200004", "SCT")
s.data.rgb = (220, 129, 101)
s.setFile("liver.nii.gz", labelID=1)

seg.write("output.seg.dcm", "dicom_image_dir/")
```

### Interoperability with pydicom / highdicom

```python
from pydicom.sr.coding import Code
from pydcmqi import Triplet

# Create a Triplet from a pydicom Code
liver_code = Code("10200004", "SCT", "Liver")
t = Triplet.from_code(liver_code)

# Convert back to pydicom Code
code = t.to_code()
```

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/ImagingDataCommons/pydcmqi/workflows/CI/badge.svg
[actions-link]:             https://github.com/ImagingDataCommons/pydcmqi/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/ImagingDataCommons/pydcmqi/discussions
[pypi-link]:                https://pypi.org/project/pydcmqi/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/pydcmqi
[pypi-version]:             https://img.shields.io/pypi/v/pydcmqi
[rtd-badge]:                https://readthedocs.org/projects/pydcmqi/badge/?version=latest
[rtd-link]:                 https://pydcmqi.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
