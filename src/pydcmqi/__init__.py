"""
Copyright (c) 2024 Leonard Nürnberg. All rights reserved.

pydcmqi: Python api wrapper and utilities for the dcmqi binary.
"""

from ._version import version as __version__
from .exceptions import DcmqiError
from .segment import Segment, SegmentData
from .segimage import SegImage, SegImageData, SegImageFiles
from .triplet import Triplet
from .types import SegImageDict, SegmentDict, TripletDict

__all__ = [
    "__version__",
    "DcmqiError",
    "SegImage",
    "SegImageData",
    "SegImageFiles",
    "Segment",
    "SegmentData",
    "Triplet",
    "SegImageDict",
    "SegmentDict",
    "TripletDict",
]
