from __future__ import annotations

import importlib.metadata

import pydcmqi as m


def test_version():
    assert importlib.metadata.version("pydcmqi") == m.__version__
