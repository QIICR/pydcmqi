import importlib.metadata

import pydcmqi as m


def test_version():
    assert importlib.metadata.version("pydcmqi") == m.__version__


def test_public_api():
    """Verify all expected symbols are importable from the top-level package."""
    from pydcmqi import (
        DcmqiError,
        SegImage,
        Triplet,
    )

    # Verify they are the right types
    assert callable(SegImage)
    assert callable(Triplet)
    assert callable(DcmqiError)
