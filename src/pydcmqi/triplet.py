from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import TripletDict


def _path(path: str | Path) -> Path:
    if isinstance(path, str):
        return Path(path)
    if isinstance(path, Path):
        return path

    msg = "Invalid path type."
    raise ValueError(msg)


class Triplet:
    """
    A triplet is a data structure that consists of three elements:
    - `code meaning`, a human readable label
    - `code value`, a unique identifier
    - `coding scheme designator`, the issuer of the code value
    """

    @staticmethod
    def fromDict(d: TripletDict) -> Triplet:
        """
        Create a LabeledTriplet from a dictionary.

        ```
        {
          "CodeMeaning": "<code meaning>",
          "CodeValue": "<code value>",
          "CodingSchemeDesignator": "<coding scheme designator>"
        }
        ```
        """
        return Triplet(d["CodeMeaning"], d["CodeValue"], d["CodingSchemeDesignator"])

    @staticmethod
    def fromTuple(t: tuple[str, str, str]) -> Triplet:
        """
        Create a LabeledTriplet from a tuple.

        ```
        ("<code meaning>", "<code value>", "<coding scheme designator>")
        ```
        """
        return Triplet(t[0], t[1], t[2])

    @staticmethod
    def from_code(code: Any) -> Triplet:
        """
        Create a Triplet from a pydicom Code or highdicom CodedConcept.

        Accepts any object with `value`, `scheme_designator`, and `meaning` attributes.
        """
        return Triplet(code.meaning, code.value, code.scheme_designator)

    @staticmethod
    def empty() -> Triplet:
        """
        Create an empty triplet.
        """
        return Triplet("", "", "")

    def __init__(self, label: str, code: str, scheme: str) -> None:
        self.label = label
        self.code = code
        self.scheme = scheme

    def __repr__(self) -> str:
        return f"[{self.code}:{self.scheme}|{self.label}]"

    def __str__(self) -> str:
        return self.label

    def asdict(self) -> TripletDict:
        """
        Convert the triplet to a dictionary:

        ```
        {
          "CodeMeaning": "<code meaning>",
          "CodeValue": "<code value>",
          "CodingSchemeDesignator": "<coding scheme designator>"
        }
        ```
        """

        return {
            "CodeMeaning": self.label,
            "CodeValue": self.code,
            "CodingSchemeDesignator": self.scheme,
        }

    def to_code(self) -> Any:
        """
        Convert to a pydicom Code. Requires pydicom to be installed.
        """
        from pydicom.sr.coding import Code
        return Code(self.code, self.scheme, self.label)

    @property
    def valid(self) -> bool:
        """
        A triplet is valid if all fields are non-empty.
        Evaluates to `True` if all fields are non-empty, `False` otherwise.
        """

        return all([self.label != "", self.code != "", self.scheme != ""])
