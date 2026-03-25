from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .triplet import Triplet, _path
from .types import SegmentDict


def get_min_max_values(image: sitk.Image) -> tuple[float, float]:
    sitk_filter = sitk.MinimumMaximumImageFilter()
    sitk_filter.Execute(image)
    return sitk_filter.GetMinimum(), sitk_filter.GetMaximum()


class SegmentData:
    """
    A class to store and manipulate the data for a segmentation or region of interest.
    """

    def __init__(self) -> None:
        self._data: SegmentDict = {
            "labelID": 0,
            "SegmentLabel": "",
            "SegmentDescription": "",
            "SegmentAlgorithmName": "",
            "SegmentAlgorithmType": "",
            "recommendedDisplayRGBValue": [0, 0, 0],
            "SegmentedPropertyCategoryCodeSequence": {
                "CodeMeaning": "",
                "CodeValue": "",
                "CodingSchemeDesignator": "",
            },
            "SegmentedPropertyTypeCodeSequence": {
                "CodeMeaning": "",
                "CodeValue": "",
                "CodingSchemeDesignator": "",
            },
        }

    def setConfigData(self, config: dict) -> None:
        self._data = config.copy()

    def _bake_data(self) -> SegmentDict:
        # start from internal data
        d = self._data.copy()

        # triplets
        triplet_keys = [
            "SegmentedPropertyCategoryCodeSequence",
            "SegmentedPropertyTypeCodeSequence",
            "SegmentedPropertyTypeModifierCodeSequence",
            "AnatomicRegionSequence",
            "AnatomicRegionModifierSequence",
        ]

        # optional triplets
        triplet_keys_optional = [
            "SegmentedPropertyTypeModifierCodeSequence",
            "AnatomicRegionSequence",
            "AnatomicRegionModifierSequence",
        ]

        for k in triplet_keys:
            if k not in d and k in triplet_keys_optional:
                continue
            t = self._triplet_factory(k)
            d[k] = t.asdict()

        # return constructed data
        return d

    @staticmethod
    def validate(data: dict) -> bool:
        required_fields = [
            "labelID" in data,
            "SegmentLabel" in data,
            "SegmentDescription" in data,
            "SegmentAlgorithmName" in data,
            "SegmentAlgorithmType" in data,
            "recommendedDisplayRGBValue" in data,
            "SegmentedPropertyCategoryCodeSequence" in data,
            Triplet.fromDict(data["SegmentedPropertyCategoryCodeSequence"]).valid,
            "SegmentedPropertyTypeCodeSequence" in data,
            Triplet.fromDict(data["SegmentedPropertyTypeCodeSequence"]).valid,
        ]

        optional_fields = [
            "SegmentedPropertyTypeModifierCodeSequence" not in data
            or Triplet.fromDict(
                data["SegmentedPropertyTypeModifierCodeSequence"]
            ).valid,
            "AnatomicRegionSequence" not in data
            or Triplet.fromDict(data["AnatomicRegionSequence"]).valid,
            "AnatomicRegionModifierSequence" not in data
            or Triplet.fromDict(data["AnatomicRegionModifierSequence"]).valid,
        ]

        return all(required_fields) and all(optional_fields)

    def getConfigData(self, bypass_validation: bool = False) -> dict:
        if not bypass_validation and not self.validate(self.data):
            raise ValueError("Segment data failed validation.")
        return self.data.copy()

    @property
    def data(self) -> SegmentDict:
        return self._bake_data()

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def _triplet_factory(self, key: str) -> Triplet:
        if not hasattr(self, f"__tpf_{key}"):
            if key in self._data:
                t = Triplet.fromDict(self._data[key])
            else:
                t = Triplet.empty()
            setattr(self, f"__tpf_{key}", t)
        return getattr(self, f"__tpf_{key}")

    @property
    def label(self) -> str:
        return self._data["SegmentLabel"]

    @label.setter
    def label(self, label: str) -> None:
        self._data["SegmentLabel"] = label

    @property
    def description(self) -> str:
        return self._data["SegmentDescription"]

    @description.setter
    def description(self, description: str) -> None:
        self._data["SegmentDescription"] = description

    @property
    def rgb(self) -> tuple[int, int, int]:
        return tuple(self._data["recommendedDisplayRGBValue"])

    @rgb.setter
    def rgb(self, rgb: tuple[int, int, int]) -> None:
        self._data["recommendedDisplayRGBValue"] = list(rgb)

    @property
    def labelID(self) -> int:
        return self._data["labelID"]

    @labelID.setter
    def labelID(self, labelID: int) -> None:
        self._data["labelID"] = labelID

    @property
    def segmentAlgorithmName(self) -> str:
        return self._data["SegmentAlgorithmName"]

    @segmentAlgorithmName.setter
    def segmentAlgorithmName(self, segmentAlgorithmName: str) -> None:
        self._data["SegmentAlgorithmName"] = segmentAlgorithmName

    @property
    def segmentAlgorithmType(self) -> str:
        return self._data["SegmentAlgorithmType"]

    @segmentAlgorithmType.setter
    def segmentAlgorithmType(self, segmentAlgorithmType: str) -> None:
        self._data["SegmentAlgorithmType"] = segmentAlgorithmType

    def _triplet_setter(self, key: str, value: tuple[str, str, str] | Triplet) -> None:
        if isinstance(value, Triplet):
            pass
        elif isinstance(value, tuple):
            value = Triplet.fromTuple(value)
        elif hasattr(value, "value") and hasattr(value, "scheme_designator") and hasattr(value, "meaning"):
            value = Triplet.from_code(value)
        else:
            raise TypeError(f"Expected Triplet, tuple, or Code-like object, got {type(value)}")
        self._data[key] = value.asdict()

    @property
    def segmentedPropertyCategory(self) -> Triplet:
        return self._triplet_factory("SegmentedPropertyCategoryCodeSequence")

    @segmentedPropertyCategory.setter
    def segmentedPropertyCategory(self, value: tuple[str, str, str] | Triplet) -> None:
        self._triplet_setter("SegmentedPropertyCategoryCodeSequence", value)

    @property
    def segmentedPropertyType(self) -> Triplet:
        return self._triplet_factory("SegmentedPropertyTypeCodeSequence")

    @segmentedPropertyType.setter
    def segmentedPropertyType(self, value: tuple[str, str, str] | Triplet) -> None:
        self._triplet_setter("SegmentedPropertyTypeCodeSequence", value)

    @property
    def segmentedPropertyTypeModifier(self) -> Triplet:
        return self._triplet_factory("SegmentedPropertyTypeModifierCodeSequence")

    @segmentedPropertyTypeModifier.setter
    def segmentedPropertyTypeModifier(
        self, value: tuple[str, str, str] | Triplet
    ) -> None:
        self._triplet_setter("SegmentedPropertyTypeModifierCodeSequence", value)

    @property
    def hasSegmentedPropertyTypeModifier(self) -> bool:
        return "SegmentedPropertyTypeModifierCodeSequence" in self._data

    @property
    def anatomicRegion(self) -> Triplet:
        return self._triplet_factory("AnatomicRegionSequence")

    @anatomicRegion.setter
    def anatomicRegion(self, value: tuple[str, str, str] | Triplet) -> None:
        self._triplet_setter("AnatomicRegionSequence", value)

    @property
    def hasAnatomicRegion(self) -> bool:
        return "AnatomicRegionSequence" in self._data

    @property
    def anatomicRegionModifier(self) -> Triplet:
        return self._triplet_factory("AnatomicRegionModifierSequence")

    @anatomicRegionModifier.setter
    def anatomicRegionModifier(self, value: tuple[str, str, str] | Triplet) -> None:
        self._triplet_setter("AnatomicRegionModifierSequence", value)

    @property
    def hasAnatomicRegionModifier(self) -> bool:
        return "AnatomicRegionModifierSequence" in self._data


class Segment:
    """A single segmentation with associated file path, image data, and metadata."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.data = SegmentData()

        self._cached_itk: sitk.Image | None = None
        self._cached_numpy: np.ndarray | None = None

    @property
    def config(self) -> dict:
        return self.data.getConfigData()

    @config.setter
    def config(self, config: dict) -> None:
        self.data.setConfigData(config)

    @property
    def labelID(self) -> int:
        return self.data.labelID

    @labelID.setter
    def labelID(self, labelID: int) -> None:
        self.data.labelID = labelID

    def setFile(
        self, path: str | Path, labelID: int, disable_sanity_check: bool = False
    ) -> None:
        # make sure path is a Path object
        path = _path(path)

        # run sanity checks
        if not disable_sanity_check:
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {path}")

            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")

            # read image
            image = sitk.ReadImage(str(path))

            # check file has as many labels as expected
            if image.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError(
                    f"Image must have only one component per pixel: {path}"
                )

            # get min/max values
            min_val, max_val = get_min_max_values(image)
            if min_val != 0.0:
                raise ValueError(f"Image minimum value must be 0, got {min_val}: {path}")
            if max_val < labelID:
                raise ValueError(f"Image max value ({max_val}) is less than labelID ({labelID}): {path}")

        # set path and label id
        self.path = path
        self.labelID = labelID

    @property
    def itk(self) -> sitk.Image:
        # read image if not cached
        if self._cached_itk is None:
            self._cached_itk = sitk.ReadImage(str(self.path))

        # return image
        return self._cached_itk

    @property
    def numpy(self) -> np.ndarray:
        # read image if not cached
        if self._cached_numpy is None:
            self._cached_numpy = sitk.GetArrayFromImage(self.itk)

        # convert to numpy
        return self._cached_numpy

    def isBinary(self) -> bool:
        uv = np.unique(self.numpy)
        return len(uv) == 2 and 0 in uv and 1 in uv

    def isMultiLabel(self) -> bool:
        uv = np.unique(self.numpy)
        return len(uv) > 2

    def isLabel(self, label: int) -> bool:
        return label in np.unique(self.numpy)

    def isLabelSet(self, labels: list[int]) -> bool:
        return all(self.isLabel(label) for label in labels)

    def isLabelRange(self, start: int, end: int) -> bool:
        return self.isLabelSet(list(range(start, end + 1)))

    @property
    def binary(self) -> np.ndarray:
        return self.numpy == self.labelID

    def saveAsBinary(self, path: str | Path) -> None:
        # make sure path is a Path object
        path = _path(path)

        # create image
        image = sitk.GetImageFromArray(self.binary)
        image.CopyInformation(self.itk)

        # write image
        sitk.WriteImage(image, str(path))
