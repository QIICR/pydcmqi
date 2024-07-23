from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .types import SegImageDict, SegmentDict, TripletDict

# --== helper and utility functions ==--


def get_min_max_values(image: sitk.Image) -> tuple[float, float]:
    sitk_filter = sitk.MinimumMaximumImageFilter()
    sitk_filter.Execute(image)
    return sitk_filter.GetMinimum(), sitk_filter.GetMaximum()


def _path(path: str | Path) -> Path:
    if isinstance(path, str):
        return Path(path)
    if isinstance(path, Path):
        return path

    msg = "Invalid path type."
    raise ValueError(msg)


# --==      class definitions       ==--


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

    @property
    def valid(self) -> bool:
        """
        A triplet is valid if all fields are non-empty.
        Evaluates to `True` if all fields are non-empty, `False` otherwise.
        """

        return all([self.label != "", self.code != "", self.scheme != ""])


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
        # TODO: implement full validation (schema, code values, etc.)

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
        assert bypass_validation or self.validate(self.data)
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

    def _triplet_setter(self, key: str, value: tuple[str, str, str] | Triplet):
        if isinstance(value, tuple):
            value = Triplet.fromTuple(value)
        assert isinstance(value, Triplet)
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
    """
    A class to store and manipulate the data for a segmentation or region of interest.
    """

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
        self, path: str | Path, labelID: int, diable_sanity_check: bool = False
    ) -> None:
        # make sure path is a Path object
        path = _path(path)

        # run sanity checks
        if not diable_sanity_check:
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
            assert min_val == 0.0
            assert max_val >= labelID

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


class SegImageData:
    """
    A class to store and manipulate the data for a segmentation or region of interest.
    """

    def __init__(self) -> None:
        self._data: SegImageDict = {}

    def setConfigData(self, config: SegImageDict) -> None:
        # NOTE: _data is a pass-by-reference object if we don't use copy
        self._data = config.copy()
        self._data["segmentAttributes"] = []

    def getConfigData(self) -> SegImageDict:
        return self.asdict()

    @property
    def bodyPartExamined(self) -> str:
        return self._data["BodyPartExamined"]

    @bodyPartExamined.setter
    def bodyPartExamined(self, bodyPartExamined: str) -> None:
        self._data["BodyPartExamined"] = bodyPartExamined

    @property
    def clinicalTrialCoordinatingCenterName(self) -> str:
        return self._data["ClinicalTrialCoordinatingCenterName"]

    @clinicalTrialCoordinatingCenterName.setter
    def clinicalTrialCoordinatingCenterName(
        self, clinicalTrialCoordinatingCenterName: str
    ) -> None:
        self._data["ClinicalTrialCoordinatingCenterName"] = (
            clinicalTrialCoordinatingCenterName
        )

    @property
    def clinicalTrialSeriesID(self) -> str:
        return self._data["ClinicalTrialSeriesID"]

    @clinicalTrialSeriesID.setter
    def clinicalTrialSeriesID(self, clinicalTrialSeriesID: str) -> None:
        self._data["ClinicalTrialSeriesID"] = clinicalTrialSeriesID

    @property
    def clinicalTrialTimePointID(self) -> str:
        return self._data["ClinicalTrialTimePointID"]

    @clinicalTrialTimePointID.setter
    def clinicalTrialTimePointID(self, clinicalTrialTimePointID: str) -> None:
        self._data["ClinicalTrialTimePointID"] = clinicalTrialTimePointID

    @property
    def contentCreatorName(self) -> str:
        return self._data["ContentCreatorName"]

    @contentCreatorName.setter
    def contentCreatorName(self, contentCreatorName: str) -> None:
        # TODO: incorporate dicom string format factory & validation
        self._data["ContentCreatorName"] = contentCreatorName

    @property
    def instanceNumber(self) -> str:
        return self._data["InstanceNumber"]

    @instanceNumber.setter
    def instanceNumber(self, instanceNumber: str) -> None:
        self._data["InstanceNumber"] = instanceNumber

    @property
    def seriesDescription(self) -> str:
        return self._data["SeriesDescription"]

    @seriesDescription.setter
    def seriesDescription(self, seriesDescription: str) -> None:
        self._data["SeriesDescription"] = seriesDescription

    @property
    def seriesNumber(self) -> str:
        return self._data["SeriesNumber"]

    @seriesNumber.setter
    def seriesNumber(self, seriesNumber: str) -> None:
        self._data["SeriesNumber"] = seriesNumber

    def asdict(self) -> SegImageDict:
        return self._data.copy()


class SegImageFiles:
    """
    A class to store and manipulate the file paths for a segmentation or region of interest.
    """

    def __init__(self) -> None:
        self._dicomseg: Path | None = None
        self._config: Path | None = None

    @property
    def dicomseg(self) -> Path | None:
        return self._dicomseg

    @property
    def config(self) -> Path | None:
        return self._config


class SegImage:
    """
    A class to store and manipulate the data for a segmentation or region of interest.
    """

    verbose: bool = False

    @classmethod
    def reset(cls) -> None:
        cls.verbose = False

    def __init__(
        self, verbose: bool | None = None, tmp_dir: Path | str | None = None
    ) -> None:
        # set verbose
        if verbose is not None:
            self.verbose = verbose

        # set tmp_dir
        if tmp_dir is None:
            self.tmp_dir = Path(tempfile.gettempdir())
        elif isinstance(tmp_dir, Path):
            self.tmp_dir = tmp_dir
        elif isinstance(tmp_dir, str):
            self.tmp_dir = Path(tmp_dir)
        else:
            raise ValueError(
                "Invalid tmp_dir, must be either None for default, a Path or a string."
            )

        # set instance state variables
        self.data = SegImageData()
        self.loaded = False
        self.files = SegImageFiles()

        self._config: dict | None = None
        self._segments: list[Segment] = []

    def load(
        self,
        dicomseg_file: Path | str,
        output_dir: Path | str | None = None,
    ) -> bool:
        # print(f"Converting file: {dicomseg_file} into {output_dir}.") # TODO: use logging

        # we create a temporary output directory if none is provided in the specified tmp dir
        if output_dir is None:
            output_dir = Path(self.tmp_dir) / "output"
        else:
            output_dir = _path(output_dir)

        # create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # build subprocess command
        cmd = [
            "segimage2itkimage",
            "-t",
            "nifti",
            "-p",
            "pydcmqi",
            "--outputDirectory",
            str(output_dir),
            "--inputDICOM",
            str(dicomseg_file),
        ]

        # run subprocess
        subprocess.run(cmd, check=True)

        # import data
        self._import(output_dir)

        # update file paths
        self.files._dicomseg = dicomseg_file  # pylint: disable=W0212

    def _import(
        self, output_dir: Path, disable_file_sanity_checks: bool = False
    ) -> None:
        # iterate all files in the output directory
        # - store the config file
        # - store the image files

        self._config = None
        self._segments = []

        # read in the config file
        config_file = output_dir / "pydcmqi-meta.json"

        # load the config file
        with Path.open(config_file, encoding="utf-8") as f:
            self._config = json.load(f)

        # load data
        # TODO: or property self.config ??
        self.data.setConfigData(self._config)

        # load each segmentation as item
        for i, s in enumerate(self._config["segmentAttributes"]):
            # find generated export file
            f = output_dir / f"pydcmqi-{i+1}.nii.gz"

            # load all configs from segment definition
            for config in s:
                labeID = int(config["labelID"])

                # create new segment
                segment = self.new_segment()
                segment.setFile(f, labeID, disable_file_sanity_checks)
                segment.config = config

        # update state
        self.loaded = True

        # store file paths
        self.files._config = config_file  # pylint: disable=W0212

    def write(
        self,
        output_file: str | Path,
        dicom_dir: str | Path,
        export_config_to_file: str | Path | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        # make sure the output file is a Path object
        output_file = _path(output_file)
        dicom_dir = _path(dicom_dir)

        if export_config_to_file is not None:
            export_config_to_file = _path(export_config_to_file)

        # check output file
        if not output_file.name.endswith(".seg.dcm"):
            raise ValueError("Output file must end with .seg.dcm.")

        # check if the file already exists and if overwriting protection is disabled
        if output_file.exists() and not allow_overwrite:
            raise FileExistsError(f"Output file already exists: {output_file}.")

        # check that the dicom directory exists and contains at least one *.dcm file
        if not dicom_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {dicom_dir}.")

        if not dicom_dir.is_dir():
            raise ValueError(f"Path is not a directory: {dicom_dir}.")

        # check that the dicom directory contains at least one *.dcm file
        if len(list(dicom_dir.glob("*.dcm"))) == 0:
            raise ValueError(
                f"Directory does not contain any DICOM files: {dicom_dir}."
            )

        # get config
        config = self.config
        files = self.segmentation_files

        # store in the output directory
        # but for now just print
        # print(json.dumps(config, indent=2)) # TODO: use logging

        # store in _debug_test_meta.json
        meta_tmp_file = Path(self.tmp_dir) / "_debug_test_meta.json"
        with Path.open(meta_tmp_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # export config file if requested
        if export_config_to_file is not None:
            shutil.copy(meta_tmp_file, export_config_to_file)

        # construct dcmqi cli command
        cmd = [
            "itkimage2segimage",
            "--inputImageList",
            ",".join(str(fp) for fp in files),
            "--inputDICOMDirectory",
            str(dicom_dir),
            "--outputDICOM",
            str(output_file),
            "--inputMetadata",
            str(meta_tmp_file),
        ]

        # run command
        subprocess.run(cmd, check=True)

    def getExportedConfiguration(self) -> dict:
        assert self.loaded
        return self._config

    @property
    def config(self) -> SegImageDict:
        # generate the config file from the segments
        # NOTE: returns a copy, not a reference to the data json dict
        # NOTE: equivalent to self.data.getConfigData()
        config = self.data.asdict()

        # make sure all segments have a file specified
        for s in self._segments:
            if s.path is None:
                raise ValueError(f"Segment {s} has no file specified.")

        # sort segments by files
        f2s: dict[str, list[Segment]] = {}
        for s in self._segments:
            p = str(s.path)
            if p not in f2s:
                f2s[p] = []
            f2s[p].append(s)

        # sort the segments by their labelID
        f2s = {k: sorted(v, key=lambda x: x.labelID) for k, v in f2s.items()}

        # order the dictionary by it's keys
        of2s = OrderedDict(sorted(f2s.items()))

        # check that for all files
        # - no duplicate labelIDs are present
        # - all labelIDs are continuous and start at 1
        for f, s in of2s.items():
            labelIDs = [x.labelID for x in s]
            if len(labelIDs) != len(set(labelIDs)):
                raise ValueError(f"Duplicate labelIDs found in {f}.")
            # if min(labelIDs) != 1:
            #     raise ValueError(f"LabelIDs must start at 1 in {f}.")
            # if max(labelIDs) != len(labelIDs):
            #     raise ValueError(f"LabelIDs must be continuous in {f}.")

        # add each segment to the config
        config["segmentAttributes"] = [[s.config for s in ss] for ss in of2s.values()]

        # return the generated config
        return config

    @property
    def segmentation_files(self) -> list[Path]:
        return sorted({s.path for s in self._segments})

    @config.setter
    def config(self, config: SegImageDict) -> None:
        self.data.setConfigData(config)

    @property
    def segments(self) -> list[Segment]:
        return self._segments

    def add_segment(self, segment: Segment) -> None:
        self._segments.append(segment)

    def new_segment(self) -> Segment:
        segment = Segment()
        self._segments.append(segment)
        return segment
