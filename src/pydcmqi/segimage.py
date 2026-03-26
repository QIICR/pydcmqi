import json
import logging
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .exceptions import DcmqiError
from .segment import Segment
from .triplet import _path
from .types import SegImageDict

logger = logging.getLogger(__name__)


def _run_dcmqi(
    cmd: list[str], *, verbose: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a dcmqi CLI command with proper error handling."""
    tool = cmd[0]
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"dcmqi tool '{tool}' not found. Install dcmqi: pip install dcmqi"
        )

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise DcmqiError(cmd, result.returncode, result.stderr)

    if verbose and result.stdout:
        logger.debug("stdout: %s", result.stdout.strip())
    if verbose and result.stderr:
        logger.debug("stderr: %s", result.stderr.strip())

    return result


class SegImageData:
    """Metadata container for a DICOM segmentation image (series-level attributes)."""

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
    """File path container for DICOM segmentation and config files."""

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
    """Main class for loading, creating, and writing DICOM segmentation objects via dcmqi."""

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

        self._config: dict[str, Any] | None = None
        self._segments: list[Segment] = []

    def load(
        self,
        dicomseg_file: Path | str,
        output_dir: Path | str | None = None,
        merge_segments: bool = False,
    ) -> None:
        logger.debug("Loading DICOM SEG: %s", dicomseg_file)

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

        if merge_segments:
            cmd.append("--mergeSegments")

        # run subprocess
        _run_dcmqi(cmd, verbose=self.verbose)

        # import data
        self._import(output_dir)

        # update file paths
        self.files._dicomseg = _path(dicomseg_file)  # pylint: disable=W0212

    def _import(
        self, output_dir: Path, disable_file_sanity_checks: bool = False
    ) -> None:
        self._config = None
        self._segments = []

        # read in the config file
        config_file = output_dir / "pydcmqi-meta.json"

        # load the config file
        with Path.open(config_file, encoding="utf-8") as f:
            self._config = json.load(f)

        # load data
        assert self._config is not None
        self.data.setConfigData(self._config)  # type: ignore[arg-type]

        # load each segmentation as item
        for i, seg_attrs in enumerate(self._config["segmentAttributes"]):
            # find generated export file
            seg_file = output_dir / f"pydcmqi-{i+1}.nii.gz"

            # load all configs from segment definition
            for seg_config in seg_attrs:
                label_id = int(seg_config["labelID"])

                # create new segment
                segment = self.new_segment()
                segment.setFile(seg_file, label_id, disable_file_sanity_checks)
                segment.config = seg_config

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
        skip_empty_slices: bool = True,
        geometry_check: bool = True,
        use_label_id_as_segment_number: bool = False,
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

        logger.debug("Writing config: %s", json.dumps(config, indent=2))

        # write metadata to temp file
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
            "--skip",
            "1" if skip_empty_slices else "0",
            "--referencesGeometryCheck",
            "1" if geometry_check else "0",
        ]

        if use_label_id_as_segment_number:
            cmd.append("--useLabelIDAsSegmentNumber")

        # run command
        _run_dcmqi(cmd, verbose=self.verbose)

    def getExportedConfiguration(self) -> dict:  # type: ignore[type-arg]
        if not self.loaded or self._config is None:
            raise RuntimeError("No data loaded. Call load() first.")
        return self._config

    @property
    def config(self) -> SegImageDict:
        # generate the config file from the segments
        # NOTE: returns a copy, not a reference to the data json dict
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

        # order the dictionary by its keys
        of2s = OrderedDict(sorted(f2s.items()))

        # check that for all files no duplicate labelIDs are present
        for file_path, segs in of2s.items():
            labelIDs = [x.labelID for x in segs]
            if len(labelIDs) != len(set(labelIDs)):
                raise ValueError(f"Duplicate labelIDs found in {file_path}.")

        # add each segment to the config
        config["segmentAttributes"] = [[s.config for s in ss] for ss in of2s.values()]

        # return the generated config
        return config

    @property
    def segmentation_files(self) -> list[Path]:
        return sorted(s.path for s in self._segments if s.path is not None)

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
