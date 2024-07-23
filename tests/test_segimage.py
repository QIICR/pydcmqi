from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from idc_index import index

from pydcmqi.segimage import SegImage, SegmentData, Triplet

TEST_DIR = Path(__file__).resolve().parent / "test_data"

# force loading of the segmentation data
# NOTE: only set to false to speed up manual testing.
FORCE_LOADING = True


# helper function to sort dictionaries
# ONLY FOR FILE EXPORT
# DICTS ARE NOT PERSISTENTLY ORDERED IN PYTHON
def _iterative_dict_sort(d):
    if isinstance(d, list):
        return [_iterative_dict_sort(v) for v in d]
    if isinstance(d, dict):
        return {k: _iterative_dict_sort(v) for k, v in sorted(d.items())}
    return d


class TestTriplets:
    def test_triplet_from_tuple(self):
        d = ("Liver", "123037004", "SCT")
        t = Triplet.fromTuple(d)

        assert t.label == "Liver"
        assert t.code == "123037004"
        assert t.scheme == "SCT"
        assert t.valid

    def test_triplet_from_dict(self):
        d = {
            "CodeMeaning": "Anatomical Structure",
            "CodeValue": "123037004",
            "CodingSchemeDesignator": "SCT",
        }
        t = Triplet.fromDict(d)

        assert t.label == "Anatomical Structure"
        assert t.code == "123037004"
        assert t.scheme == "SCT"
        assert t.valid

    def test_triplet_initializer(self):
        t = Triplet("Anatomical Structure", "123037004", "SCT")

        assert t.label == "Anatomical Structure"
        assert t.code == "123037004"
        assert t.scheme == "SCT"
        assert t.valid

    def test_empty_triplet(self):
        t = Triplet.empty()

        assert t.label == ""
        assert t.code == ""
        assert t.scheme == ""
        assert not t.valid


class TestSegmentData:
    def test_triplet_property_from_tuple(self):
        d = SegmentData()
        d.segmentedPropertyCategory = ("Anatomical Structure", "123037004", "SCT")

        assert isinstance(d.segmentedPropertyCategory, Triplet)
        assert d.segmentedPropertyCategory.label == "Anatomical Structure"
        assert d.segmentedPropertyCategory.code == "123037004"
        assert d.segmentedPropertyCategory.scheme == "SCT"
        assert (
            d.data["SegmentedPropertyCategoryCodeSequence"]
            == d.segmentedPropertyCategory.asdict()
        )

    def test_triplet_property_from_object(self):
        d = SegmentData()
        d.segmentedPropertyCategory = Triplet(
            "Anatomical Structure", "123037004", "SCT"
        )

        assert isinstance(d.segmentedPropertyCategory, Triplet)
        assert d.segmentedPropertyCategory.label == "Anatomical Structure"
        assert d.segmentedPropertyCategory.code == "123037004"
        assert d.segmentedPropertyCategory.scheme == "SCT"
        assert (
            d.data["SegmentedPropertyCategoryCodeSequence"]
            == d.segmentedPropertyCategory.asdict()
        )

    def test_triplet_property_from_properties(self):
        d = SegmentData()
        d.segmentedPropertyCategory.label = "Anatomical Structure"
        d.segmentedPropertyCategory.code = "123037004"
        d.segmentedPropertyCategory.scheme = "SCT"

        assert isinstance(d.segmentedPropertyCategory, Triplet)
        assert d.segmentedPropertyCategory.label == "Anatomical Structure"
        assert d.segmentedPropertyCategory.code == "123037004"
        assert d.segmentedPropertyCategory.scheme == "SCT"
        assert (
            d.data["SegmentedPropertyCategoryCodeSequence"]
            == d.segmentedPropertyCategory.asdict()
        )


class TestSegImageClass:
    ### SETUP
    def setup_method(self):
        SegImage.reset()

    ### verbosity
    def test_verbosity_default(self):
        # initialize
        segimg = SegImage()

        # verbose mode is disabled by default
        assert not SegImage.verbose
        assert not segimg.verbose

    def test_verbosity_default_global_override(self):
        # set globally to true
        SegImage.verbose = True

        # initialize
        segimg1 = SegImage()
        segimg2 = SegImage(verbose=False)

        #
        assert SegImage.verbose
        assert segimg1.verbose
        assert not segimg2.verbose

    def test_verbosity_instance_override(self):
        # initialize
        segimg = SegImage(verbose=True)

        #
        assert not SegImage.verbose
        assert segimg.verbose

    ### tmp_dir
    def test_tmp_dir(self):
        # initialize
        segimg1 = SegImage()
        segimg2 = SegImage(tmp_dir=None)
        segimg3 = SegImage(tmp_dir="tmp")
        segimg4 = SegImage(tmp_dir=Path("tmp"))

        # all instances will have a temp dir set
        assert segimg1.tmp_dir is not None
        assert segimg2.tmp_dir is not None
        assert segimg3.tmp_dir is not None
        assert segimg4.tmp_dir is not None

        # the type of the tmp_dir is Path, no matter how it was initilaized
        assert isinstance(segimg1.tmp_dir, Path)
        assert isinstance(segimg2.tmp_dir, Path)
        assert isinstance(segimg3.tmp_dir, Path)
        assert isinstance(segimg4.tmp_dir, Path)

        # specifying a tmp_dir with e.g. a numeric type will raise an ValueError
        with pytest.raises(
            ValueError,
            match="Invalid tmp_dir, must be either None for default, a Path or a string.",
        ):
            _ = SegImage(tmp_dir=1)


class TestSegimageRead:
    # set-up download data from idc
    def setup_class(self):
        # define output directory
        self.tmp_dir = TEST_DIR / "tmp"
        self.img_dir = TEST_DIR / "image"
        self.seg_dir = TEST_DIR / "seg"
        self.out_dir = TEST_DIR / "out"

        # create directories
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # initialize a SegImage instance used in multiple tests
        self.segimg = SegImage(self.tmp_dir)

        # initialize idc index client
        client = index.IDCClient()

        # download a specific sid from idc
        #  CT image:  1.3.6.1.4.1.14519.5.2.1.8421.4008.761093011533106086639756339870
        #  LIVER SEG: 1.2.276.0.7230010.3.1.3.17436516.270878.1696966588.943239

        # PT image       (6):   1.3.6.1.4.1.14519.5.2.1.4334.1501.680033973739971488930649469577
        # LUNG+TUMOR SEG (300): 1.2.276.0.7230010.3.1.3.17436516.538020.1696968975.507837

        if not [f for f in os.listdir(self.img_dir) if f.endswith(".dcm")]:
            print("downloading image")
            client.download_dicom_series(
                "1.3.6.1.4.1.14519.5.2.1.4334.1501.680033973739971488930649469577",
                self.img_dir,
                quiet=False,
            )

        if not [f for f in os.listdir(self.seg_dir) if f.endswith(".dcm")]:
            print("downloading segment")
            client.download_dicom_series(
                "1.2.276.0.7230010.3.1.3.17436516.538020.1696968975.507837",
                self.seg_dir,
                quiet=False,
            )

        # check download and assign files
        sf = [f for f in os.listdir(self.seg_dir) if f.endswith(".dcm")]
        assert len(sf) == 1
        self.seg_file = self.seg_dir / sf[0]

    def setup_method(self):
        # rezet segimage
        SegImage.reset()

        # load segmentation
        if not self.segimg.loaded:
            if not FORCE_LOADING and (self.out_dir / "pydcmqi-meta.json").exists():
                print("======= SKIPPING LOAD AND IMPORT FROM CACHE ========")
                self.segimg._import(self.out_dir, disable_file_sanity_checks=True)
                self.segimg.files._dicomseg = self.seg_file  # fix
            else:
                self.segimg.load(self.seg_file, output_dir=self.out_dir)

    def test_loading(self):
        assert self.segimg.loaded

        # files
        assert self.segimg.files.config is not None
        assert self.segimg.files.config == self.out_dir / "pydcmqi-meta.json"
        assert self.segimg.files.dicomseg is not None
        assert self.segimg.files.dicomseg == self.seg_file

        # config
        assert self.segimg._config is not None
        assert self.segimg.getExportedConfiguration() is not None

    def test_loaded_segment_content(self):
        # check
        segments = list(self.segimg.segments)
        assert len(segments) == 2

        # check segment content
        for segment in segments:
            # get path
            print("item path:", segment.path)

            # compoare numpy shape with itk size
            assert segment.numpy.transpose(2, 1, 0).shape == segment.itk.GetSize()

            # check the file is bindary
            assert segment.isLabel(segment.labelID)

            # check the binary mask
            import numpy as np

            assert segment.binary.shape == segment.numpy.shape
            assert segment.binary.dtype == bool
            assert np.array_equal(segment.binary, (segment.numpy == segment.labelID))
            assert segment.binary.min() == 0
            assert segment.binary.max() == 1

    def test_loaded_segment_config(self):
        # get the raw config
        config = self.segimg.getExportedConfiguration()

        # check config matching per segment
        for i, segment_config in enumerate(config["segmentAttributes"]):
            segment = self.segimg.segments[i]

            # the extracted, generated and validated segment config is identical with the
            #   segment configuration from the original config file
            assert segment.config == segment_config[0]

    def test_loaded_segment_data(self):
        # get the raw config
        # NOTE: this is the json from the original exported config file
        config = self.segimg.getExportedConfiguration()

        # iterate all imported segments and compare the extracted data with
        #  the original json config
        attrs = config["segmentAttributes"]
        for i, segment in enumerate(self.segimg.segments):
            attr = attrs[i][0]

            # check mandtory fields
            assert segment.data.label == attr["SegmentLabel"]
            assert segment.data.description == attr["SegmentDescription"]
            assert segment.data.rgb == tuple(attr["recommendedDisplayRGBValue"])
            assert segment.data.labelID == attr["labelID"]
            assert segment.data.segmentAlgorithmName == attr["SegmentAlgorithmName"]
            assert segment.data.segmentAlgorithmType == attr["SegmentAlgorithmType"]
            assert (
                str(segment.data.segmentedPropertyCategory)
                == attr["SegmentedPropertyCategoryCodeSequence"]["CodeMeaning"]
            )
            assert (
                segment.data.segmentedPropertyCategory.label
                == attr["SegmentedPropertyCategoryCodeSequence"]["CodeMeaning"]
            )
            assert (
                segment.data.segmentedPropertyCategory.code
                == attr["SegmentedPropertyCategoryCodeSequence"]["CodeValue"]
            )
            assert (
                segment.data.segmentedPropertyCategory.scheme
                == attr["SegmentedPropertyCategoryCodeSequence"][
                    "CodingSchemeDesignator"
                ]
            )
            assert (
                str(segment.data.segmentedPropertyType)
                == attr["SegmentedPropertyTypeCodeSequence"]["CodeMeaning"]
            )
            assert (
                segment.data.segmentedPropertyType.label
                == attr["SegmentedPropertyTypeCodeSequence"]["CodeMeaning"]
            )
            assert (
                segment.data.segmentedPropertyType.code
                == attr["SegmentedPropertyTypeCodeSequence"]["CodeValue"]
            )
            assert (
                segment.data.segmentedPropertyType.scheme
                == attr["SegmentedPropertyTypeCodeSequence"]["CodingSchemeDesignator"]
            )

            # check optional fields
            assert segment.data.hasSegmentedPropertyTypeModifier == (
                "SegmentedPropertyTypeModifierCodeSequence" in attr
            )
            assert segment.data.hasAnatomicRegion == (
                "AnatomicRegionModifierSequence" in attr
            )
            assert segment.data.hasAnatomicRegionModifier == (
                "AnatomicRegionModifierSequence" in attr
            )

            if segment.data.hasSegmentedPropertyTypeModifier:
                # NOTE: why ...CodeSequence and ...Sequence ?
                assert (
                    str(segment.data.segmentedPropertyTypeModifier)
                    == attr["SegmentedPropertyTypeModifierCodeSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.segmentedPropertyTypeModifier.label
                    == attr["SegmentedPropertyTypeModifierCodeSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.segmentedPropertyTypeModifier.code
                    == attr["SegmentedPropertyTypeModifierCodeSequence"]["CodeValue"]
                )
                assert (
                    segment.data.segmentedPropertyTypeModifier.scheme
                    == attr["SegmentedPropertyTypeModifierCodeSequence"][
                        "CodingSchemeDesignator"
                    ]
                )

            if segment.data.hasAnatomicRegion:
                assert (
                    str(segment.data.anatomicRegion)
                    == attr["AnatomicRegionSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.anatomicRegion.label
                    == attr["AnatomicRegionSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.anatomicRegion.code
                    == attr["AnatomicRegionSequence"]["CodeValue"]
                )
                assert (
                    segment.data.anatomicRegion.scheme
                    == attr["AnatomicRegionSequence"]["CodingSchemeDesignator"]
                )

            if segment.data.hasAnatomicRegionModifier:
                assert (
                    str(segment.data.anatomicRegionModifier)
                    == attr["AnatomicRegionModifierSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.anatomicRegionModifier.label
                    == attr["AnatomicRegionModifierSequence"]["CodeMeaning"]
                )
                assert (
                    segment.data.anatomicRegionModifier.code
                    == attr["AnatomicRegionModifierSequence"]["CodeValue"]
                )
                assert (
                    segment.data.anatomicRegionModifier.scheme
                    == attr["AnatomicRegionModifierSequence"]["CodingSchemeDesignator"]
                )

    def test_loaded_config(self):
        # get the raw config
        config = self.segimg.getExportedConfiguration()

        # check the config
        assert self.segimg.config == config

    def test_loaded_data(self):
        # get the raw config
        config = self.segimg.getExportedConfiguration()

        # check the data
        assert self.segimg.data.bodyPartExamined == config["BodyPartExamined"]
        assert (
            self.segimg.data.clinicalTrialCoordinatingCenterName
            == config["ClinicalTrialCoordinatingCenterName"]
        )
        assert self.segimg.data.clinicalTrialSeriesID == config["ClinicalTrialSeriesID"]
        assert (
            self.segimg.data.clinicalTrialTimePointID
            == config["ClinicalTrialTimePointID"]
        )
        assert self.segimg.data.contentCreatorName == config["ContentCreatorName"]
        assert self.segimg.data.instanceNumber == config["InstanceNumber"]
        assert self.segimg.data.seriesDescription == config["SeriesDescription"]
        assert self.segimg.data.seriesNumber == config["SeriesNumber"]


class TestSegimageWrite:
    def setup_class(self):
        # define output directory
        self.tmp_dir = TEST_DIR / "tmp"
        self.img_dir = TEST_DIR / "image"
        self.seg_dir = TEST_DIR / "seg"
        self.out_dir = TEST_DIR / "out"

        # create directories
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # initialize idc index client
        client = index.IDCClient()

        if not [f for f in os.listdir(self.img_dir) if f.endswith(".dcm")]:
            print("downloading image")
            client.download_dicom_series(
                "1.3.6.1.4.1.14519.5.2.1.4334.1501.680033973739971488930649469577",
                self.img_dir,
                quiet=False,
            )

        if not [f for f in os.listdir(self.seg_dir) if f.endswith(".dcm")]:
            print("downloading segment")
            client.download_dicom_series(
                "1.2.276.0.7230010.3.1.3.17436516.538020.1696968975.507837",
                self.seg_dir,
                quiet=False,
            )

        # check download and assign files
        sf = [f for f in os.listdir(self.seg_dir) if f.endswith(".dcm")]
        assert len(sf) == 1
        self.seg_file = self.seg_dir / sf[0]

        # get the segmentation files
        self.dseg_config_file = self.out_dir / "pydcmqi-meta.json"
        self.lung_seg_file = self.out_dir / "pydcmqi-1.nii.gz"
        self.tumor_seg_file = self.out_dir / "pydcmqi-2.nii.gz"

        # extract nifit files from segmentation if not already done
        if (
            not self.dseg_config_file.exists()
            or not self.lung_seg_file.exists()
            or not self.tumor_seg_file.exists()
        ):
            self.segimg.load(self.seg_file, output_dir=self.out_dir)

    def test_write(self):
        # NOTE: this test is not yet dynamic and only works with the specified dicomseg file.

        # load original config
        with Path.open(self.dseg_config_file) as f:
            config = json.load(f)

        # initialize a SegImage instance used in multiple tests
        segimg = SegImage(self.tmp_dir)

        # specify segimg data
        segimg.data.bodyPartExamined = "LUNG"
        segimg.data.clinicalTrialCoordinatingCenterName = "dcmqi"
        segimg.data.clinicalTrialSeriesID = "Session1"
        segimg.data.clinicalTrialTimePointID = "1"
        segimg.data.contentCreatorName = "BAMFHealth^AI"
        segimg.data.instanceNumber = "1"
        segimg.data.seriesDescription = "AIMI lung and FDG tumor AI segmentation"
        segimg.data.seriesNumber = "300"

        # add lung segment
        lung = segimg.new_segment()
        lung.data.label = "Lung"
        lung.data.description = "Lung"
        lung.data.rgb = (128, 174, 128)
        lung.data.labelID = 1
        lung.data.segmentAlgorithmName = "BAMF-Lung-FDG-PET-CT"
        lung.data.segmentAlgorithmType = "AUTOMATIC"
        lung.data.segmentedPropertyCategory = (
            "Anatomical Structure",
            "123037004",
            "SCT",
        )
        lung.data.segmentedPropertyType = ("Lung", "39607008", "SCT")
        lung.data.segmentedPropertyTypeModifier = ("Right and left", "51440002", "SCT")

        lung.setFile(self.lung_seg_file, labelID=1)

        # add tumor segment
        tumor = segimg.new_segment()
        tumor.data.label = "FDG-Avid Tumor"
        tumor.data.description = "FDG-Avid Tumor"
        tumor.data.rgb = (174, 41, 14)
        tumor.data.labelID = 2
        tumor.data.segmentAlgorithmName = "BAMF-Lung-FDG-PET-CT"
        tumor.data.segmentAlgorithmType = "AUTOMATIC"
        tumor.data.segmentedPropertyCategory = ("Radiologic Finding", "C35869", "NCIt")
        tumor.data.segmentedPropertyType.label = "FDG-Avid Tumor"
        tumor.data.segmentedPropertyType.code = "C168968"
        tumor.data.segmentedPropertyType.scheme = "NCIt"
        tumor.data.anatomicRegion = lung.data.segmentedPropertyType
        tumor.data.anatomicRegionModifier = lung.data.segmentedPropertyTypeModifier

        tumor.setFile(self.tumor_seg_file, labelID=2)

        # check config
        assert segimg.config == config

        # write file
        output_file = self.out_dir / "test.seg.dcm"
        segimg.write(output_file, self.img_dir, allow_overwrite=True)

        # check the file was created
        assert output_file.exists()
