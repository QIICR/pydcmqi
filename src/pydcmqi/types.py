from typing import Literal, TypedDict


class TripletDict(TypedDict):
    """A dictionary defining the keys for any generic DICOM code triplet."""

    CodeMeaning: str
    CodeValue: str
    CodingSchemeDesignator: str


SegmentAlgorithmType = Literal["AUTOMATIC", "MANUAL", "SEMIAUTOMATIC"]


class _SegmentDictRequired(TypedDict):
    """Required fields per dcmqi seg-schema.json."""

    labelID: int
    SegmentedPropertyCategoryCodeSequence: TripletDict
    SegmentedPropertyTypeCodeSequence: TripletDict
    SegmentAlgorithmType: str


class SegmentDict(_SegmentDictRequired, total=False):
    """A dictionary defining the keys for a single segment within a segmentation image."""

    SegmentLabel: str
    SegmentDescription: str
    SegmentAlgorithmName: str
    recommendedDisplayRGBValue: list[int]
    RecommendedDisplayCIELabValue: list[int]
    SegmentedPropertyTypeModifierCodeSequence: TripletDict
    AnatomicRegionSequence: TripletDict
    AnatomicRegionModifierSequence: TripletDict
    TrackingIdentifier: str
    TrackingUniqueIdentifier: str


class SegImageDict(TypedDict, total=False):
    """A dictionary defining the keys for a segmentation image."""

    BodyPartExamined: str
    ClinicalTrialCoordinatingCenterName: str
    ClinicalTrialSeriesID: str
    ClinicalTrialTimePointID: str
    ContentCreatorName: str
    ContentLabel: str
    ContentDescription: str
    InstanceNumber: str
    SeriesDescription: str
    SeriesNumber: str
    segmentAttributes: list[list[SegmentDict]]
