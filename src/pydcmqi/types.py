from __future__ import annotations

from typing import TypedDict


class TripletDict(TypedDict):
    """
    A dictionary defining the keys for any generic triplet.
    """

    CodeMeaning: str
    CodeValue: str
    CodingSchemeDesignator: str


class SegmentDict(TypedDict):
    """
    A dictionary defining the keys for a single segment within a segmentation image.
    """

    labelID: int
    SegmentLabel: str
    SegmentDescription: str
    SegmentAlgorithmName: str
    SegmentAlgorithmType: str
    recommendedDisplayRGBValue: list[int]
    SegmentedPropertyCategoryCodeSequence: TripletDict
    SegmentedPropertyTypeCodeSequence: TripletDict


class SegImageDict(TypedDict):
    """
    A dictionary defining the keys for a segmentation image.
    """

    BodyPartExamined: str
    ClinicalTrialCoordinatingCenterName: str
    ClinicalTrialSeriesID: str
    ClinicalTrialTimePointID: str
    ContentCreatorName: str
    InstanceNumber: str
    SeriesDescription: str
    SeriesNumber: str
    segmentAttributes: list[list[SegmentDict]]
