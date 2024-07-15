from __future__ import annotations

from typing import TypedDict


class TripletDict(TypedDict):
    CodeMeaning: str
    CodeValue: str
    CodingSchemeDesignator: str


class SegmentDict(TypedDict):
    labelID: int
    SegmentLabel: str
    SegmentDescription: str
    SegmentAlgorithmName: str
    SegmentAlgorithmType: str
    recommendedDisplayRGBValue: list[int]
    SegmentedPropertyCategoryCodeSequence: TripletDict
    SegmentedPropertyTypeCodeSequence: TripletDict


class SegImageDict(TypedDict):
    BodyPartExamined: str
    ClinicalTrialCoordinatingCenterName: str
    ClinicalTrialSeriesID: str
    ClinicalTrialTimePointID: str
    ContentCreatorName: str
    InstanceNumber: str
    SeriesDescription: str
    SeriesNumber: str
    segmentAttributes: list[list[SegmentDict]]
