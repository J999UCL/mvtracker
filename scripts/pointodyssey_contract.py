"""Fixed source-data contract for the PointOdyssey Track A collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple


VIEW_IDS = (0, 1, 2, 3)
POINT_COUNT = 2600
SOURCE_HEIGHT = 1080
SOURCE_WIDTH = 1920

SOURCE_SUBROOTS = {
    "raw": Path("raw_fp32_tarzst/tapvid3d_raw_fp32/track_a_train"),
    "short": Path(
        "shortform_zip/"
        "mixamo_camera_super_aggro_v1_fixed_intrinsics_1080p_s16_p2600_viewunion/"
        "track_a_train"
    ),
    "long": Path("longform_tar/tapvid3d_v5_zarr_combined_20260627/track_a_train"),
}

SOURCE_FRAME_COUNTS = {"raw": 120, "short": 61, "long": 2000}
SOURCE_FPS = {"raw": 30, "short": 15, "long": 15}


@dataclass(frozen=True)
class SourceAssignment:
    split: str
    layout: str
    sequence: str
    environment_family: str
    first_scene_id: int


ASSIGNMENTS = (
    SourceAssignment("train", "raw", "candidate_empty_office", "empty_office", 0),
    SourceAssignment("train", "raw", "indoor_00_pawn_shop_manual", "indoor_00_pawn_shop", 1),
    SourceAssignment("train", "raw", "indoor_09_barbershop_manual", "indoor_09_barbershop", 2),
    SourceAssignment("train", "raw", "warehouse_ledge_manual", "warehouse", 3),
    SourceAssignment("train", "short", "indoor_00_pawn_shop", "indoor_00_pawn_shop", 4),
    SourceAssignment("train", "short", "indoor_05_modern_kitchen", "indoor_05_modern_kitchen", 5),
    SourceAssignment("train", "short", "indoor_06_office", "indoor_06_office", 6),
    SourceAssignment("train", "short", "outdoor_00_namaqualand", "outdoor_00_namaqualand", 7),
    SourceAssignment("train", "short", "outdoor_07_forest_road", "outdoor_07_forest_road", 8),
    SourceAssignment("train", "short", "outdoor_08_seacliff_beach", "outdoor_08_seacliff_beach", 9),
    SourceAssignment("train", "long", "candidate_empty_office", "empty_office", 10),
    SourceAssignment("train", "long", "candidate_parking", "candidate_parking", 27),
    SourceAssignment("train", "long", "candidate_warehouse", "warehouse", 44),
    SourceAssignment("train", "long", "og_parking_lot", "og_parking_lot", 61),
    SourceAssignment("validation", "raw", "indoor_01_classroom_manual", "indoor_01_classroom", 0),
    SourceAssignment("validation", "raw", "outdoor_02_hidden_alley_manual", "outdoor_02_hidden_alley", 1),
    SourceAssignment("validation", "short", "indoor_01_classroom", "indoor_01_classroom", 2),
    SourceAssignment("test", "raw", "indoor_04_modern_loft_manual", "indoor_04_modern_loft", 0),
    SourceAssignment("test", "raw", "outdoor_06_city_scene_manual", "outdoor_06_city_scene", 1),
    SourceAssignment("test", "short", "indoor_04_modern_loft", "indoor_04_modern_loft", 2),
)


def unique_source_keys() -> Iterator[Tuple[str, str]]:
    """Yield each approved ``(layout, sequence)`` pair once, in split-plan order."""
    seen = set()
    for assignment in ASSIGNMENTS:
        key = (assignment.layout, assignment.sequence)
        if key not in seen:
            seen.add(key)
            yield key
