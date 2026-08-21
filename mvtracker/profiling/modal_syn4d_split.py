"""Fixed Syn4D environment/split manifest and dependency contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SYN4D_REPO_ID = "Syn4D/Syn4D"
SYN4D_REVISION = "181c6a2da735b216826ab9411b08e0d1d225aced"
SYN4D_SUBSET = "data/syn4d_v1_stride_1"
MAPPING_FILENAME = f"{SYN4D_SUBSET}/sequence_to_asset_mapping.csv"
ARCHIVE_BYTES = {
    "bigoffice_v1": 21_116_734_665,
    "scifiroom_bald": 19_974_734_231,
    "cave_group": 16_483_535_123,
    "flying_group": 31_462_981_644,
    "desert_bald": 26_119_416_514,
    "space_bald": 22_450_393_703,
    "brushify_bald": 28_349_560_577,
    "countryside": 94_504_965_685,
    "lab_bald": 16_004_849_235,
    "post_bald": 18_476_889_064,
    "castle": 100_737_800_531,
    "winter": 60_688_089_858,
    "hospital": 61_153_460_562,
    "temple_group": 13_287_559_476,
    "planet_bald": 16_362_782_150,
    "cyber_bald": 27_539_280_782,
    "warehouse_group_static": 12_056_261_820,
    "winter_group_static": 16_281_604_483,
    "middleeast_bald": 16_522_064_454,
    "japenese_bald": 19_650_221_315,
}
TRAIN_ENVIRONMENTS = (
    ("bigoffice_v1", "seq_000004"),
    ("scifiroom_bald", "seq_000010"),
    ("cave_group", "seq_000008"),
    ("flying_group", "seq_000015"),
    ("desert_bald", "seq_000012"),
    ("space_bald", "seq_000003"),
    ("brushify_bald", "seq_000005"),
    ("countryside", "seq_000019"),
    ("lab_bald", "seq_000017"),
    ("post_bald", "seq_000002"),
    ("castle", "seq_000007"),
    ("winter", "seq_000015"),
    ("hospital", "seq_000003"),
    ("temple_group", "seq_000011"),
    ("planet_bald", "seq_000018"),
    ("cyber_bald", "seq_000001"),
)
VALIDATION_ENVIRONMENTS = (
    ("warehouse_group_static", "seq_000018"),
    ("winter_group_static", "seq_000009"),
    ("middleeast_bald", "seq_000009"),
    ("japenese_bald", "seq_000009"),
)
SPLIT_MANIFEST = tuple(
    {"environment": environment, "sequence": sequence, "split": "train"}
    for environment, sequence in TRAIN_ENVIRONMENTS
) + tuple(
    {"environment": environment, "sequence": sequence, "split": "validation"}
    for environment, sequence in VALIDATION_ENVIRONMENTS
)
SPLIT_ROOTS = {
    "train": Path("datasets/syn4d-mvtracker/train"),
    "validation": Path("datasets/syn4d-mvtracker/validation"),
}
ARCHIVE_ROOT = Path("datasets/syn4d")
SHARED_METADATA_ROOT = Path("datasets/syn4d/v1-stride1-12train-4validation/metadata")
BODY_ROOT = SHARED_METADATA_ROOT / "b2_motions_npz_training/motions_npz_training"
CLOTHING_ROOT = SHARED_METADATA_ROOT / "b2_assetdata_download/clothing/npz"
OBJECT_ROOT = SHARED_METADATA_ROOT / "new_weight_bone"
SMPLX_ADDON_PARTS = Path("datasets/syn4d/temple_group/private/smplx_addon_parts")
LEGACY_ARCHIVES = {
    "lab_bald": ARCHIVE_ROOT / "lab_bald/source/lab_bald.tar.zst",
    "temple_group": ARCHIVE_ROOT / "temple_group/source/temple_group.tar.zst",
}
SHARD_A_ENVIRONMENTS = (
    "castle", "winter", "brushify_bald", "desert_bald", "bigoffice_v1",
    "scifiroom_bald", "middleeast_bald", "cave_group", "winter_group_static",
    "temple_group",
)
SHARD_B_ENVIRONMENTS = (
    "countryside", "hospital", "flying_group", "cyber_bald", "space_bald",
    "japenese_bald", "post_bald", "planet_bald", "lab_bald", "warehouse_group_static",
)


@dataclass(frozen=True)
class EnvironmentJob:
    environment: str
    sequence: str
    split: str
    archive_bytes: int

    @property
    def output_root(self) -> Path:
        return SPLIT_ROOTS[self.split]

    @property
    def output_name(self) -> str:
        return f"{self.environment}__{self.sequence}"


def jobs() -> tuple[EnvironmentJob, ...]:
    result = tuple(
        EnvironmentJob(
            environment=row["environment"],
            sequence=row["sequence"],
            split=row["split"],
            archive_bytes=ARCHIVE_BYTES[row["environment"]],
        )
        for row in SPLIT_MANIFEST
    )
    if len(result) != 20 or sum(job.split == "train" for job in result) != 16:
        raise RuntimeError("fixed Syn4D split must contain 16 train and 4 validation jobs")
    return result


def jobs_for(environments: Iterable[str]) -> tuple[EnvironmentJob, ...]:
    selected = set(environments)
    result = tuple(job for job in jobs() if job.environment in selected)
    if len(result) != len(selected):
        raise ValueError(f"unknown Syn4D environments: {sorted(selected - {job.environment for job in jobs()})}")
    return result


def dependency_plan(mapping_path: Path) -> tuple[dict[str, object], ...]:
    """Resolve one body and three objects for each fixed environment job."""

    from mvtracker.preprocessing.syn4d import sequence_dependencies

    return tuple(
        {
            "job": job,
            "dependencies": sequence_dependencies(
                mapping_path, scene=job.environment, sequence_base=job.sequence
            ),
        }
        for job in jobs()
    )


__all__ = [
    "ARCHIVE_BYTES", "ARCHIVE_ROOT", "BODY_ROOT", "CLOTHING_ROOT",
    "EnvironmentJob", "LEGACY_ARCHIVES", "MAPPING_FILENAME", "OBJECT_ROOT",
    "SHARD_A_ENVIRONMENTS", "SHARD_B_ENVIRONMENTS", "SHARED_METADATA_ROOT",
    "SPLIT_MANIFEST", "SPLIT_ROOTS", "SMPLX_ADDON_PARTS", "SYN4D_REPO_ID",
    "SYN4D_REVISION", "SYN4D_SUBSET", "TRAIN_ENVIRONMENTS", "VALIDATION_ENVIRONMENTS",
    "dependency_plan", "jobs", "jobs_for",
]
