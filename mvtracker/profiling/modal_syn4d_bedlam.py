"""Selective BEDLAM2 dependencies for the pinned Syn4D scene selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from typing import Callable, Iterable


BEDLAM_URL = "https://download.is.tue.mpg.de/download.php"
BODY_SOURCE = "ground_truth/motions/b2_motions_npz_training.tar"
BODY_SOURCE_BYTES = 2_863_616_000
BODY_SOURCE_SHA256 = "a87c9b201d003125864b0cf4960e37bdb4a58f2162d309c9949aa3d5c5ab7db9"
CLOTHING_SOURCE_ROOT = "assets/clothing/npz"
DEPENDENCY_ROOT = Path(
    "datasets/syn4d/v1-stride1-12train-4validation/metadata"
)
BODY_ROOT = DEPENDENCY_ROOT / "b2_motions_npz_training/motions_npz_training"
CLOTHING_ROOT = DEPENDENCY_ROOT / "b2_assetdata_download/clothing/npz"
SYN4D_MAPPING_PATH = "data/syn4d_v1_stride_5/sequence_to_asset_mapping.csv"
SYN4D_REPO_ID = "Syn4D/Syn4D"
SYN4D_REVISION = "181c6a2da735b216826ab9411b08e0d1d225aced"

SELECTED_SCENES = frozenset(
    {
        "arena",
        "bigoffice_v1",
        "brushify_bald",
        "cave_group",
        "desert_bald",
        "flying_group",
        "genshin_group",
        "hallway",
        "hospital",
        "lab_bald",
        "post_bald",
        "scifiroom_bald",
        "temple_group",
        "village",
        "warehouse_group_static",
        "winter",
    }
)
EXCLUDED_SEQUENCES = frozenset(
    {
        ("cave_group", "seq_000004"),
        ("village", "seq_000001"),
        ("village", "seq_000014"),
        ("hallway", "seq_000007"),
        ("hallway", "seq_000013"),
        ("hospital", "seq_000015"),
    }
)


def build_dependency_plan(
    mapping_csv: str, archive_map: dict[str, list[str]]
) -> dict[str, object]:
    """Join the Syn4D mapping to BEDLAM's clothing archive map."""

    sequence_bases: set[tuple[str, str]] = set()
    motions: set[str] = set()
    for row in csv.DictReader(io.StringIO(mapping_csv)):
        scene = row["scene"]
        if scene not in SELECTED_SCENES:
            continue
        sequence_base = row["sequence_name"].rsplit("_", 1)[0]
        if (scene, sequence_base) in EXCLUDED_SEQUENCES:
            continue
        sequence_bases.add((scene, sequence_base))
        if row["asset_type"] == "bedlam2_body":
            motions.add(row["asset"])

    member_to_archive = {
        member: archive for archive, members in archive_map.items() for member in members
    }
    required_members: dict[str, list[str]] = {}
    for motion in sorted(motions):
        subject, animation = motion.rsplit("_", 1)
        member = f"{subject}/{animation}/{animation}.npz"
        archive = member_to_archive.get(member)
        if archive is None:
            raise RuntimeError(f"BEDLAM archive map has no clothing member {member}")
        required_members.setdefault(archive, []).append(member)

    if (len(sequence_bases), len(motions), len(required_members)) != (299, 298, 54):
        raise RuntimeError(
            "unexpected Syn4D BEDLAM plan counts: "
            f"sequences={len(sequence_bases)} motions={len(motions)} "
            f"archives={len(required_members)}"
        )
    return {
        "sequence_count": len(sequence_bases),
        "motions": sorted(motions),
        "required_members": {
            archive: sorted(members)
            for archive, members in sorted(required_members.items())
        },
    }


def _request(
    session,
    sfile: str,
    email: str,
    password: str,
):
    response = session.post(
        BEDLAM_URL,
        params={"domain": "bedlam2", "resume": "1", "sfile": sfile},
        data={"username": email, "password": password},
        headers={"Accept-Encoding": "identity"},
        stream=True,
        timeout=(30, 300),
    )
    response.raise_for_status()
    return response


def download_text(session, sfile: str, email: str, password: str) -> str:
    response = _request(session, sfile, email, password)
    try:
        payload = response.content
    finally:
        response.close()
    if payload.lstrip().lower().startswith(b"<!doctype html"):
        raise RuntimeError("BEDLAM credentials were rejected")
    return payload.decode("utf-8")


def copy_selected_tar_members(
    source_stream, destination: Path, required_members: Iterable[str]
) -> dict[str, int]:
    """Stream one source TAR and retain only the requested NPZ members."""

    required = set(required_members)
    copied: dict[str, int] = {}
    with tarfile.open(fileobj=source_stream, mode="r|") as source, tarfile.open(
        destination, "w"
    ) as output:
        for member in source:
            if not member.isfile() or member.name not in required:
                continue
            member_stream = source.extractfile(member)
            if member_stream is None:
                raise RuntimeError(f"cannot read clothing member {member.name}")
            with member_stream:
                output.addfile(member, member_stream)
            copied[member.name] = member.size
    missing = sorted(required.difference(copied))
    if missing:
        raise RuntimeError(f"source clothing TAR lacks required members: {missing}")
    return copied


def _validate_sparse_tar(path: Path, expected: Iterable[str]) -> dict[str, int]:
    expected_names = sorted(expected)
    with tarfile.open(path, "r") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if [member.name for member in members] != expected_names:
            raise RuntimeError(f"sparse clothing TAR member mismatch: {path}")
        for member in members:
            stream = archive.extractfile(member)
            if stream is None or stream.read(4) != b"PK\x03\x04":
                raise RuntimeError(f"clothing member is not NPZ: {member.name}")
    return {member.name: member.size for member in members}


def download_sparse_clothing_tar(
    data_root: Path,
    archive_name: str,
    required_members: list[str],
    email: str,
    password: str,
) -> dict[str, object]:
    import requests

    destination = Path(data_root) / CLOTHING_ROOT / f"{archive_name}.tar"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        sizes = _validate_sparse_tar(destination, required_members)
        return {
            "archive": archive_name,
            "status": "skipped_verified",
            "size_bytes": destination.stat().st_size,
            "member_sizes": sizes,
        }

    sfile = f"{CLOTHING_SOURCE_ROOT}/{archive_name}.tar"
    with requests.Session() as session:
        response = _request(session, sfile, email, password)
        source_size = int(response.headers.get("Content-Length", 0))
        if response.headers.get("Content-Type") != "application/octet-stream":
            response.close()
            raise RuntimeError(f"BEDLAM did not return clothing TAR {archive_name}")
        with tempfile.TemporaryDirectory(prefix=f"{archive_name}-") as directory:
            local_tar = Path(directory) / f"{archive_name}.tar"
            try:
                copy_selected_tar_members(response.raw, local_tar, required_members)
            finally:
                response.close()
            sizes = _validate_sparse_tar(local_tar, required_members)
            partial = destination.with_name(f".{destination.name}.partial")
            shutil.copyfile(local_tar, partial)
            partial.replace(destination)

    return {
        "archive": archive_name,
        "status": "downloaded",
        "size_bytes": destination.stat().st_size,
        "source_size_bytes": source_size,
        "member_sizes": sizes,
    }


def download_body_motions(
    data_root: Path,
    motions: list[str],
    email: str,
    password: str,
) -> dict[str, object]:
    import requests

    destination_root = Path(data_root) / BODY_ROOT
    destination_root.mkdir(parents=True, exist_ok=True)
    existing = [path for path in destination_root.glob("*.npz") if path.stat().st_size]
    if len(existing) == len(motions) and {path.stem for path in existing} == set(motions):
        return {
            "status": "skipped_verified",
            "file_count": len(existing),
            "size_bytes": sum(path.stat().st_size for path in existing),
        }

    with tempfile.TemporaryDirectory(prefix="bedlam-body-") as directory:
        source = Path(directory) / "b2_motions_npz_training.tar"
        digest = hashlib.sha256()
        size = 0
        with requests.Session() as session:
            response = _request(session, BODY_SOURCE, email, password)
            try:
                with source.open("wb") as output:
                    for chunk in response.iter_content(8 << 20):
                        if not chunk:
                            continue
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            finally:
                response.close()
        if size != BODY_SOURCE_BYTES or digest.hexdigest() != BODY_SOURCE_SHA256:
            raise RuntimeError(
                f"BEDLAM body source verification failed: bytes={size} sha256={digest.hexdigest()}"
            )

        with tarfile.open(source, "r") as archive:
            for motion in motions:
                member = archive.getmember(f"motions_npz_training/{motion}.npz")
                input_stream = archive.extractfile(member)
                if input_stream is None:
                    raise RuntimeError(f"cannot read BEDLAM body motion {motion}")
                destination = destination_root / f"{motion}.npz"
                partial = destination.with_name(f".{destination.name}.partial")
                with input_stream, partial.open("wb") as output:
                    shutil.copyfileobj(input_stream, output, 8 << 20)
                partial.replace(destination)

    files = list(destination_root.glob("*.npz"))
    return {
        "status": "downloaded",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def materialize_dependencies(
    data_root: Path,
    mapping_csv: str,
    archive_map: dict[str, list[str]],
    source_checksums: str,
    email: str,
    password: str,
    *,
    workers: int = 4,
    progress: Callable[[dict[str, object]], None] = print,
    commit: Callable[[], None] = lambda: None,
) -> dict[str, object]:
    plan = build_dependency_plan(mapping_csv, archive_map)
    motions = plan["motions"]
    required_members = plan["required_members"]
    assert isinstance(motions, list) and isinstance(required_members, dict)

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        body_future = pool.submit(
            download_body_motions, data_root, motions, email, password
        )
        futures = {
            pool.submit(
                download_sparse_clothing_tar,
                data_root,
                archive,
                members,
                email,
                password,
            ): archive
            for archive, members in required_members.items()
        }
        for future in as_completed([*futures, body_future]):
            result = future.result()
            results.append(result)
            commit()
            progress({"event": "dependency_complete", **result})

    clothing_root = Path(data_root) / CLOTHING_ROOT
    pruned_map = {
        archive: required_members[archive] for archive in sorted(required_members)
    }
    (clothing_root / "archive_map.json").write_text(
        json.dumps(pruned_map, indent=2, sort_keys=True) + "\n"
    )
    required_filenames = {f"{archive}.tar" for archive in required_members}
    checksum_lines = [
        line
        for line in source_checksums.splitlines()
        if line.split() and Path(line.split()[-1]).name in required_filenames
    ]
    (clothing_root / "source_archives.xxh128").write_text(
        "\n".join(checksum_lines) + "\n"
    )

    body_result = next(result for result in results if "file_count" in result)
    clothing_results = sorted(
        (result for result in results if "archive" in result),
        key=lambda result: str(result["archive"]),
    )
    manifest = {
        "format": "syn4d-selective-bedlam2-v1",
        "syn4d_repo_id": SYN4D_REPO_ID,
        "syn4d_revision": SYN4D_REVISION,
        "sequence_count": plan["sequence_count"],
        "body_motion_count": len(motions),
        "clothing_member_count": sum(len(v) for v in required_members.values()),
        "clothing_archive_count": len(required_members),
        "body": body_result,
        "clothing": clothing_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = Path(data_root) / DEPENDENCY_ROOT / "bedlam2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    commit()
    return manifest


def probe_dependencies(
    mapping_csv: str,
    archive_map: dict[str, list[str]],
    email: str,
    password: str,
) -> dict[str, object]:
    import requests

    plan = build_dependency_plan(mapping_csv, archive_map)
    required = plan["required_members"]
    assert isinstance(required, dict)

    def source_size(archive_name: str) -> int:
        with requests.Session() as session:
            response = _request(
                session,
                f"{CLOTHING_SOURCE_ROOT}/{archive_name}.tar",
                email,
                password,
            )
            try:
                if response.headers.get("Content-Type") != "application/octet-stream":
                    raise RuntimeError(f"BEDLAM did not return {archive_name}")
                return int(response.headers["Content-Length"])
            finally:
                response.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        source_sizes = dict(zip(required, pool.map(source_size, required)))
    return {
        "sequence_count": plan["sequence_count"],
        "motion_count": len(plan["motions"]),
        "clothing_archive_count": len(required),
        "clothing_member_count": sum(len(v) for v in required.values()),
        "source_clothing_bytes": sum(source_sizes.values()),
        "smallest_source_archive_bytes": min(source_sizes.values()),
        "largest_source_archive_bytes": max(source_sizes.values()),
        "range_requests_supported": False,
        "streaming_sparse_extraction": True,
    }
