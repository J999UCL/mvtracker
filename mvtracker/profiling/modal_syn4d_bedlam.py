"""Selective BEDLAM2 dependencies for the pinned Syn4D scene selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RemoteTarMember:
    name: str
    data_offset: int
    size: int
    mode: int
    mtime: int


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
    *,
    byte_range: tuple[int, int] | None = None,
):
    headers = {"Accept-Encoding": "identity"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    response = session.post(
        BEDLAM_URL,
        params={"domain": "bedlam2", "resume": "1", "sfile": sfile},
        data={"username": email, "password": password},
        headers=headers,
        stream=True,
        timeout=(30, 300),
    )
    response.raise_for_status()
    if byte_range is not None:
        start, end = byte_range
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"BEDLAM did not honor byte range {start}-{end} for {sfile}"
            )
        expected = f"bytes {start}-{end}/"
        if not response.headers.get("Content-Range", "").startswith(expected):
            response.close()
            raise RuntimeError(f"invalid BEDLAM Content-Range for {sfile}")
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


def _parse_octal(field: bytes) -> int:
    stripped = field.rstrip(b"\0 ").lstrip(b" ")
    return int(stripped or b"0", 8)


def parse_tar_header(block: bytes, offset: int) -> RemoteTarMember | None:
    if len(block) != 512:
        raise RuntimeError("short remote TAR header")
    if block == bytes(512):
        return None
    stored_checksum = _parse_octal(block[148:156])
    observed_checksum = sum(block[:148]) + 8 * ord(" ") + sum(block[156:])
    if stored_checksum != observed_checksum:
        raise RuntimeError(f"invalid remote TAR header checksum at offset {offset}")
    name = block[:100].split(b"\0", 1)[0].decode("utf-8")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8")
    if prefix:
        name = f"{prefix}/{name}"
    return RemoteTarMember(
        name=name,
        data_offset=offset + 512,
        size=_parse_octal(block[124:136]),
        mode=_parse_octal(block[100:108]),
        mtime=_parse_octal(block[136:148]),
    )


def scan_remote_tar(session, sfile: str, email: str, password: str) -> dict[str, RemoteTarMember]:
    members: dict[str, RemoteTarMember] = {}
    offset = 0
    for _ in range(20_000):
        response = _request(
            session, sfile, email, password, byte_range=(offset, offset + 511)
        )
        try:
            block = response.raw.read(512)
        finally:
            response.close()
        member = parse_tar_header(block, offset)
        if member is None:
            return members
        members[member.name] = member
        offset = member.data_offset + ((member.size + 511) // 512) * 512
    raise RuntimeError(f"remote TAR index exceeded limit: {sfile}")


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
        remote_members = scan_remote_tar(session, sfile, email, password)
        missing = sorted(set(required_members).difference(remote_members))
        if missing:
            raise RuntimeError(f"{archive_name} lacks required members: {missing}")
        selected = [remote_members[name] for name in sorted(required_members)]

        with tempfile.TemporaryDirectory(prefix=f"{archive_name}-") as directory:
            local_tar = Path(directory) / f"{archive_name}.tar"
            with tarfile.open(local_tar, "w") as output:
                for member in selected:
                    response = _request(
                        session,
                        sfile,
                        email,
                        password,
                        byte_range=(
                            member.data_offset,
                            member.data_offset + member.size - 1,
                        ),
                    )
                    try:
                        info = tarfile.TarInfo(member.name)
                        info.size = member.size
                        info.mode = member.mode
                        info.mtime = member.mtime
                        output.addfile(info, response.raw)
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
        body_future = pool.submit(
            download_body_motions, data_root, motions, email, password
        )
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
    archive_name = "b2_clothing_npz_000"
    with requests.Session() as session:
        clothing_index = scan_remote_tar(
            session,
            f"{CLOTHING_SOURCE_ROOT}/{archive_name}.tar",
            email,
            password,
        )
        body_header = _request(
            session, BODY_SOURCE, email, password, byte_range=(0, 511)
        )
        try:
            parse_tar_header(body_header.raw.read(512), 0)
        finally:
            body_header.close()
    needed = required[archive_name]
    return {
        "sequence_count": plan["sequence_count"],
        "motion_count": len(plan["motions"]),
        "clothing_archive_count": len(required),
        "clothing_member_count": sum(len(v) for v in required.values()),
        "probe_archive": archive_name,
        "probe_archive_members": len(clothing_index),
        "probe_required_members": len(needed),
        "probe_required_bytes": sum(clothing_index[name].size for name in needed),
        "range_requests_supported": True,
    }
