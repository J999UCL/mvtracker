from collections.abc import Sequence


def select_scene_names(
    available: Sequence[str],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Filter scene names while preserving the dataset's canonical order."""
    available = list(available)
    if len(available) != len(set(available)):
        raise ValueError("available scene names must be unique")

    include_set = _validated_selection("include_scene_ids", include, available)
    exclude_set = _validated_selection("exclude_scene_ids", exclude, available)
    if include_set is not None and include_set & exclude_set:
        overlap = sorted(include_set & exclude_set)
        raise ValueError(f"scene IDs cannot be both included and excluded: {overlap}")

    return [
        scene_name
        for scene_name in available
        if (include_set is None or scene_name in include_set)
        and scene_name not in exclude_set
    ]


def _validated_selection(
    label: str,
    values: Sequence[str] | None,
    available: Sequence[str],
) -> set[str] | None:
    if values is None:
        return None
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    missing = sorted(set(values) - set(available))
    if missing:
        raise ValueError(f"{label} contains unknown scene IDs: {missing}")
    return set(values)
