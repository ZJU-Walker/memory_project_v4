"""Derive the v4 per-episode fact labels from the frozen v36 manifest (V4_PLAN.md §2.5).

No relabeling and no new data: every fact is a deterministic function of manifest fields that
were 3x side-verified for the v36 freeze. The bin scene always contains exactly two objects,
one per bin, so:

    prompted object  -> located_in -> target_side
    other object     -> located_in -> opposite(target_side)

Fact-slot vocabulary (static, index = slot id):
    slot 0: (banana, located_in)
    slot 1: (grey_pepper_box, located_in)
    slots 2..memory_fact_slots-1: unpopulated (label `unknown`, observable nowhere)

Target vocabulary (must match Pi0Config.memory_fact_targets=3):
    0 = left_bin, 1 = right_bin, 2 = unknown

The output sidecar is create-only, self-hashed, and pins the SHA-256 of the manifest it was
derived from, so Gate A can authenticate the derivation exactly like every other frozen input.
"""

import argparse
import dataclasses
import hashlib
import json
import pathlib

SCHEMA_VERSION = "openpi.v4.fact-labels.v1"
FACT_SLOTS = (
    {"slot": 0, "entity": "banana", "relation": "located_in"},
    {"slot": 1, "entity": "grey_pepper_box", "relation": "located_in"},
)
TARGET_VOCAB = ("left_bin", "right_bin", "unknown")
UNKNOWN = TARGET_VOCAB.index("unknown")
_SIDE_TO_TARGET = {"left": TARGET_VOCAB.index("left_bin"), "right": TARGET_VOCAB.index("right_bin")}
_OPPOSITE = {"left": "right", "right": "left"}
_KNOWN_OBJECTS = tuple(spec["entity"] for spec in FACT_SLOTS)


@dataclasses.dataclass(frozen=True)
class EpisodeFacts:
    stable_id: str
    split: str
    fact_targets: tuple[int, ...]  # one target id per fact slot, aligned with FACT_SLOTS


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def derive_episode_facts(episode: dict) -> EpisodeFacts:
    """Facts for one included manifest episode. Fails loudly on any unknown vocabulary."""
    prompted = episode["object"]
    side = episode["target_side"]
    if prompted not in _KNOWN_OBJECTS:
        raise ValueError(f"{episode['stable_id']}: unknown object {prompted!r}.")
    if side not in _OPPOSITE:
        raise ValueError(f"{episode['stable_id']}: unknown target_side {side!r}.")
    targets = []
    for spec in FACT_SLOTS:
        object_side = side if spec["entity"] == prompted else _OPPOSITE[side]
        targets.append(_SIDE_TO_TARGET[object_side])
    return EpisodeFacts(
        stable_id=episode["stable_id"],
        split=episode["split"],
        fact_targets=tuple(targets),
    )


def build_fact_labels(manifest_path: pathlib.Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("review_status") != "frozen":
        raise ValueError(f"manifest {manifest_path} is not frozen (review_status={manifest.get('review_status')!r}).")
    episodes = [e for e in manifest["episodes"] if e.get("include")]
    if not episodes:
        raise ValueError("the manifest contains no included episodes.")
    records = {}
    for episode in episodes:
        facts = derive_episode_facts(episode)
        if facts.stable_id in records:
            raise ValueError(f"duplicate stable_id {facts.stable_id!r}.")
        records[facts.stable_id] = {
            "split": facts.split,
            "fact_targets": list(facts.fact_targets),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "dataset_version": manifest.get("dataset_version"),
        "fact_slots": [dict(spec) for spec in FACT_SLOTS],
        "target_vocab": list(TARGET_VOCAB),
        "unknown_target": UNKNOWN,
        "num_episodes": len(records),
        "episodes": records,
    }
    body = _canonical_dumps(payload)
    payload["content_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("../data/0830_0831_episode_manifest_v36_frozen.json"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("../data/v4_fact_labels_0830_0831.json"),
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing sidecar")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists (create-only; pass --force to rebuild).")
    payload = build_fact_labels(args.manifest)
    args.output.write_text(_canonical_dumps(payload))
    per_split: dict[str, int] = {}
    for record in payload["episodes"].values():
        per_split[record["split"]] = per_split.get(record["split"], 0) + 1
    print(f"wrote {args.output} ({payload['num_episodes']} episodes, splits={per_split})")
    print(f"source manifest sha256: {payload['source_manifest_sha256']}")
    print(f"content sha256: {payload['content_sha256']}")


if __name__ == "__main__":
    main()
