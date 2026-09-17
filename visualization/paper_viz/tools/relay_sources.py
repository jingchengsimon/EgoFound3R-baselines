"""Relay the frozen GT/baseline predictions into the renderer's per-window layout.

The 2D renderer reads one npz per method and 60-frame window, laid out as
``<src-dir>/<cache_id>/{gt,wilor,pad_hand,egoforce,reviv4d,dyn_hamr}.npz``.  Those
files live in the frozen evaluation roots registered in
``source_alignment_auxmethods_full104_5001.json`` (usually under /mnt/oss, which
is a dead mount on some nodes):

* ``gt.npz``       <- ``<gt_root>/<cache_id>.npz``
* ``<method>.npz`` <- ``<method_roots[method]>/<cache_id>/predictions.npz``
* ``dyn_hamr.npz`` <- ``<dyn_hamr_formal_root>/<cache_id>/predictions.npz``

HaWoR is deliberately *not* relayed: the renderer reads it straight from
``--hawor-root`` + ``--hawor-index``.  Archive members are copied at the zip
level, so unused keys (depth maps, world-space mirrors) are not recompressed.

Example
-------
``python3 tools/relay_sources.py --dataset arctic \
    --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
    --out /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917``
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

METHOD_KEYS = ("wilor", "pad_hand", "egoforce", "reviv4d", "dyn_hamr")
DEFAULT_ALIGNMENT = ("visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/"
                     "source_alignment_auxmethods_full104_5001.json")


def cache_ids_of(entry: dict) -> list[str]:
    """Chronological, de-duplicated window cache ids of a 300-frame segment."""
    ids: list[str] = []
    for ref in entry.get("frame_refs") or []:
        cache = ref.get("cache_id")
        if cache and cache not in ids:
            ids.append(cache)
    return ids


def segment_id_of(entry: dict) -> str:
    frames = entry.get("frame_ids") or []
    return f"{entry['dataset']}__{frames[0]}-{frames[-1]}" if frames else entry["window_id"]


def method_source(spec: dict, method: str, cache: str) -> Path | None:
    if method == "gt":
        return Path(spec["gt_root"]) / f"{cache}.npz"
    if method == "dyn_hamr":
        root = spec.get("dyn_hamr_formal_root")
        if not root:
            return None
        return Path(root) / cache / "predictions.npz"
    roots = spec.get("method_roots", {}).get(method) or []
    for root in roots:
        candidate = Path(root) / cache / "predictions.npz"
        if candidate.is_file():
            return candidate
    return Path(roots[0]) / cache / "predictions.npz" if roots else None


def copy_archive(source: Path, target: Path) -> int:
    """Copy every member of ``source`` into ``target`` without recompressing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.incoming")
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            dst.writestr(item.filename, data, compress_type=zipfile.ZIP_DEFLATED)
        count = len(src.infolist())
    temporary.replace(target)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--alignment", type=Path, default=Path(DEFAULT_ALIGNMENT))
    parser.add_argument("--manifest", type=Path,
                        help="2D selection manifest; cache ids are taken per segment")
    parser.add_argument("--caches", nargs="*", default=[],
                        help="explicit window cache ids (instead of --manifest)")
    parser.add_argument("--methods", nargs="*", default=["gt", *METHOD_KEYS])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite existing relays")
    parser.add_argument("--check", action="store_true",
                        help="only report which sources are present (no writing)")
    args = parser.parse_args()

    alignment = json.loads(args.alignment.read_text())
    spec = alignment[args.dataset]
    if args.manifest:
        entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        caches = []
        for entry in entries:
            if entry.get("dataset") != args.dataset:
                continue
            caches.extend(cache_ids_of(entry))
    else:
        caches = list(args.caches)
    caches = list(dict.fromkeys(caches))
    if not caches:
        raise SystemExit("no cache ids selected (pass --manifest or --caches)")

    missing, copied = [], 0
    for cache in caches:
        for method in args.methods:
            source = method_source(spec, method, cache)
            target = args.out / cache / f"{method}.npz"
            if source is None or not source.is_file():
                missing.append({"cache": cache, "method": method, "source": str(source)})
                continue
            if args.check:
                continue
            if target.is_file() and not args.force:
                continue
            copied += 1
            members = copy_archive(source, target)
            print(f"relay {cache} {method}: {members} members -> {target}")
    print(json.dumps({"dataset": args.dataset, "caches": len(caches), "copied": copied,
                      "missing": len(missing), "missing_detail": missing[:20]}, indent=2))
    if missing:
        raise SystemExit("some sources are unavailable (dead /mnt/oss mount?); "
                         "pull them with ~/bin/ossutil2 and re-run")


if __name__ == "__main__":
    main()
