"""Regroup a tiled-mosaic dataset so splits divide on physical site, not on file.

Repeat surveys of one location are separate GeoTIFFs with unrelated names
(``Simmonds_kelp_U0171``, ``Simonds_kelp_U0902``, ``Simmonds_kelp_U0547``), so a
split assigned per file silently puts the same reef in train and val. The model
then memorises structure that does not change between flights -- substrate,
bathymetry, channel shape -- and the held-out score measures recall of a known
place rather than generalization to a new one.

Sites are recovered from geography rather than filenames: footprints are
reprojected to WGS84 and any two mosaics overlapping by more than
``--min-overlap`` of the smaller footprint are unioned into one site. Each site
is then assigned whole to a split, and the output is a tree of symlinks in the
layout ``make_chip_dataset`` expects, so the source archive is never touched.

Report the grouping and per-site kelp budget first::

    python -m hakai_ml_train.prepare.resplit_by_site <src> --report

then build the tree, naming sites by their canonical (alphabetically first) stem::

    python -m hakai_ml_train.prepare.resplit_by_site <src> <dst> \
        --val SpiderHIRMD_U0903,Stryker_kelp_U0548 --test Golden_kelp_U0546
"""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from tqdm import tqdm

SPLITS = ("train", "val", "test")


def load_footprints(src: Path) -> dict[str, dict]:
    """Map mosaic stem -> its split, WGS84 bounds and image/label paths."""
    out: dict[str, dict] = {}
    for split in SPLITS:
        for img in sorted((src / split / "images").glob("*.tif")):
            label = src / split / "labels" / img.name
            if not label.exists():
                raise SystemExit(f"Missing label for {img}")
            with rasterio.open(img) as ds:
                bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            out[img.stem] = {
                "split": split,
                "bounds": bounds,
                "image": img,
                "label": label,
            }
    if not out:
        raise SystemExit(f"No mosaics found under {src}/<split>/images")
    return out


def _overlap(a, b) -> float:
    """Intersection as a fraction of the smaller footprint."""
    dx = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if dx * dy <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return dx * dy / min(area_a, area_b)


def group_sites(fp: dict[str, dict], min_overlap: float) -> dict[str, list[str]]:
    """Union mosaics whose footprints overlap into sites, keyed by canonical stem."""
    parent = {name: name for name in fp}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(fp, 2):
        if _overlap(fp[a]["bounds"], fp[b]["bounds"]) > min_overlap:
            parent[find(a)] = find(b)

    groups = defaultdict(list)
    for name in fp:
        groups[find(name)].append(name)
    # Key each site by its alphabetically first member so the id is stable.
    return {min(members): sorted(members) for members in groups.values()}


def kelp_budget(label_path: Path, max_dim: int = 3000) -> tuple[float, float]:
    """Approximate (macro, nereo) pixel counts, read decimated for speed."""
    with rasterio.open(label_path) as ds:
        h, w = ds.shape
        step = max(1, int(max(h, w) / max_dim))
        arr = ds.read(
            1, out_shape=(h // step, w // step), resampling=Resampling.nearest
        )
    values, counts = np.unique(arr, return_counts=True)
    frac = dict(zip(values.tolist(), (counts / counts.sum()).tolist(), strict=True))
    area = h * w
    return frac.get(2, 0.0) * area, frac.get(3, 0.0) * area


def report(fp: dict[str, dict], sites: dict[str, list[str]]) -> None:
    rows = []
    for site, members in tqdm(sites.items(), desc="Reading labels"):
        macro = nereo = 0.0
        for m in members:
            a, b = kelp_budget(fp[m]["label"])
            macro += a
            nereo += b
        splits = sorted({fp[m]["split"] for m in members})
        rows.append((macro + nereo, site, members, macro, nereo, splits))
    rows.sort(reverse=True)
    total_m = sum(r[3] for r in rows) or 1.0
    total_n = sum(r[4] for r in rows) or 1.0

    print(f"\n{len(fp)} mosaics -> {len(sites)} sites\n")
    header = f"{'site (canonical stem)':30s} {'srv':>3s} {'macro':>9s} {'nereo':>9s}  current"
    print(header)
    print("-" * len(header))
    for _, site, members, macro, nereo, splits in rows:
        flag = " <- SPANS SPLITS" if len(splits) > 1 else ""
        print(
            f"{site:30s} {len(members):3d} {macro / total_m:8.1%} "
            f"{nereo / total_n:8.1%}  {','.join(splits)}{flag}"
        )
        if len(members) > 1:
            for m in members:
                print(f"{'':30s}     {fp[m]['split']:5s} {m}")


def build(
    fp: dict[str, dict], sites: dict[str, list[str]], assign: dict[str, str], dst: Path
) -> None:
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    for site, members in sites.items():
        split = assign[site]
        for m in members:
            for kind in ("image", "label"):
                src_path = fp[m][kind].resolve()
                link = dst / split / f"{kind}s" / src_path.name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(src_path)
            counts[split] += 1

    print(f"\nLinked into {dst}:")
    for split in SPLITS:
        n_sites = sum(1 for s in sites if assign[s] == split)
        print(f"  {split:5s} {counts[split]:3d} mosaics from {n_sites} sites")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Dataset root holding <split>/images.")
    parser.add_argument(
        "dst", type=Path, nargs="?", help="Where to write the symlink tree."
    )
    parser.add_argument(
        "--report", action="store_true", help="Print grouping and exit."
    )
    parser.add_argument("--val", default="", help="Comma-separated canonical stems.")
    parser.add_argument("--test", default="", help="Comma-separated canonical stems.")
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.01,
        help="Footprint overlap fraction above which two mosaics are one site.",
    )
    args = parser.parse_args()

    fp = load_footprints(args.src)
    sites = group_sites(fp, args.min_overlap)

    if args.report or args.dst is None:
        report(fp, sites)
        return

    val = [s for s in args.val.split(",") if s]
    test = [s for s in args.test.split(",") if s]
    unknown = [s for s in val + test if s not in sites]
    if unknown:
        raise SystemExit(
            f"Not canonical site stems: {unknown}\nKnown sites: {sorted(sites)}"
        )
    overlap = set(val) & set(test)
    if overlap:
        raise SystemExit(f"Sites assigned to both val and test: {sorted(overlap)}")

    assign = {s: "train" for s in sites}
    assign.update({s: "val" for s in val})
    assign.update({s: "test" for s in test})
    build(fp, sites, assign, args.dst)


if __name__ == "__main__":
    main()
