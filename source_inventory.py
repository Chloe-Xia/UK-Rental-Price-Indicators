"""Record read-only raw source inventory for reproducibility."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    raw = root / "data/raw"
    groups = {}
    for path in sorted(raw.iterdir()):
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
        else:
            files = [path]
        groups[path.name] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    checksums = {}
    for relative in ["price_index_of_private_rents.xlsx",
                     "regional_gross_disposable_house_hold_income.xlsx",
                     "regional_gdhi_local_authorities_1997_2024.xlsx",
                     "Geography Lookup/ONSPD_AUG_2024.zip"]:
        path = raw / relative
        checksums[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                "raw_root": str(raw), "groups": groups, "reference_checksums": checksums}
    output = root / "data/metadata/source_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
