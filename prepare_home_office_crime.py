"""Map Home Office police recorded crime (CSP open data) to rent local authorities.

The police.uk street-level files have gaps, most notably Greater Manchester
from July 2019. The Home Office Community Safety Partnership (CSP) tables are
quarterly but complete, so they provide a second, gap-free crime measure.

* CSPs that equal a local authority map by name; predecessor CSPs merge into
  their successor authority (e.g. Allerdale -> Cumberland).
* CSPs covering several authorities are split by each authority's share of the
  combined population in that year; such rows are flagged ``allocated``.
* "Unassigned" offences (no CSP) are redistributed within the same force,
  quarter, and offence group by each CSP's share of the force's assigned
  offences over the previous four quarters (the current quarter only when no
  history exists). Assignment can fail for a whole CSP in a quarter (Western
  Suffolk 2015-Q2), so current-quarter shares are not reliable.
* Fraud is excluded (recorded by Action Fraud, not by CSP).
Writes only to processed/, failed/, and metadata/; raw/ is read-only.
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from prepare_reference import write_csv


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/Home Office Crime"
T = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
FIRST_QUARTER, LAST_QUARTER = (2014, 1), (2024, 4)
# One extra year is read so that redistribution weights exist for 2014.
READ_FROM = (2013, 1)
HEADER_ALIASES = {"Force Name": "Police Force", "Number of Offences": "Offence Count"}
GROUPS = {"violence against the person": "violence", "sexual offences": "sexual",
          "robbery": "robbery", "theft offences": "theft",
          "criminal damage and arson": "damage", "drug offences": "drugs",
          "public order offences": "public_order", "possession of weapons offences": "public_order",
          "miscellaneous crimes against society": "misc", "miscellaneous crimes": "misc"}
GROUP_KEYS = ["violence", "sexual", "robbery", "burglary", "theft", "damage", "drugs",
              "public_order", "misc"]
SUCCESSORS = {
    "Allerdale": ["Cumberland"], "Carlisle": ["Cumberland"], "Copeland": ["Cumberland"],
    "Barrow-in-Furness": ["Westmorland and Furness"], "Eden": ["Westmorland and Furness"],
    "South Lakeland": ["Westmorland and Furness"],
    "Bournemouth": ["Bournemouth, Christchurch and Poole"],
    "Poole": ["Bournemouth, Christchurch and Poole"],
    "Corby": ["North Northamptonshire"], "East Northamptonshire": ["North Northamptonshire"],
    "Kettering": ["North Northamptonshire"], "Wellingborough": ["North Northamptonshire"],
    "Northampton": ["West Northamptonshire"],
    "Daventry and South Northamptonshire": ["West Northamptonshire"],
    "Aylesbury Vale": ["Buckinghamshire"], "Chiltern": ["Buckinghamshire"],
    "South Bucks": ["Buckinghamshire"], "Wycombe": ["Buckinghamshire"],
    "Suffolk Coastal": ["East Suffolk"], "Waveney": ["East Suffolk"],
    "Shepway": ["Folkestone and Hythe"], "Manchester Airport": ["Manchester"],
    "Western Suffolk": ["Babergh", "Mid Suffolk", "West Suffolk"],
    "East and Mid Devon": ["East Devon", "Mid Devon"],
    "Northern Devon": ["North Devon", "Torridge"],
    "South Devon and Dartmoor": ["South Hams", "Teignbridge", "West Devon"],
    "Dartford and Gravesham": ["Dartford", "Gravesham"],
    "South Nottinghamshire": ["Broxtowe", "Gedling", "Rushcliffe"],
    "South Warwickshire": ["Stratford-on-Avon", "Warwick"],
    "North Worcestershire": ["Bromsgrove", "Redditch", "Wyre Forest"],
    "South Worcester": ["Malvern Hills", "Worcester", "Wychavon"],
    "Cwm Taf": ["Merthyr Tydfil", "Rhondda Cynon Taf"],
    "North Hampshire": ["Basingstoke and Deane", "Hart", "Rushmoor"],
}
# Not in the 316-authority rent sample, consistent with the police.uk mapping.
OUTSIDE_SAMPLE = {"City of London", "London, City of", "Isles of Scilly"}


def norm(name):
    name = name.lower().replace("&", "and").replace("'", "")
    name = re.sub(r"[,.\-]", " ", name)
    name = re.sub(r"\b(city of|county of|borough of|the)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def ods_rows(path: Path):
    with zipfile.ZipFile(path) as archive, archive.open("content.xml") as stream:
        for event, element in ET.iterparse(stream, events=("end",)):
            if element.tag != T + "table-row":
                continue
            values = []
            for cell in element:
                repeat = int(cell.get(T + "number-columns-repeated", "1"))
                value = cell.get(OFFICE + "value") or "".join(cell.itertext())
                values.extend([value] * (repeat if value else 1))
            yield values
            element.clear()


def calendar_quarter(financial_year: str, quarter: str):
    start, number = int(financial_year[:4]), int(quarter)
    return (start + 1, 1) if number == 4 else (start, number + 1)


def previous_quarters(period, count=4):
    index = period[0] * 4 + period[1] - 1
    return [((i // 4), i % 4 + 1) for i in range(index - count, index)]


def redistribute(counts):
    """Add unassigned offences to each CSP of the same force, quarter, and group."""
    assigned, unassigned = defaultdict(int), defaultdict(int)
    current = defaultdict(set)
    for (force, csp, period, group), number in counts.items():
        if csp.lower().startswith("unassigned"):
            unassigned[(force, period, group)] += number
        else:
            assigned[(force, csp, period, group)] += number
            current[(force, period)].add(csp)
    adjusted = defaultdict(float, assigned)
    for (force, period, group), number in unassigned.items():
        if not number:
            continue
        csps = current[(force, period)]
        weights = {csp: sum(assigned.get((force, csp, p, group), 0) for p in previous_quarters(period))
                   for csp in csps}
        if not sum(weights.values()):
            weights = {csp: assigned.get((force, csp, period, group), 0) for csp in csps}
        total = sum(weights.values())
        for csp, weight in weights.items():
            if weight:
                adjusted[(force, csp, period, group)] += number * weight / total
    return adjusted, assigned, unassigned


def read_counts():
    counts = defaultdict(int)
    stats = Counter()
    for path in sorted(SOURCE.glob("prc-csp-*.ods")):
        header = None
        for values in ods_rows(path):
            if values and values[0] == "Financial Year":
                header = {HEADER_ALIASES.get(v, v): i for i, v in enumerate(values) if v}
                continue
            if not header or len(values) <= header["Offence Count"] or not values[0][:2] == "20":
                continue
            field = lambda name: values[header[name]].strip()
            period = calendar_quarter(field("Financial Year"), field("Financial Quarter"))
            if not READ_FROM <= period <= LAST_QUARTER:
                continue
            group = field("Offence Group").lower()
            number = int(float(field("Offence Count") or 0))
            if group == "fraud offences":
                stats["fraud_excluded"] += number
                continue
            key = GROUPS[group]
            if key == "theft" and "burglary" in field("Offence Subgroup").lower():
                key = "burglary"
            counts[(norm(field("Police Force")), field("CSP Name"), period, key)] += number
            stats["offences"] += number
    return counts, stats


def populations():
    shares = {}
    with (ROOT / "data/processed/reference/gdhi_lad_annual.csv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            shares[(row["lad_code"], int(row["year"]))] = float(row["population"])
    return shares


def main():
    with (ROOT / "data/processed/reference/rent_lad_monthly.csv").open(encoding="utf-8") as file:
        names = {row["lad_code"]: row["lad_name"] for row in csv.DictReader(file)}
    by_name = {norm(name): code for code, name in names.items()}
    for targets in SUCCESSORS.values():
        missing = [t for t in targets if norm(t) not in by_name]
        if missing:
            raise RuntimeError(f"Crosswalk targets not in rent sample: {missing}")
    population = populations()
    counts, stats = read_counts()
    adjusted, assigned, unassigned = redistribute(counts)
    force_assigned = defaultdict(int)
    for (force, _, period, group), number in assigned.items():
        force_assigned[(force, period, group)] += number

    totals = defaultdict(lambda: defaultdict(float))
    sources = defaultdict(set)
    failed = []
    for (force, csp, period, group), number in adjusted.items():
        if period < FIRST_QUARTER:
            continue
        if csp in OUTSIDE_SAMPLE:
            failed.append({"force": force, "csp": csp, "quarter": "%d-Q%d" % period,
                           "offence_group": group, "count": number,
                           "failure_reason": "csp_outside_rent_sample"})
            continue
        targets = SUCCESSORS.get(csp, [csp])
        codes = [by_name.get(norm(t)) for t in targets]
        if None in codes:
            failed.append({"force": force, "csp": csp, "quarter": "%d-Q%d" % period,
                           "offence_group": group, "count": number,
                           "failure_reason": "csp_not_mapped_to_lad"})
            continue
        year = min(max(period[0], 2014), 2024)
        weights = [population[(code, year)] for code in codes]
        for code, weight in zip(codes, weights):
            totals[(code, period)][group] += number * weight / sum(weights)
            sources[(code, period)].add((csp, len(codes) > 1, force))

    periods = [(y, q) for y in range(FIRST_QUARTER[0], LAST_QUARTER[0] + 1) for q in range(1, 5)]
    rows, problems = [], []
    for code in sorted(names):
        for period in periods:
            source = sources.get((code, period), set())
            split = {s for s in source if s[1]}
            if not source:
                problems.append(f"{code} {period}: no CSP")
            elif split and len(source) > len(split):
                problems.append(f"{code} {period}: direct and split CSPs {sorted(source)}")
            forces = {s[2] for s in source}
            denominator = sum(force_assigned[(f, period, g)] + unassigned[(f, period, g)]
                              for f in forces for g in GROUP_KEYS)
            share = (sum(unassigned[(f, period, g)] for f in forces for g in GROUP_KEYS) /
                     denominator if denominator else None)
            values = totals.get((code, period), {})
            rows.append({"lad_code": code, "lad_name": names[code],
                         "quarter": "%d-Q%d" % period, "year": period[0], "quarter_number": period[1],
                         "recorded_crime_total": round(sum(values.values()), 3),
                         **{f"recorded_{g}": round(values.get(g, 0.0), 3) for g in GROUP_KEYS},
                         "allocated": bool(split),
                         "force_unassigned_share": None if share is None else round(share, 6),
                         "source_csps": ";".join(sorted({s[0] for s in source}))})
    if problems:
        raise RuntimeError("CSP mapping problems:\n" + "\n".join(problems[:50]))
    write_csv(ROOT / "data/processed/reference/recorded_crime_lad_quarterly.csv", list(rows[0]), rows)
    write_csv(ROOT / "data/failed/recorded_crime_unmapped.csv",
              ["force", "csp", "quarter", "offence_group", "count", "failure_reason"], failed)
    shares = [r["force_unassigned_share"] for r in rows if r["force_unassigned_share"] is not None]
    result = {"quarters": f"{periods[0]} to {periods[-1]}", "rows": len(rows),
              "expected_rows": len(names) * len(periods),
              "offences_read": stats["offences"], "fraud_excluded": stats["fraud_excluded"],
              "unassigned_redistributed": sum(v for (_, period, _), v in unassigned.items()
                                              if period >= FIRST_QUARTER),
              "failed_by_reason": dict(Counter(f["failure_reason"] for f in failed)),
              "allocated_lads": sorted({r["lad_name"] for r in rows if r["allocated"]}),
              "lad_quarters_unassigned_over_10pct": sum(s > 0.10 for s in shares),
              "zero_total_rows": sum(r["recorded_crime_total"] <= 0 for r in rows)}
    path = ROOT / "data/metadata/recorded_crime_reference.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if len(rows) != result["expected_rows"] or result["zero_total_rows"] or \
            result["failed_by_reason"].get("csp_not_mapped_to_lad"):
        raise RuntimeError("Recorded crime reference did not cover every local authority quarter")


if __name__ == "__main__":
    main()
