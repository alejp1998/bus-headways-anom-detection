#!/usr/bin/env python3
"""Rebuild London lines_dict.json from the live TfL Route/Sequence API.

Produces the {line: {direction: {name, stops, length}, destinations}} structure
the original algorithms expect, with CURRENT naptan stop IDs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "London" / "Data" / "Static" / "lines_dict.json"

LINES = ["18", "24", "25", "73"]


def fetch_line(line: str) -> dict:
    """Fetch route metadata + stop sequence for both directions."""
    meta_resp = requests.get(f"https://api.tfl.gov.uk/Line/{line}", timeout=10)
    meta = meta_resp.json()
    if not isinstance(meta, list) or not meta:
        raise RuntimeError(f"meta {meta_resp.status_code}: {str(meta)[:80]}")
    meta = meta[0]
    line_name = meta.get("name", line)

    out = {}
    destinations = []
    for direction_name, direction_id in (("outbound", "1"), ("inbound", "2")):
        r = requests.get(
            f"https://api.tfl.gov.uk/Line/{line}/Route/Sequence/{direction_name}",
            timeout=10,
        )
        time.sleep(1.0)  # be gentle with the public API
        if r.status_code != 200:
            continue
        data = r.json()
        stops = []
        for seq in data.get("stopPointSequences", []):
            for st in seq.get("stopPoint", []):
                if st.get("id") not in stops:
                    stops.append(st["id"])
        dest = data.get("destinationName", "") or data.get("lineName", "")
        # Prefer the route name ("Terminus A &harr; Terminus B") from orderedLineRoutes
        for olr in data.get("orderedLineRoutes", []):
            rname = olr.get("name", "")
            if "&harr;" in rname or " ↔ " in rname or " - " in rname:
                dest = rname
                break
        dest = dest.replace("&harr;", "↔")
        for sep in ("↔", " - "):
            if sep in dest:
                parts = [p.strip() for p in dest.split(sep) if p.strip()]
                dest = parts[0]  # primary terminus label
                for p in parts[1:]:
                    if p not in destinations:
                        destinations.append(p)
                break
        if dest and dest not in destinations and len(destinations) < 2:
            destinations.append(dest)
        out[direction_id] = {
            "name": line_name,
            "stops": stops,
            "length": len(stops),
        }

    return {
        "1": out.get("1", {"name": line_name, "stops": [], "length": 0}),
        "2": out.get("2", {"name": line_name, "stops": [], "length": 0}),
        "destinations": destinations,
    }


def main() -> None:
    result = {}
    for line in LINES:
        try:
            result[line] = fetch_line(line)
            print(
                f"  Line {line}: {len(result[line]['1']['stops'])} stops outbound, "
                f"{len(result[line]['2']['stops'])} inbound, destinations={result[line]['destinations']}"
            )
        except Exception as e:
            print(f"  Line {line}: FAILED {e}")
    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
