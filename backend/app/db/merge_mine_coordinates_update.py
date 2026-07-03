"""
One-off data merge: incorporate an updated coordinate file received from
Coal India into mine_master and mine_coordinates, matching on mine_code.

Scope and constraints:
  - mine_master and mine_coordinates SCHEMA is untouched (no new columns,
    no new tables) -- this is a DATA update only, explicitly requested by
    the project owner.
  - Only latitude, longitude, and is_mapped are written to mine_master for
    matched mine_codes. mine_name, subsidiary, area_code, area_description
    are never touched -- "preserve existing data where appropriate" means
    every non-coordinate field is left exactly as it was.
  - mine_master rows whose mine_code does NOT appear in the new file are
    left completely untouched (including their is_mapped flag) -- absence
    from this file is not evidence the mine is unmapped, just that this
    particular update doesn't cover it.
  - For mine_codes already mapped (is_mapped=1) that also appear in the
    new file, the new file's coordinates are treated as authoritative
    (it is described as an "updated" file from Coal India) and overwrite
    the old values -- but every such overwrite is recorded in the report
    so nothing changes silently.
  - mine_coordinates is upserted (insert new mine_codes, update existing)
    to stay in sync with mine_master for every newly-or-already mapped
    mine_code in this file.

DMS coordinate parsing: the source file stores lat/lon as degree-minute-
second strings (e.g. 23°47'29.39"N). Parsed to decimal degrees via the
standard formula deg + min/60 + sec/3600, negated for S/W hemispheres.
"""

import re
import sqlite3
from datetime import datetime

import pandas as pd

DMS_PATTERN = re.compile(r"(\d+)[\u00b0\s]+(\d+)['\u2032\s]+([\d.]+)[\"\u2033]?\s*([NSEW])")


def dms_to_decimal(s: str) -> float:
    s = s.strip()
    m = DMS_PATTERN.match(s)
    if not m:
        raise ValueError(f"Could not parse DMS coordinate: {s!r}")
    deg, mins, secs, hemi = m.groups()
    val = float(deg) + float(mins) / 60 + float(secs) / 3600
    if hemi in ("S", "W"):
        val = -val
    return round(val, 6)


def load_new_coordinates(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    df["latitude"] = df["Latitude"].apply(dms_to_decimal)
    df["longitude"] = df["Longitude"].apply(dms_to_decimal)
    return df[["Mine_code", "Subsidiary", "Area", "latitude", "longitude"]].rename(
        columns={"Mine_code": "mine_code", "Subsidiary": "subsidiary_in_file", "Area": "area_in_file"}
    )


def merge(db_path: str, xlsx_path: str) -> dict:
    new_coords = load_new_coordinates(xlsx_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()

        mm = pd.read_sql(
            "SELECT mine_code, mine_name, subsidiary, latitude, longitude, is_mapped FROM mine_master", con
        )

        new_codes = set(new_coords["mine_code"])
        existing_codes = set(mm["mine_code"])

        unmatched_in_file = sorted(new_codes - existing_codes)
        matched_codes = new_codes & existing_codes

        merged = mm.merge(new_coords, on="mine_code", how="inner")

        newly_mapped_mask = merged["is_mapped"] == 0
        already_mapped_mask = merged["is_mapped"] == 1

        newly_mapped_rows = merged[newly_mapped_mask].copy()
        already_mapped_rows = merged[already_mapped_mask].copy()

        # detect actual coordinate changes among already-mapped mines (rounding-safe)
        already_mapped_rows["lat_diff"] = (already_mapped_rows["latitude_x"] - already_mapped_rows["latitude_y"]).abs()
        already_mapped_rows["lon_diff"] = (already_mapped_rows["longitude_x"] - already_mapped_rows["longitude_y"]).abs()
        refreshed_rows = already_mapped_rows[
            (already_mapped_rows["lat_diff"] > 0.0001) | (already_mapped_rows["lon_diff"] > 0.0001)
        ].copy()

        # --- apply updates to mine_master ---
        update_payload = [
            (float(row["latitude_y"]), float(row["longitude_y"]), int(row["mine_code"]))
            for _, row in merged.iterrows()
        ]
        cur.executemany(
            "UPDATE mine_master SET latitude = ?, longitude = ?, is_mapped = 1 WHERE mine_code = ?",
            update_payload,
        )

        # --- upsert mine_coordinates (manual, since mine_code has no UNIQUE
        #     constraint on this pre-existing table -- ON CONFLICT is not usable) ---
        cur.execute("SELECT mine_code FROM mine_coordinates")
        existing_coord_codes = {r[0] for r in cur.fetchall()}

        coord_payload = [
            (int(row["mine_code"]), float(row["latitude_y"]), float(row["longitude_y"]))
            for _, row in merged.iterrows()
        ]
        to_update = [c for c in coord_payload if c[0] in existing_coord_codes]
        to_insert = [c for c in coord_payload if c[0] not in existing_coord_codes]

        cur.executemany(
            "UPDATE mine_coordinates SET latitude = ?, longitude = ? WHERE mine_code = ?",
            [(lat, lon, code) for code, lat, lon in to_update],
        )
        cur.executemany(
            "INSERT INTO mine_coordinates (mine_code, latitude, longitude) VALUES (?, ?, ?)",
            to_insert,
        )

        # audit log entry (additive only, consistent with project convention)
        cur.execute(
            """
            INSERT INTO data_quality_log (issue_type, severity, affected_table, description, row_count, resolution)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "mine_coordinates_update",
                "LOW",
                "mine_master,mine_coordinates",
                f"Merged updated coordinate file from Coal India (Latitude_-Longitude.xlsx, {len(new_coords)} rows) "
                f"into mine_master/mine_coordinates by matching mine_code. "
                f"{len(matched_codes)} matched, {len(newly_mapped_rows)} newly mapped (is_mapped 0->1), "
                f"{len(already_mapped_rows)} previously-mapped mines retained "
                f"({len(refreshed_rows)} had coordinates refreshed by >0.0001 deg), "
                f"{len(unmatched_in_file)} mine_codes in file could not be matched to mine_master.",
                len(matched_codes),
                "No further action needed; this is a completed one-off data merge.",
            ),
        )

        con.commit()

        cur.execute("SELECT COUNT(*) FROM mine_master WHERE is_mapped = 1")
        final_mapped = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM mine_master")
        total_mines = cur.fetchone()[0]

        return {
            "total_mines_in_master": total_mines,
            "total_in_new_file": len(new_coords),
            "matched": len(matched_codes),
            "newly_mapped": len(newly_mapped_rows),
            "previously_mapped_retained": len(already_mapped_rows),
            "previously_mapped_coordinates_refreshed": len(refreshed_rows),
            "refreshed_mine_codes": refreshed_rows["mine_code"].tolist() if len(refreshed_rows) else [],
            "unmatched_in_file": unmatched_in_file,
            "final_mapped_count": final_mapped,
            "final_coverage_pct": round(100 * final_mapped / total_mines, 2),
            "newly_mapped_mine_codes": newly_mapped_rows["mine_code"].tolist(),
            "merged_at": datetime.now().isoformat(),
        }
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    import json
    db_file = sys.argv[1] if len(sys.argv) > 1 else "coal_eie.db"
    xlsx_file = sys.argv[2] if len(sys.argv) > 2 else "Latitude_-Longitude.xlsx"
    result = merge(db_file, xlsx_file)
    print(json.dumps(result, indent=2, default=str))
