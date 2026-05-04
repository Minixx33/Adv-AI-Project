"""
fix_susp_won.py

Adds a corrected `n_village_sus_won` column to every summary.csv under
the results directory.

The problem with the original `n_susp_agents_won`:
  It counts every sus agent whose team won — including POSSESSED agents,
  who are wolf-aligned and win when wolves win. This inflates the metric
  in wolf-win games and can make n_susp_agents_won > n_susp_agents_survived.

The fix:
  n_village_sus_won = n_suspicion_agents   if village won
                    = 0                    if wolves won

  This correctly counts how many sus agents were on the winning village
  side, regardless of survival. POSSESSED sus agents winning with wolves
  are no longer counted.

Usage:
  python fix_susp_won.py                        # patches all runs under ./results
  python fix_susp_won.py --results path/to/dir  # custom results directory
  python fix_susp_won.py --dry-run              # preview only, no files written
"""

import argparse
import csv
import os
import shutil


def fix_summary(path: str, dry_run: bool) -> dict:
    """
    Reads a summary.csv, adds n_village_sus_won, writes back in-place.
    Returns a small stats dict for reporting.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"skipped": True, "reason": "empty file"}

    if "village_win" not in rows[0] or "n_suspicion_agents" not in rows[0]:
        return {"skipped": True, "reason": "missing required columns"}

    if "n_village_sus_won" in rows[0]:
        return {"skipped": True, "reason": "already patched"}

    fixed = 0
    for row in rows:
        try:
            village_win = row["village_win"].strip().lower() in ("true", "1")
            n_sus = int(row["n_suspicion_agents"] or 0)
        except (ValueError, KeyError):
            row["n_village_sus_won"] = ""
            continue

        row["n_village_sus_won"] = n_sus if village_win else 0
        fixed += 1

    if dry_run:
        return {"patched_rows": fixed, "dry_run": True}

    # Build new fieldnames — insert n_village_sus_won right after n_susp_agents_won
    original_fields = list(rows[0].keys())
    original_fields.remove("n_village_sus_won")  # DictReader added it at end
    try:
        insert_at = original_fields.index("n_susp_agents_won") + 1
    except ValueError:
        insert_at = original_fields.index("n_susp_agents_survived") + 1
    new_fields = (
        original_fields[:insert_at]
        + ["n_village_sus_won"]
        + original_fields[insert_at:]
    )

    # Write back (via temp file for safety)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    shutil.move(tmp_path, path)
    return {"patched_rows": fixed}


def main():
    parser = argparse.ArgumentParser(description="Fix n_susp_agents_won in summary CSVs.")
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Path to the results directory (default: ./results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing any files.",
    )
    args = parser.parse_args()

    results_dir = args.results
    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        return

    summary_files = []
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname == "summary.csv":
                summary_files.append(os.path.join(root, fname))

    if not summary_files:
        print("No summary.csv files found.")
        return

    print(f"Found {len(summary_files)} summary.csv file(s).")
    if args.dry_run:
        print("DRY RUN — no files will be written.\n")

    patched = 0
    skipped = 0
    for path in sorted(summary_files):
        run_name = os.path.basename(os.path.dirname(path))
        result = fix_summary(path, dry_run=args.dry_run)

        if result.get("skipped"):
            print(f"  SKIP  {run_name}  ({result['reason']})")
            skipped += 1
        else:
            tag = "[DRY]" if result.get("dry_run") else "  OK  "
            print(f"  {tag} {run_name}  ({result['patched_rows']} rows patched)")
            patched += 1

    print(f"\nDone. {patched} file(s) patched, {skipped} skipped.")
    if not args.dry_run:
        print("Column 'n_village_sus_won' added after 'n_susp_agents_won' in each file.")


if __name__ == "__main__":
    main()
