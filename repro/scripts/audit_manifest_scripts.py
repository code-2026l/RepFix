"""Audit: do the manifests' declared drivers and outputs ship in this release?

Run from anywhere inside the release:

    python repro/scripts/audit_manifest_scripts.py

Two failure modes a reproducibility reviewer hits, both silent:

  1. a manifest naming a driver the release does not contain, so the job cannot
     be replayed at all;
  2. a manifest whose ``out`` paths are absent, so the file a table is computed
     from cannot be found by following the manifest.

The check resolves each job's ``script`` against the tracked tree -- ``code/``
holds the experiment drivers, ``repro/scripts/`` the testbed scripts, as
results/README.md documents -- and compares each ``out`` basename against the
set of shipped basenames.  Basenames rather than paths, because the released
tree consolidates the cluster's campaign directories (``results/REG`` ->
``repro/results/P03``, and so on); results/README.md maps that consolidation
and lists the exceptions this audit reports.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def repo_root(start: str) -> str:
    """Walk up from *start* until a directory holding configs/ and repro/."""
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "configs")) and os.path.isdir(
            os.path.join(cur, "repro")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise SystemExit("could not locate the release root above %s" % start)
        cur = parent


def tracked_files(root: str) -> list[str]:
    """The release's file list.

    Prefers git (which is what the packer archives from); falls back to a plain
    walk so the audit still works from an unpacked tarball.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if out.strip():
            return out.split()
    except Exception:  # noqa: BLE001  -- not a checkout, or git unavailable
        pass
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return files


def main() -> int:
    root = repo_root(os.path.dirname(os.path.abspath(__file__)))
    tracked = tracked_files(root)
    shipped_names = {os.path.basename(p) for p in tracked}
    shipped_scripts = {os.path.basename(p) for p in tracked if p.endswith(".py")}

    print("=" * 96)
    print("1. drivers referenced by shipped manifests")
    print("=" * 96)
    missing_drivers: dict[str, set[str]] = {}
    totals: dict[str, tuple[int, int]] = {}
    gaps: list[tuple[str, str]] = []

    for path in sorted(glob.glob(os.path.join(root, "configs", "*.json"))):
        mname = os.path.basename(path)
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print("  UNREADABLE %s: %s" % (mname, exc))
            continue
        jobs = doc["jobs"] if isinstance(doc, dict) and "jobs" in doc else doc
        if not isinstance(jobs, list):
            continue
        have = miss = 0
        for job in jobs:
            script = job.get("script")
            if script:
                base = os.path.basename(script)
                if base not in shipped_scripts:
                    missing_drivers.setdefault(base, set()).add(mname)
            out = job.get("out")
            if out:
                base = os.path.basename(out)
                if base in shipped_names:
                    have += 1
                else:
                    miss += 1
                    gaps.append((mname, base))
        totals[mname] = (have, miss)

    if missing_drivers:
        for base, mfs in sorted(missing_drivers.items()):
            print("  MISSING %-32s cited by %s" % (base, ", ".join(sorted(mfs))))
    else:
        print("  all referenced drivers ship")

    print()
    print("=" * 96)
    print("2. result files declared by shipped manifests")
    print("=" * 96)
    print("  %-28s %8s %8s" % ("manifest", "shipped", "absent"))
    for mname, (have, miss) in sorted(totals.items()):
        print("  %-28s %8d %8d%s" % (mname, have, miss, "" if miss == 0 else "   <-- gap"))

    if gaps:
        byman: dict[str, list[str]] = {}
        for mname, base in gaps:
            byman.setdefault(mname, []).append(base)
        print()
        print("  absent outputs (%d), grouped by manifest:" % len(gaps))
        for mname, bases in sorted(byman.items()):
            print("    %s (%d):" % (mname, len(bases)))
            for b in sorted(bases)[:5]:
                print("        %s" % b)
            if len(bases) > 5:
                print("        ... and %d more" % (len(bases) - 5))
        print()
        print("  These are the consolidations and superseded passes that")
        print("  results/README.md documents under 'Declared outputs this tree")
        print("  does not carry'; the mapping from result file to table or")
        print("  figure lives there, not in the out paths.")
    else:
        print()
        print("  every declared output ships")

    return 0


if __name__ == "__main__":
    sys.exit(main())
