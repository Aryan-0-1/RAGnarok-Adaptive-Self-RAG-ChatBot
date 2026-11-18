#!/usr/bin/env python3
"""
backdate_commits.py

Run this INSIDE your project folder. It splits your files into several
batches and creates one git commit per batch, with commit dates spread
across a date range (default: the last 60 days).

Usage:
    python backdate_commits.py                      # 25 commits over last 60 days
    python backdate_commits.py --commits 30
    python backdate_commits.py --start 2026-09-18 --end 2026-11-17
    python backdate_commits.py --dry-run            # preview only, changes nothing

Then push:
    git remote add origin https://github.com/<you>/<repo>.git
    git push -u origin main

Notes:
  * Set your git identity first so commits are linked to your GitHub account:
        git config user.name  "Your Name"
        git config user.email "the-email-on-your-github-account@example.com"
  * Standard library only, no installs needed.
  * Run it once on a fresh project. If a .git folder already exists with
    commits, the script stops so it doesn't mix with existing history.
"""

import argparse
import datetime as dt
import os
import random
import subprocess
import sys

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
}
IGNORE_FILES = {".DS_Store", "Thumbs.db"}


def run(cmd, env=None, capture=False):
    return subprocess.run(
        cmd, check=True, env=env, text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
    )


def collect_files(root, self_name):
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f in IGNORE_FILES or f == self_name:
                continue
            rel = os.path.relpath(os.path.join(folder, f), root)
            found.append(rel.replace(os.sep, "/"))
    return found


def category(path):
    """Lower number = committed earlier, so history looks like a real build-up."""
    p = path.lower()
    name = os.path.basename(p)
    ext = os.path.splitext(name)[1]
    if name in {".gitignore", "requirements.txt", "pyproject.toml", "setup.py",
                "environment.yml", "package.json", "config.py", "config.yaml"}:
        return 0, "Add project setup"
    if ext in {".csv", ".json", ".xlsx", ".parquet", ".txt"} and (
            "data" in p or "dataset" in p):
        return 1, "Add data"
    if p.startswith("test") or "/test" in p or name.startswith("test_"):
        return 4, "Add tests"
    if ext == ".ipynb":
        return 3, "Add notebook"
    if ext in {".md", ".rst", ".pdf", ".png", ".jpg", ".jpeg", ".gif"} or name.startswith("readme"):
        return 5, "Add docs"
    return 2, "Add"


def build_batches(files, n):
    files = sorted(files, key=lambda f: (category(f)[0], f))
    n = min(n, len(files))
    size, extra = divmod(len(files), n)
    batches, i = [], 0
    for k in range(n):
        step = size + (1 if k < extra else 0)
        batches.append(files[i:i + step])
        i += step
    return batches


def message_for(batch):
    _, verb = category(batch[0])
    first = os.path.basename(batch[0])
    if len(batch) == 1:
        return f"{verb}: {first}"
    return f"{verb}: {first} and {len(batch) - 1} more"


def pick_datetimes(start, end, n, tz):
    total_days = (end - start).days + 1
    if n > total_days:
        sys.exit(f"Need at least {n} days in range, got {total_days}. Lower --commits.")
    day_offsets = sorted(random.sample(range(total_days), n))
    # make sure the range is actually used from start to end
    day_offsets[0] = 0
    day_offsets[-1] = total_days - 1
    day_offsets = sorted(set(day_offsets))
    while len(day_offsets) < n:
        cand = random.randrange(total_days)
        if cand not in day_offsets:
            day_offsets.append(cand)
            day_offsets.sort()
    stamps = []
    for off in day_offsets:
        d = start + dt.timedelta(days=off)
        t = dt.time(random.randint(10, 22), random.randint(0, 59), random.randint(0, 59))
        stamps.append(f"{d.isoformat()} {t.strftime('%H:%M:%S')} {tz}")
    return stamps


def main():
    today = dt.date.today()
    ap = argparse.ArgumentParser(description="Spread commits across a date range.")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=173))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=today - dt.timedelta(days=113))
    ap.add_argument("--commits", type=int, default=40, help="number of commits (default 40)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--tz", default="+0530", help="timezone offset, default +0530 (IST)")
    ap.add_argument("--seed", type=int, default=None, help="fix randomness for repeatable output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    if args.end < args.start:
        sys.exit("--end must be after --start")

    root = os.getcwd()
    self_name = os.path.basename(__file__)
    files = collect_files(root, self_name)
    if not files:
        sys.exit("No files found. Run this from inside your project folder.")

    if os.path.isdir(os.path.join(root, ".git")):
        has_commits = subprocess.run(
            ["git", "rev-parse", "HEAD"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode == 0
        if has_commits:
            sys.exit("This folder already has commits. Use a fresh copy of the project.")

    batches = build_batches(files, args.commits)
    stamps = pick_datetimes(args.start, args.end, len(batches), args.tz)

    print(f"{len(files)} files -> {len(batches)} commits "
          f"between {args.start} and {args.end}\n")
    for stamp, batch in zip(stamps, batches):
        print(f"{stamp}  {message_for(batch)}  ({len(batch)} files)")

    if args.dry_run:
        print("\nDry run only. Nothing was changed.")
        return

    if not os.path.isdir(os.path.join(root, ".git")):
        run(["git", "init"])
    run(["git", "checkout", "-B", args.branch])

    for stamp, batch in zip(stamps, batches):
        run(["git", "add", "--"] + batch)
        env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
        run(["git", "commit", "-m", message_for(batch)], env=env)

    # pick up anything left over (e.g. files created while running)
    status = run(["git", "status", "--porcelain"], capture=True).stdout.strip()
    if status:
        run(["git", "add", "-A"])
        env = dict(os.environ, GIT_AUTHOR_DATE=stamps[-1], GIT_COMMITTER_DATE=stamps[-1])
        run(["git", "commit", "-m", "Final cleanup"], env=env)

    print("\nDone. Check with:  git log --format='%ad  %s' --date=short")
    print("Then push:         git remote add origin <url> && git push -u origin "
          + args.branch)


if __name__ == "__main__":
    main()
