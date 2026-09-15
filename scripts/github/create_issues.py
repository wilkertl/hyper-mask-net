#!/usr/bin/env python3
"""Create labels, milestones, and issues from docs/backlog/issues.md with the GitHub CLI.

Safe to re-run: labels are updated, and milestones and issues whose titles already exist are
skipped. Run from inside the repository so that `gh` resolves the target repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

BACKLOG = Path(__file__).resolve().parents[2] / "docs" / "backlog" / "issues.md"
LABEL_COLORS = {
    "type": "1d76db",
    "module": "5319e7",
    "phase": "0e8a16",
    "gate": "d93f0b",
    "status": "fbca04",
}
EXTRA_LABELS = {
    "protocol-change": ("b60205", "Changes the scientific protocol"),
    "invalidates-results": ("b60205", "Invalidates previously reported results"),
}
MILESTONE_HEADER = re.compile(r"^# (Milestone .+)$")
ISSUE_HEADER = re.compile(r"^## \[(?P<ticket>[^\]]+)\] (?P<title>.+)$")
META = re.compile(r"^<!--\s*(?P<body>.*?)\s*-->$")


@dataclass
class Issue:
    ticket: str
    title: str
    milestone: str
    labels: list[str]
    depends: list[str]
    body_lines: list[str] = field(default_factory=list)

    @property
    def full_title(self) -> str:
        return f"[{self.ticket}] {self.title}"


def parse_meta(line: str) -> dict[str, str]:
    match = META.match(line.strip())
    if not match:
        raise ValueError(f"expected a metadata comment, got {line!r}")
    pairs = (item.split(":", 1) for item in match["body"].split(";") if item.strip())
    return {key.strip(): value.strip() for key, value in pairs}


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_backlog(text: str) -> tuple[dict[str, str], list[Issue]]:
    milestones: dict[str, str] = {}
    issues: list[Issue] = []
    milestone: str | None = None
    current: Issue | None = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if header := MILESTONE_HEADER.match(line):
            milestone = header[1]
            milestones[milestone] = parse_meta(lines[index + 1]).get("description", "")
            current = None
            index += 2
        elif header := ISSUE_HEADER.match(line):
            if milestone is None:
                raise ValueError(f"issue {header['ticket']} appears before any milestone")
            meta = parse_meta(lines[index + 1])
            current = Issue(
                header["ticket"],
                header["title"],
                milestone,
                split_list(meta.get("labels", "")),
                split_list(meta.get("depends", "")),
            )
            issues.append(current)
            index += 2
        else:
            if current is not None:
                current.body_lines.append(line)
            index += 1

    defined: set[str] = set()
    for issue in issues:
        if issue.ticket in defined:
            raise ValueError(f"duplicate ticket {issue.ticket}")
        missing = [ticket for ticket in issue.depends if ticket not in defined]
        if missing:
            raise ValueError(f"{issue.ticket} depends on {missing}, which must be defined earlier")
        defined.add(issue.ticket)
    return milestones, issues


def gh(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def label_spec(name: str) -> tuple[str, str]:
    if name in EXTRA_LABELS:
        return EXTRA_LABELS[name]
    prefix, _, value = name.partition(":")
    if prefix not in LABEL_COLORS or not value:
        raise ValueError(f"label {name!r} has no known prefix")
    return LABEL_COLORS[prefix], f"{prefix.capitalize()}: {value}"


def ensure_labels(issues: list[Issue], dry_run: bool) -> None:
    names = sorted({label for issue in issues for label in issue.labels} | set(EXTRA_LABELS))
    for name in names:
        color, description = label_spec(name)
        if dry_run:
            print(f"label      {name}")
            continue
        gh("label", "create", name, "--color", color, "--description", description, "--force")


def ensure_milestones(milestones: dict[str, str], dry_run: bool) -> None:
    existing: set[str] = set()
    if not dry_run:
        response = gh("api", "repos/{owner}/{repo}/milestones?state=all&per_page=100")
        existing = {milestone["title"] for milestone in json.loads(response)}
    for title, description in milestones.items():
        if title in existing:
            print(f"milestone  {title} (exists)")
        elif dry_run:
            print(f"milestone  {title}")
        else:
            gh(
                "api",
                "repos/{owner}/{repo}/milestones",
                "-f",
                f"title={title}",
                "-f",
                f"description={description}",
            )
            print(f"milestone  {title} (created)")


def create_issues(issues: list[Issue], dry_run: bool) -> None:
    existing: dict[str, int] = {}
    if not dry_run:
        response = gh(
            "issue", "list", "--state", "all", "--limit", "1000", "--json", "number,title"
        )
        existing = {issue["title"]: issue["number"] for issue in json.loads(response)}
    numbers: dict[str, int] = {}
    for issue in issues:
        body = "\n".join(issue.body_lines).strip()
        if issue.depends:
            refs = ", ".join(
                f"#{numbers[ticket]}" if ticket in numbers else ticket for ticket in issue.depends
            )
            body += f"\n\n**Blocked by:** {refs}"
        if issue.full_title in existing:
            numbers[issue.ticket] = existing[issue.full_title]
            print(f"issue      #{numbers[issue.ticket]} {issue.full_title} (exists)")
            continue
        if dry_run:
            print(f"issue      {issue.full_title} [{', '.join(issue.labels)}] -> {issue.milestone}")
            continue
        args = ["issue", "create", "--title", issue.full_title, "--body-file", "-"]
        args += ["--milestone", issue.milestone]
        for label in issue.labels:
            args += ["--label", label]
        url = gh(*args, stdin=body).strip().splitlines()[-1]
        numbers[issue.ticket] = int(url.rstrip("/").rsplit("/", 1)[-1])
        print(f"issue      #{numbers[issue.ticket]} {issue.full_title} (created)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, default=BACKLOG)
    parser.add_argument("--dry-run", action="store_true", help="print actions without calling gh")
    args = parser.parse_args()
    milestones, issues = parse_backlog(args.backlog.read_text())
    print(f"{len(milestones)} milestones, {len(issues)} issues parsed from {args.backlog}")
    ensure_labels(issues, args.dry_run)
    ensure_milestones(milestones, args.dry_run)
    create_issues(issues, args.dry_run)


if __name__ == "__main__":
    main()
