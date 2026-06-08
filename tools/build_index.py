#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "papers"
INDEX_DIR = ROOT / "index"
REQUIRED_FIELDS = {
    "id",
    "title",
    "authors",
    "year",
    "read_date",
    "status",
    "topics",
    "source_url",
    "local_pdf",
    "notes",
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.isdigit():
        return int(value)
    return value


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith(" ") or ":" not in raw:
            raise ValueError(f"{path}: unsupported YAML line: {raw}")
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = parse_scalar(value)
            continue

        nested: list[str] | dict[str, Any] | None = None
        while index < len(lines):
            child = lines[index]
            if not child.strip():
                index += 1
                continue
            if not child.startswith(" "):
                break
            stripped = child.strip()
            if stripped.startswith("- "):
                if nested is None:
                    nested = []
                if not isinstance(nested, list):
                    raise ValueError(f"{path}: mixed list/map for {key}")
                nested.append(parse_scalar(stripped[2:]))
            elif ":" in stripped:
                if nested is None:
                    nested = {}
                if not isinstance(nested, dict):
                    raise ValueError(f"{path}: mixed list/map for {key}")
                child_key, child_value = stripped.split(":", 1)
                nested[child_key.strip()] = parse_scalar(child_value.strip())
            else:
                raise ValueError(f"{path}: unsupported nested line: {child}")
            index += 1
        data[key] = nested if nested is not None else {}
    return data


def load_papers() -> list[dict[str, Any]]:
    papers = []
    for meta_path in sorted(PAPERS_DIR.glob("*/meta.yaml")):
        meta = parse_simple_yaml(meta_path)
        missing = sorted(REQUIRED_FIELDS - set(meta))
        if missing:
            raise ValueError(f"{meta_path}: missing required fields: {', '.join(missing)}")
        paper_dir = meta_path.parent
        meta["paper_dir"] = paper_dir.relative_to(ROOT).as_posix()
        papers.append(meta)
    return papers


def paper_link(paper: dict[str, Any]) -> str:
    return f"[{paper['title']}](../{paper['paper_dir']}/README.md)"


def notes_link(paper: dict[str, Any], note_key: str, label: str) -> str:
    notes = paper.get("notes", {})
    if not isinstance(notes, dict) or note_key not in notes:
        return ""
    return f"[{label}](../{paper['paper_dir']}/{notes[note_key]})"


def row_for(paper: dict[str, Any]) -> str:
    authors = ", ".join(paper["authors"]) if isinstance(paper["authors"], list) else str(paper["authors"])
    topics = ", ".join(f"`{topic}`" for topic in paper["topics"])
    reading = notes_link(paper, "reading", "reading")
    summary = notes_link(paper, "summary", "summary")
    return (
        f"| {paper_link(paper)} | {paper['year']} | {paper['read_date']} | "
        f"`{paper['status']}` | {authors} | {topics} | {reading} / {summary} |"
    )


def write_all_papers(papers: list[dict[str, Any]]) -> None:
    lines = [
        "# All Papers",
        "",
        "| Paper | Year | Read Date | Status | Authors | Topics | Notes |",
        "|---|---:|---|---|---|---|---|",
    ]
    for paper in sorted(papers, key=lambda item: (str(item["read_date"]), item["title"]), reverse=True):
        lines.append(row_for(paper))
    (INDEX_DIR / "all-papers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_by_date(papers: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        grouped[str(paper["read_date"])].append(paper)
    lines = ["# Papers By Date", ""]
    for read_date in sorted(grouped, reverse=True):
        lines.extend([f"## {read_date}", ""])
        for paper in sorted(grouped[read_date], key=lambda item: item["title"]):
            lines.append(f"- {paper_link(paper)} (`{paper['status']}`)")
        lines.append("")
    (INDEX_DIR / "by-date.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_by_topic(papers: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        for topic in paper["topics"]:
            grouped[str(topic)].append(paper)
    lines = ["# Papers By Topic", ""]
    for topic in sorted(grouped):
        lines.extend([f"## `{topic}`", ""])
        for paper in sorted(grouped[topic], key=lambda item: (item["year"], item["title"]), reverse=True):
            lines.append(f"- {paper_link(paper)} ({paper['year']}, `{paper['status']}`)")
        lines.append("")
    (INDEX_DIR / "by-topic.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_by_status(papers: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        grouped[str(paper["status"])].append(paper)
    lines = ["# Papers By Status", ""]
    for status in sorted(grouped):
        lines.extend([f"## `{status}`", ""])
        for paper in sorted(grouped[status], key=lambda item: (str(item["read_date"]), item["title"]), reverse=True):
            lines.append(f"- {paper_link(paper)} ({paper['read_date']})")
        lines.append("")
    (INDEX_DIR / "by-status.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    papers = load_papers()
    write_all_papers(papers)
    write_by_date(papers)
    write_by_topic(papers)
    write_by_status(papers)
    print(f"Indexed {len(papers)} paper(s).")


if __name__ == "__main__":
    main()
