# Paper Read Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert this workspace into a Markdown-first paper reading library with per-paper folders, metadata, templates, and generated indexes.

**Architecture:** Each paper lives under `papers/<paper-id>/` with `meta.yaml`, source files, notes, and assets. A Python script scans `papers/*/meta.yaml` and regenerates Markdown indexes under `index/`.

**Tech Stack:** Markdown, YAML-compatible simple metadata, Python standard library.

---

### Task 1: Create structure and templates

**Files:**
- Create: `papers/`
- Create: `index/`
- Create: `templates/meta.yaml`
- Create: `templates/reading-notes.md`
- Create: `templates/summary.zh.md`
- Create: `templates/external-views.md`

- [ ] Create the directories with `mkdir -p papers index templates tools`.
- [ ] Add reusable templates for new papers.

### Task 2: Migrate MAI-Thinking-1

**Files:**
- Create: `papers/2026-06-mai-thinking-1/`
- Move: `mai-thinking-1.pdf` to `papers/2026-06-mai-thinking-1/paper.pdf`
- Move: `mai-thinking-1.en.txt` to `papers/2026-06-mai-thinking-1/source/paper.en.txt`
- Move: `mai-thinking-1.zh.md` to `papers/2026-06-mai-thinking-1/source/paper.zh.md`
- Move: `mai-thinking-1.reading-notes.md` to `papers/2026-06-mai-thinking-1/notes/reading-notes.md`
- Move: `mai-thinking-1.zh-summary.md` to `papers/2026-06-mai-thinking-1/notes/summary.zh.md`

- [ ] Create `source/`, `notes/`, and `assets/figures/`.
- [ ] Move existing MAI files into the new paper directory.
- [ ] Keep transient translation cache/sample files out of the curated paper folder.

### Task 3: Add metadata and README

**Files:**
- Create: `papers/2026-06-mai-thinking-1/meta.yaml`
- Create: `papers/2026-06-mai-thinking-1/README.md`
- Create: `README.md`

- [ ] Add metadata fields needed for indexing: `id`, `title`, `authors`, `year`, `read_date`, `status`, `topics`, `source_url`, `local_pdf`, and `notes`.
- [ ] Add a root README explaining project layout and workflow.

### Task 4: Add index builder

**Files:**
- Create: `tools/build_index.py`

- [ ] Implement a standard-library Python script that reads paper metadata.
- [ ] Generate `index/all-papers.md`, `index/by-date.md`, `index/by-topic.md`, and `index/by-status.md`.
- [ ] Fail with a clear error if required metadata fields are missing.

### Task 5: Generate and verify indexes

**Files:**
- Create/Update: `index/all-papers.md`
- Create/Update: `index/by-date.md`
- Create/Update: `index/by-topic.md`
- Create/Update: `index/by-status.md`

- [ ] Run `python3 tools/build_index.py`.
- [ ] Verify generated index files include `MAI-Thinking-1`.
- [ ] Verify the moved reading notes remain accessible in the new location.
