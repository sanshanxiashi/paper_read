---
name: paper-read-init-paper
description: Use when adding, importing, downloading, initializing, or organizing a new paper in this paper_read repo, especially when given a PDF path, arXiv URL, paper URL, or request to create the paper directory and metadata.
---

# Paper Read Init Paper

Initialize one paper in this Markdown-first paper library without leaving formal paper files in the repository root.

## Workflow

1. Read project instructions first: `AGENTS.md`, then inspect `templates/` and nearby existing `papers/*/meta.yaml`.
2. Determine `paper-id` as `YYYY-MM-short-slug`. Prefer a stable title-derived lowercase slug; do not use raw arXiv id unless explicitly requested.
3. Create the standard structure:

```text
papers/<paper-id>/
  meta.yaml
  README.md
  paper.pdf
  source/paper.en.txt
  source/paper.zh.md
  notes/reading-notes.md
  notes/summary.zh.md
  notes/external-views.md
  assets/figures/
```

4. If the PDF is already in the repo, move it to `papers/<paper-id>/paper.pdf`. Preserve user files; do not delete paper PDFs, notes, translations, metadata, or generated indexes unless explicitly asked.
5. Extract English text to `source/paper.en.txt`, preferably with `pdftotext -layout`. Do not only preview PDF text on stdout; write or refresh the file under `source/`. If extraction fails, state the failure and leave a clear placeholder.
6. Fill `meta.yaml` from `templates/meta.yaml`:
   - `read_date`: today.
   - `status`: `queued` unless the user is already reading/discussing it, then `reading`.
   - `source_url`: canonical paper URL, preferably arXiv abs page for arXiv papers.
   - `local_pdf`: `paper.pdf`.
7. Fill `README.md` with title, URL, file list, topics, and a short initial定位.
8. Create starter notes:
   - `notes/summary.zh.md`: concise Chinese导读 with paper info, one-sentence summary, key points, contributions, reading suggestions.
   - `notes/reading-notes.md`: note index, core question, paper claims, interpretation, open questions.
   - `notes/external-views.md`: source links; pending external commentary if none.
   - `source/paper.zh.md`: placeholder unless full translation is requested.
9. After any `meta.yaml` addition/change, run `python3 tools/build_index.py` from repo root. Confirm successful exit before saying indexes were updated.
10. Verify with `find papers/<paper-id> -maxdepth 3 -type f | sort`, `wc -l papers/<paper-id>/source/paper.en.txt`, and `rg` for the paper in `index/`. Run `git status --short` when the workspace is a git repository; in disposable or non-git fixtures, report that git status is not applicable instead of treating it as a failure.

## Rules

- Keep `README.md` URL consistent with `meta.yaml:source_url`.
- Use stable lowercase English topic slugs. Prefer existing AGENTS.md topics; add domain slugs only when useful.
- Very large author lists may be shortened to the team plus representative authors if full metadata would be noisy.
- This workflow is Markdown/data/PDF only. If Python code must change, obey the repo spec-first rule before editing.

## Triggers

- "我下载了这篇论文，按规范初始化"
- "新增一篇 paper"
- "把这个 PDF 放进库里"
- "create paper directory / metadata / notes for this paper"
