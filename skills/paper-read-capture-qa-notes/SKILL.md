---
name: paper-read-capture-qa-notes
description: Use when turning multi-turn discussion, Q&A, explanations, paper-reading dialogue, or user/assistant analysis into notes/reading-notes.md in this paper_read repo.
---

# Paper Read Capture QA Notes

Append paper-discussion Q&A into the relevant paper's `notes/reading-notes.md` as durable reading notes.

## Workflow

1. Identify the target paper directory. Prefer the paper currently being discussed; otherwise search with `rg` over `papers/*/meta.yaml`, `README.md`, and `notes/`.
2. Read the current `notes/reading-notes.md` before editing. Preserve all existing user notes.
3. Summarize the discussion into notes, not a chat transcript. Keep the user's question, the answer's main reasoning, examples, caveats, and action implications.
4. Add a new section or append to the most relevant existing section. Prefer appending under an existing indexed section when the discussion is a continuation of that topic. Do not rewrite the whole file unless explicitly asked.
5. Update the note index whenever adding a new `##` section. If adding a `###` subsection under an indexed `##` section, either keep it clearly inside that indexed section or add an index entry when it should be directly jumpable.
6. Use explicit ASCII anchors for index links when headings contain Chinese punctuation, English acronyms, slashes, parentheses, or mixed scripts:

```markdown
- [章节标题](#stable-ascii-anchor)

<a id="stable-ascii-anchor"></a>
## 章节标题
```

7. Distinguish:
   - `论文明确说的内容`: claims, methods, numbers, and reported experiments from the paper.
   - `推断 / 个人理解`: interpretation, transfer ideas, design advice, and speculation.
8. Preserve source links for external articles, blogs, videos, or discussion threads. If no external source is used, do not invent one.
9. This workflow is Markdown-only. Do not run `tools/build_index.py` unless `meta.yaml` changed. Do not add specs unless Python code changes.
10. Verify by reading the edited range and using `rg` for new anchors/headings.

## Red Flags

- A newly added discussion heading exists but the note index cannot jump to it, and it is not clearly nested under an already indexed section.
- A new index link points to a generated Chinese or punctuation-heavy heading anchor instead of an explicit ASCII `<a id="..."></a>` anchor.
- The note reads like a chat transcript rather than durable paper-reading notes.

## Note Style

- Prefer concise Chinese explanations with concrete examples.
- Keep formulas, labels, commands, file paths, and model names in backticks when useful.
- Capture decision boundaries and failure modes explicitly, especially for ambiguous labels or training recipes.
- Preserve uncertainty: use "论文明确报告..." for evidence from the paper and "个人理解/推断..." for extrapolation.

## Triggers

- "把上述讨论加入 notes"
- "补充到读书笔记"
- "把这几轮 QA 总结进 reading-notes"
- "记录一下我们刚才关于这篇论文的理解"
