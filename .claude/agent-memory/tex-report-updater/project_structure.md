---
name: Document repository structure and build system
description: Location, file structure, document class, and build toolchain for the embedded-AI coursework LaTeX report
type: project
---

Repository: `/Users/shevlfs/embedded-ai-coursework-documents/`

Files:
- `main.tex` — single-file document (no \input chapters except title page)
- `title_kr.tex` — title page include
- `refs.bib` — BibLaTeX bibliography (biblatex + biber, style=numeric, sorting=none)
- `Makefile` — builds with `xelatex` + `biber` (3-pass: xelatex, biber, xelatex, xelatex)
- `graphics/` — figures directory (referenced via \graphicspath{{graphics/}})

**Why:** Single-file structure means all edits go into `main.tex` directly; no chapter files to hunt for.
**How to apply:** Always edit `main.tex` for content changes; add bib entries to `refs.bib`.
