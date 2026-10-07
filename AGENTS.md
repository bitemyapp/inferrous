# Agent instructions

## Markdown formatting

- Never hard-wrap Markdown prose. Write each paragraph, list item, blockquote paragraph and table row on a single line, however long it is, and let the renderer wrap it.
- Separate blocks (paragraphs, lists, tables, headings, code fences) with one blank line. Line breaks inside fenced code blocks are fine.

## Documentation layout

- All documentation lives under `docs/`; only `README.md` and `AGENTS.md` stay at the repository root. Keep `docs/README.md` up to date as the index.
- Engine documentation goes in `docs/<engine>/` (for example `docs/qwen3-8b-w8a8/`): optimization records in `optimization/round-N.md` and generated result summaries in `results/`. Plans go in `docs/plans/`, research in `docs/research/`.
