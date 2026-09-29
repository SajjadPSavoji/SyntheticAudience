# AutoPolish — ACCV 2026 / CV4DC workshop submission

Port of the NeurIPS Creative AI submission (`../neurips_creative_ai/`) into the
official ACCV 2026 LNCS template. Body text, figures, tables and the whole
supplementary material are unchanged; only the formatting layer was rewritten.

## Files

| file | what it is |
| --- | --- |
| `autopolish.tex` | the paper — main text **and** supplement |
| `autopolish_main.tex` | wrapper that defines `\mainonly` and skips the supplement |
| `accv.sty`, `accvabbrv.sty`, `llncs.cls`, `splncs04.bst` | official ACCV 2026 template, unmodified |
| `ACCV_template_reference.tex`, `TEMPLATE_README.md`, `lncs_readme.txt` | the template's own example paper and docs, kept for reference |
| `figs` | symlink to `../neurips_creative_ai/figs` (24 MB, not duplicated) |

Template source: <https://accv2026.org/wp-content/uploads/2026/04/ACCV_2026_template.zip>

## Build

```
latexmk -pdf autopolish_main.tex   # main paper only  -> 12 pp, ~6 MB
latexmk -pdf autopolish.tex        # paper + supplement -> 34 pp, ~27 MB
```

Both compile clean: no errors, no overfull boxes, no undefined references.

## What changed from the NeurIPS version

* `\documentclass{article}` + `neurips_2026.sty` → `\documentclass[runningheads]{llncs}` + `accv.sty`.
  The `[review]` option anonymises the author block, stamps the paper ID on every
  page and turns on line numbers, all required for double-blind ACCV review.
* Title/author/institute rewritten in LNCS form (`\titlerunning`, `\authorrunning`,
  `\institute`). Real names are kept in the source; `[review]` hides them.
* `\keywords{...}` added to the abstract (required for LNCS proceedings).
* natbib dropped (LNCS uses the `cite` package): all 68 `\citep` → `\cite`.
  A `\providecommand{\citep}{\cite}` alias remains as a safety net.
* Run-in headings `\paragraph{...}` → `\subsubsection{...}`, the LNCS 3rd level.
* The NeurIPS `\begin{ack}` block was removed (no such environment in LNCS);
  a `TODO FINAL` marker sits where acknowledgements go in the camera-ready.
* `caption`/`subcaption`/`amsmath`/`amssymb`/`xcolor`/`url`/`fontenc` removed from
  the preamble — `accv.sty` already loads them.
* The LNCS text block is narrower than NeurIPS's, so three tables and the pipeline
  figure were refit (stacked headers, `\footnotesize`, TikZ scale 0.84 → 0.82).
  No numbers or wording were touched.

## Before submitting

1. **Track / length.** Main text currently runs ~9 pages excluding references
   (12 pp total). That fits the **Proceedings Track** (14 pp excl. refs, deadline
   Oct 9 2026). The **Non-Proceedings Track** allows only 4–7 pp excl. refs, so
   it would need substantial cutting.
2. **Paper ID.** Replace `ID=*****` in `autopolish.tex` once OpenReview assigns one.
3. **Abstract length.** LNCS recommends ~150 words; the current abstract is ~390.
4. **Bibliography.** Still a hand-written `thebibliography`, not `splncs04.bst`.
   It renders close to LNCS style but is not exactly it — worth converting to a
   `.bib` + `\bibliographystyle{splncs04}` before camera-ready (`splncs04.bst` is
   already in this directory).
5. **PDF size.** ACCV asks for < 10 MB. `autopolish_main.pdf` is 6.2 MB (fine);
   the combined file with the supplement is 27 MB, so submit the supplement
   separately or downsample `figs/`.
6. **Anonymity.** The `[review]` option handles the author block, but check the
   supplement for any remaining identifying text before uploading.
