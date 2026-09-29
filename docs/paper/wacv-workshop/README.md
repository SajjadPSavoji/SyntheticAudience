# AutoPolish — WACV 2027 workshop (WVAQ) draft

Port of `docs/paper/neurips_creative_ai/autopolish.tex` onto the **official WACV 2027
author kit** (`wacv.sty`, downloaded from
<https://wacv.thecvf.com/Conferences/2027/AuthorGuides>).
Prose, numbers, figures, tables and the full appendix are unchanged; only the
layout machinery was rewritten.

## Building

```
latexmk -pdf autopolish.tex        # main paper + supplementary material (21 pp.)
latexmk -pdf autopolish_main.tex   # main paper only (8 pp.), for the page limit
```

Both build clean: no overfull boxes, no float warnings.

## Target venue

Workshop on Visual Aesthetics and Quality (WVAQ) @ WACV 2027 —
<https://wacv2027-image-quality-workshop.github.io>

| | |
|---|---|
| Template | same as the WACV 2027 main conference |
| Review | **single-blind** — author names *are* included, no anonymization |
| Submission | OpenReview, `thecvf.com/WACV/2027/Workshop/WVAQ` |
| Deadline | 12 Oct 2026 (notification 23 Oct, camera-ready 20 Nov) |
| Page limit | not stated by the workshop; WACV main is 8 pages **excluding** references |

## Style options

`autopolish.tex` currently uses `\usepackage[pagenumbers]{wacv}`: camera-ready
title block (names shown, as single-blind requires) plus page numbers for the
reviewers. Alternatives are commented in the preamble:

* `\usepackage{wacv}` — camera-ready (drop `pagebackref` from the `hyperref`
  options at the same time).
* `\usepackage[review,algorithms]{wacv}` — anonymous, line-numbered. Not what
  WVAQ asks for; kept in case a sister venue is double-blind.

## What changed relative to the NeurIPS source

* Preamble rebuilt for `wacv.sty`. Packages the style already loads (`xcolor`,
  `graphicx`, `amsmath`, `amssymb`, `booktabs`, `natbib`, `caption`,
  `subcaption`, `url`, `enumitem`, `cleveref`) were removed to avoid option
  clashes; `tikz`, `tcolorbox`, `tabularx`, `nicefrac`, `microtype`, `amsfonts`
  stay. `hyperref` is loaded last, as the kit requires.
* Author block converted to the CVF `\and` layout, with `\thanks` footnotes for
  "equal contribution" / "corresponding author".
* NeurIPS `\begin{ack}` → `\section*{Acknowledgments}` (**placeholder — fill in**).
* Every float that was laid out at NeurIPS full text width is now a
  two-column-spanning `figure*` / `table*`. Narrow supplement plots stay
  single-column.
* Figure 1 (TikZ overview) is wrapped in `\resizebox{\textwidth}` and declared
  right after `\section{Introduction}` so it lands at the top of page 2.
* `figs/ax_qualitative2.png` and `figs/ax_progression.png` are capped at
  `0.84\textheight` so their captions fit on the page.
* Supplementary title page uses the kit's own `\maketitlesupplementary`;
  `S`-prefixed section/figure/table numbering is unchanged.
* The prompt/worked-example sections at the end of the supplement are set
  `\onecolumn` — the verbatim monospaced transcripts cannot fit a 3.25 in column.

## Open items

* Acknowledgments placeholder in `autopolish.tex`.
* The bibliography is still the hand-written `thebibliography` block from the
  NeurIPS source. It renders fine, but the kit ships `ieeenat_fullname.bst` if
  you would rather match CVF reference formatting exactly.
* `pagebackref` in the `hyperref` options is a reviewer convenience; remove for
  camera-ready.
