---
name: LaTeX style and formatting conventions
description: Exact formatting parameters, packages, table style, label naming, and typographic conventions used in the report
type: project
---

## Document class and geometry
- `\documentclass[a4paper,12pt]{extarticle}`
- Margins: left 2.5cm, right 1.0cm, top 2.0cm, bottom 2.0cm
- Indent: `\setlength{\parindent}{1.25cm}`
- Line spacing: `\renewcommand{\baselinestretch}{1.5}`
- Page numbers: bottom center via fancyhdr

## Fonts
- `\usepackage{fontspec}` + `\setmainfont{Times New Roman}` (XeLaTeX)
- Language: `\usepackage[english]{babel}` — document is in English

## Key packages
- `booktabs` — tables use `\toprule/\midrule/\bottomrule` (NO `\hline`)
- `enumitem` — lists use `[leftmargin=*]` or `[label=FR\arabic*., leftmargin=*]`
- `biblatex` (backend=biber, style=numeric, sorting=none, maxbibnames=99)
- `listings` — code with frame=single, numbers=left, basicstyle=\small\ttfamily
- `threeparttable`, `tablefootnote`, `colortbl`, `tikz`, `pgf`, `subcaption`
- `chngcntr` — tables and figures numbered per-section (e.g. Table 3.1)

## Table style
- Always `\begin{table}[ht]` with `\caption{...}` BEFORE `\label{...}` BEFORE `\centering`
- Column types: plain `l`, `c`, `p{Xcm}` — no custom column types
- No vertical rules (`|`)
- Three-rule style: `\toprule`, `\midrule`, `\bottomrule`

## Label naming conventions
- Tables: `\label{table:snake_case}`
- Figures: `\label{fig:snake_case}`
- Sections: referenced as `Section~\ref{...}` (tilde non-breaking space)

## Section structure (as of 2026-03-31, updated)
1. Abstract (unnumbered, \addcontentsline)
2. Keywords (unnumbered)
3. Introduction (Background, Problem Statement, Objectives, Scope)
4. Related Work (TinyML Frameworks, Neural Architectures, Edge/IIoT, Monitoring, LLM on edge, Gap Analysis)
5. Project Description:
   - System Requirements (FR, NFR)
   - Hardware Platform: Rockchip RK3568 SoC (new, added 2026-03-31)
   - System Architecture (Component Overview, Data Format, Data Processing Pipeline)
   - Use Cases
   - Neural Network Architectures (expanded with Table 3.2, equations)
   - Model Export, Conversion, and Quantization (expanded with Table 3.3)
   - Benchmarking Infrastructure
   - Vision Model Benchmarks
   - LLM Experiment
   - ...
6. Experimental Methodology and Results
   ...
   - Vision Model Benchmark Results (\label{sec:vision_results}): Table table:vision_latency (pending)
   - LLM Experiment Results (\label{sec:llm_results}): Tables table:llm_qwen, table:llm_gemma_fc, table:llm_gemma_text, table:llm_comparison
   - OPC UA Communication Layer (new, added 2026-03-31 — server.c, client.py, client.c, data flow)
   - Observability Stack (\label{sec:observability})
   - Implementation Summary
   - Quantitative Characteristics
6. Experimental Methodology and Results
7. Conclusion (Summary, Key Findings, Future Directions)
8. References

## Listings label naming
- `\label{lst:snake_case}` — code listings
- `language=bash`, `language=C`, `language=Python` supported; yaml/docker NOT supported by listings — use bare lstlisting

## Known bib key additions (2026-03-31)
- open62541, iec62541 (@misc), python_opcua, rk3568_npu_spec
