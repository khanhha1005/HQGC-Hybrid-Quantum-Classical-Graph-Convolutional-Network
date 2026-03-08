# Vulnerability Detection Bar Charts for IEEE

## Generated Figures

Run `python plot_vulnerability_results.py` to generate:

| File | Description |
|------|-------------|
| `vuln_bar_chart_grouped.pdf` | **Main figure** – Two panels: (a) Reentrancy, (b) Timestamp Dependence. Grouped bars for Acc, Recall, Prec, F1. |
| `vuln_F1_comparison.pdf` | Single-column F1 score comparison (fits IEEE 3.5" column width) |
| `vuln_all_metrics.pdf` | 2×2 layout with all four metrics |

## LaTeX Inclusion

**For single-column (F1 only):**
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/vuln_F1_comparison.pdf}
  \caption{F1 score comparison across methods for Reentrancy and Timestamp Dependence vulnerabilities.}
  \label{fig:vuln_f1}
\end{figure}
```

**For full-width (grouped bars):**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/vuln_bar_chart_grouped.pdf}
  \caption{Per-vulnerability detection performance (\%). (a) Reentrancy. (b) Timestamp Dependence.}
  \label{fig:vuln_results}
\end{figure*}
```

## Data Source

Based on Table: Per-vulnerability detection performance (\%).
