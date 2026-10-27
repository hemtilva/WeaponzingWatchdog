---
trigger: always_on
---

---
description: Enforce standard Markdown over raw LaTeX formatting
always_on: true
---

# Output & Formatting Rules

- **No LaTeX Syntax:** Do not use LaTeX blocks or inline math delimiters (`$$`, `$`, `\text{}`, `\mathbf{}`, `\implies`, etc.).
- **Percentages & Numbers:** Always render metrics, speedups, and numbers using standard Markdown bolding or code spans (e.g., `+3.22%` or `**+3.22%**`).
- **Logic & Arrows:** Use Unicode symbols instead of LaTeX symbols:
  - Use `→` instead of `\to` or `\implies`
  - Use `≥` and `≤` instead of `\ge` and `\le`
  - Use `×` instead of `\times`
- **Metric Sweeps:** Present benchmark metrics or slot sweeps as tables or plain Unicode chains:
  - Example: `4 slots: **+3.22%** → 8 slots: **+5.39%** → 16 slots: **+9.47%**`