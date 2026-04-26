# Notebook Continuous Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `model_comparison.ipynb` and `confidence_analysis.ipynb` to handle continuous (regression) artifacts alongside discrete (classification) ones in separate sections.

**Architecture:** Split artifacts at load time into `artifacts_discrete` / `artifacts_cont` dicts. Existing sections operate unchanged on `artifacts_discrete`. New sections mirror the structure with regression metrics (`ic` primary, `mae`, `rmse`, `r2`, `directional_accuracy`, `mean_abs_pred`). All notebook edits are done via Python scripts that manipulate the notebook JSON directly to avoid fragile manual cell insertion.

**Tech Stack:** Python, Jupyter notebook JSON (`nbformat`), joblib, numpy, pandas, matplotlib, seaborn, scipy, torch (for neural continuous inference)

---

## File map

| File | Change |
|---|---|
| `models/evaluate.py` | Already done — `mean_abs_pred` added to `evaluate_regression()` |
| `notebooks/model_comparison.ipynb` | Split at load; existing sections → `artifacts_discrete`; add Section 2 continuous |
| `notebooks/confidence_analysis.ipynb` | Split at load; Parts A/B → `artifacts_discrete`; add Part C continuous |
| `notebooks/patch_model_comparison.py` | Scratch script to apply patches (delete after use) |
| `notebooks/patch_confidence_analysis.py` | Scratch script to apply patches (delete after use) |

---

## Task 1: Verify evaluate.py baseline

**Files:**
- Read: `models/evaluate.py`
- Run: `tests/test_evaluate.py`

- [ ] **Step 1: Confirm mean_abs_pred is present**

```bash
grep mean_abs_pred models/evaluate.py
```

Expected output: `"mean_abs_pred":       float(np.mean(np.abs(y_pred))),`

- [ ] **Step 2: Run tests to confirm all pass**

```bash
pip install pytest -q
python -m pytest tests/test_evaluate.py -v
```

Expected: `7 passed`

- [ ] **Step 3: Commit if not already committed**

If git status shows evaluate.py modified:
```bash
git add models/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): add mean_abs_pred to evaluate_regression output"
```

---

## Task 2: Patch model_comparison.ipynb — split + refactor existing sections

**Files:**
- Create: `notebooks/patch_model_comparison.py`
- Modify: `notebooks/model_comparison.ipynb`

The patch script replaces the artifact loading cell to add target_type to the print + split, and replaces cell 4 (build summary DataFrames) to use `artifacts_discrete`. Everything else in sections 1–10 reads from `summary` and `windows_df` which will now be built from discrete artifacts only — no other cells need changing.

- [ ] **Step 1: Write the patch script**

Create `notebooks/patch_model_comparison.py` with this content:

```python
"""Patch model_comparison.ipynb: split artifacts_discrete/cont at load time."""
import json, pathlib, copy

NB = pathlib.Path("notebooks/model_comparison.ipynb")
data = json.loads(NB.read_text(encoding="utf-8"))
cells = data["cells"]

# ── helper ────────────────────────────────────────────────────────────────────
def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }

def find_cell(cells, snippet):
    for i, c in enumerate(cells):
        if snippet in "".join(c.get("source", [])):
            return i
    raise ValueError(f"Cell with snippet not found: {snippet!r}")

# ── 1. Replace load cell (cell 3) — add target_type to print ─────────────────
idx_load = find_cell(cells, "def load_artifacts")
cells[idx_load]["source"] = (
    'def load_artifacts(path: pathlib.Path) -> dict:\n'
    '    """Load all new-format artifacts (those with cv_summary) keyed by filename stem."""\n'
    '    arts = {}\n'
    '    for f in sorted(path.glob(\'*.pkl\')):\n'
    '        a = joblib.load(f)\n'
    '        if \'cv_summary\' not in a:\n'
    '            print(f\'  skip (old format): {f.name}\')\n'
    '            continue\n'
    '        arts[f.stem] = a\n'
    '        tt = a.get("target_type", "discrete")\n'
    '        print(f\'  loaded: {f.stem:20s}  h={a["horizon"]}  mode={a["mode"]}  \'\n'
    '              f\'ft={a["ft_type"]}  target={tt}  windows={len(a["cv_metrics"])}\')\n'
    '    print(f\'\\n{len(arts)} artifact(s) loaded.\')\n'
    '    return arts\n'
    '\n'
    'artifacts = load_artifacts(ARTIFACTS_PATH)\n'
)

# ── 2. Insert split cell immediately after load cell ──────────────────────────
split_cell = code_cell(
    '# Split by target type — each section below operates on its own dict\n'
    'artifacts_discrete = {k: v for k, v in artifacts.items()\n'
    '                      if v.get("target_type", "discrete") == "discrete"}\n'
    'artifacts_cont     = {k: v for k, v in artifacts.items()\n'
    '                      if v.get("target_type", "continuous") == "continuous"}\n'
    '\n'
    'print(f"Discrete: {len(artifacts_discrete)}  |  Continuous: {len(artifacts_cont)}")\n'
)
cells.insert(idx_load + 1, split_cell)

# ── 3. Replace build-summary cell — use artifacts_discrete ────────────────────
# After insertion above, cell 4 shifted by 1; find by snippet instead
idx_sum = find_cell(cells, "Build a flat summary DataFrame")
cells[idx_sum]["source"] = (
    '# Build a flat summary DataFrame  (one row per artifact) — DISCRETE only\n'
    'METRICS = [\'balanced_accuracy\', \'roc_auc\', \'log_loss\', \'mcc\', \'accuracy\']\n'
    '\n'
    'if not artifacts_discrete:\n'
    '    print("No discrete artifacts loaded — skipping Section 1.")\n'
    '    summary = pd.DataFrame()\n'
    '    windows_df = pd.DataFrame()\n'
    'else:\n'
    '    rows = []\n'
    '    for stem, a in artifacts_discrete.items():\n'
    '        row = {\n'
    '            \'name\':    stem,\n'
    '            \'model\':   a[\'model_key\'],\n'
    '            \'horizon\': a[\'horizon\'],\n'
    '            \'mode\':    a[\'mode\'],\n'
    '            \'ft_type\': a[\'ft_type\'],\n'
    '            \'windows\': len(a[\'cv_metrics\']),\n'
    '            \'train_start\': a[\'train_start\'],\n'
    '            \'train_end\':   a[\'train_end\'],\n'
    '        }\n'
    '        for m in METRICS:\n'
    '            s = a[\'cv_summary\'].get(m, {})\n'
    '            row[f\'{m}_mean\'] = s.get(\'mean\', np.nan)\n'
    '            row[f\'{m}_std\']  = s.get(\'std\',  np.nan)\n'
    '        rows.append(row)\n'
    '\n'
    '    summary = pd.DataFrame(rows).set_index(\'name\')\n'
    '\n'
    '    window_rows = []\n'
    '    for stem, a in artifacts_discrete.items():\n'
    '        for wi, m in enumerate(a[\'cv_metrics\']):\n'
    '            window_rows.append({\n'
    '                \'name\':    stem,\n'
    '                \'model\':   a[\'model_key\'],\n'
    '                \'horizon\': a[\'horizon\'],\n'
    '                \'mode\':    a[\'mode\'],\n'
    '                \'window\':  wi,\n'
    '                **{k: v for k, v in m.items() if k in METRICS},\n'
    '            })\n'
    '    windows_df = pd.DataFrame(window_rows)\n'
    '\n'
    '    print(\'Summary shape:\', summary.shape)\n'
    '    print(\'Windows long df shape:\', windows_df.shape)\n'
)

# ── 4. Wrap all existing section cells in `if not summary.empty:` ─────────────
# Find cells 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 24 by snippets and prepend guard
GUARD_SNIPPETS = [
    ("def fmt(mean, std)",          "if summary.empty:\n    print('No discrete artifacts.')\nelse:\n "),
    ("def boxplot_metric(",         "if windows_df.empty:\n    print('No discrete artifacts.')\nelse:\n "),
    ("for horizon in sorted(windows_df",   "if windows_df.empty:\n    print('No discrete artifacts.')\nelse:\n "),
    ("for (horizon, mode), grp in windows_df",  "if windows_df.empty:\n    print('No discrete artifacts.')\nelse:\n "),
    ("tree_arts = {k: v for k, v in artifacts.items()",  None),  # replace artifacts with artifacts_discrete
    ("meta_rows = []",              "if not artifacts_discrete:\n    print('No discrete artifacts.')\nelse:\n "),
    ("if len(windows_df) > 0:",     None),  # already has guard, no change needed
    ("s_df = summary[summary['mode'] == 'sliding']", "if summary.empty:\n    print('No discrete artifacts.')\nelse:\n "),
    ("def model_complexity(",       "if not artifacts_discrete:\n    print('No discrete artifacts.')\nelse:\n "),
]

# Fix tree_arts to use artifacts_discrete
idx_tree = find_cell(cells, "tree_arts = {k: v for k, v in artifacts.items()")
cells[idx_tree]["source"] = cells[idx_tree]["source"].replace(
    "tree_arts = {k: v for k, v in artifacts.items()",
    "tree_arts = {k: v for k, v in artifacts_discrete.items()"
)

# Fix meta_rows to use artifacts_discrete
idx_meta = find_cell(cells, "meta_rows = []")
cells[idx_meta]["source"] = cells[idx_meta]["source"].replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# Fix model_complexity cell — two cells use artifacts
idx_cplx1 = find_cell(cells, "def model_complexity(")
# The next code cell after model_complexity has the loop over artifacts
idx_cplx2 = idx_cplx1 + 1
while cells[idx_cplx2]["cell_type"] != "code":
    idx_cplx2 += 1
cells[idx_cplx2]["source"] = cells[idx_cplx2]["source"].replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# Save
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched model_comparison.ipynb — split + discrete refactor done.")
```

- [ ] **Step 2: Run the patch**

```bash
cd notebooks && python patch_model_comparison.py && cd ..
```

Expected: `Patched model_comparison.ipynb — split + discrete refactor done.`

- [ ] **Step 3: Verify notebook is valid JSON**

```bash
python -c "import json; data=json.load(open('notebooks/model_comparison.ipynb')); print(len(data['cells']), 'cells — valid JSON')"
```

Expected: `26 cells — valid JSON` (25 original + 1 split cell inserted)

- [ ] **Step 4: Commit**

```bash
git add notebooks/model_comparison.ipynb notebooks/patch_model_comparison.py
git commit -m "feat(notebook): split artifacts_discrete/cont at load in model_comparison"
```

---

## Task 3: Add Section 2 (Continuous) to model_comparison.ipynb

**Files:**
- Modify: `notebooks/patch_model_comparison.py` (rewrite to append Section 2 cells)
- Modify: `notebooks/model_comparison.ipynb`

Section 2 mirrors Section 1 with regression metrics: `ic` (primary), `mae`, `rmse`, `r2`, `directional_accuracy`. Subsections: header, build DataFrames, CV Summary Table, Boxplots, Temporal Trend, Head-to-Head Win Rate, Metadata, Sliding vs Expanding, Complexity vs IC.

- [ ] **Step 1: Rewrite patch script to append Section 2**

Overwrite `notebooks/patch_model_comparison.py` with:

```python
"""Append Section 2 (continuous artifacts) to model_comparison.ipynb."""
import json, pathlib

NB = pathlib.Path("notebooks/model_comparison.ipynb")
data = json.loads(NB.read_text(encoding="utf-8"))
cells = data["cells"]

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

new_cells = []

# ── Section 2 header ──────────────────────────────────────────────────────────
new_cells.append(md_cell(
    '---\n'
    '# Section 2 — Continuous Artifacts (Regression)\n'
    'Mirrors Section 1 but with regression metrics: **IC** (primary, Spearman), MAE, RMSE, R², directional accuracy.\n'
))

# ── 2.0 Build continuous DataFrames ──────────────────────────────────────────
new_cells.append(md_cell('## 2.0  Load continuous artifacts into DataFrames'))
new_cells.append(code_cell(
    'CONT_METRICS = [\'ic\', \'mae\', \'rmse\', \'r2\', \'directional_accuracy\']\n'
    '\n'
    'if not artifacts_cont:\n'
    '    print("No continuous artifacts loaded — Section 2 skipped.")\n'
    '    summary_c = pd.DataFrame()\n'
    '    windows_c = pd.DataFrame()\n'
    'else:\n'
    '    rows_c = []\n'
    '    for stem, a in artifacts_cont.items():\n'
    '        row = {\n'
    '            \'name\':    stem,\n'
    '            \'model\':   a[\'model_key\'],\n'
    '            \'horizon\': a[\'horizon\'],\n'
    '            \'mode\':    a[\'mode\'],\n'
    '            \'ft_type\': a[\'ft_type\'],\n'
    '            \'windows\': len(a[\'cv_metrics\']),\n'
    '            \'train_start\': a[\'train_start\'],\n'
    '            \'train_end\':   a[\'train_end\'],\n'
    '        }\n'
    '        for m in CONT_METRICS:\n'
    '            s = a[\'cv_summary\'].get(m, {})\n'
    '            row[f\'{m}_mean\'] = s.get(\'mean\', np.nan)\n'
    '            row[f\'{m}_std\']  = s.get(\'std\',  np.nan)\n'
    '        rows_c.append(row)\n'
    '\n'
    '    summary_c = pd.DataFrame(rows_c).set_index(\'name\')\n'
    '\n'
    '    window_rows_c = []\n'
    '    for stem, a in artifacts_cont.items():\n'
    '        for wi, m in enumerate(a[\'cv_metrics\']):\n'
    '            window_rows_c.append({\n'
    '                \'name\':    stem,\n'
    '                \'model\':   a[\'model_key\'],\n'
    '                \'horizon\': a[\'horizon\'],\n'
    '                \'mode\':    a[\'mode\'],\n'
    '                \'window\':  wi,\n'
    '                **{k: v for k, v in m.items() if k in CONT_METRICS},\n'
    '            })\n'
    '    windows_c = pd.DataFrame(window_rows_c)\n'
    '\n'
    '    print(\'Continuous summary shape:\', summary_c.shape)\n'
    '    print(\'Continuous windows df shape:\', windows_c.shape)\n'
))

# ── 2.1 CV Metrics Summary Table ──────────────────────────────────────────────
new_cells.append(md_cell('## 2.1  CV Metrics Summary Table'))
new_cells.append(code_cell(
    'if summary_c.empty:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    def fmt_c(mean, std):\n'
    '        return f"{mean:.4f} ± {std:.4f}"\n'
    '\n'
    '    disp_cols_c = [\'model\', \'horizon\', \'mode\', \'ft_type\', \'windows\']\n'
    '    sc = summary_c.copy()\n'
    '    for m in CONT_METRICS:\n'
    '        sc[m] = sc.apply(lambda r: fmt_c(r[f\'{m}_mean\'], r[f\'{m}_std\']), axis=1)\n'
    '\n'
    '    sc[disp_cols_c + CONT_METRICS].sort_values(\n'
    '        [\'horizon\', \'ic_mean\'], ascending=[True, False]\n'
    '    ).style \\\n'
    '        .set_caption(\'Continuous CV metrics  (mean ± std across windows)\') \\\n'
    '        .set_table_styles([{\'selector\': \'caption\', \'props\': \'font-size:13px; font-weight:bold;\'}])\n'
))

# ── 2.2 Per-Window Distributions (boxplots) ───────────────────────────────────
new_cells.append(md_cell('## 2.2  Per-Window Distributions  (boxplots)'))
new_cells.append(code_cell(
    'if windows_c.empty:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    primary_c = [\'ic\', \'directional_accuracy\', \'mae\', \'rmse\']\n'
    '    fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n'
    '    for ax, metric in zip(axes.flat, primary_c):\n'
    '        order = (\n'
    '            windows_c.groupby(\'name\')[metric].median()\n'
    '              .sort_values(ascending=(metric in (\'mae\', \'rmse\'))).index.tolist()\n'
    '        )\n'
    '        palette = {n: sns.color_palette(\'tab10\')[i % 10] for i, n in enumerate(order)}\n'
    '        sns.boxplot(data=windows_c, x=metric, y=\'name\', order=order,\n'
    '                    palette=palette, width=0.55, linewidth=0.9,\n'
    '                    flierprops=dict(marker=\'x\', markersize=4), ax=ax)\n'
    '        ax.set_title(metric.replace(\'_\', \' \').title(), fontsize=11)\n'
    '        ax.set_xlabel(\'\'); ax.set_ylabel(\'\')\n'
    '        if metric == \'ic\':\n'
    '            ax.axvline(0, color=\'red\', lw=0.8, ls=\'--\', label=\'IC=0\')\n'
    '            ax.legend(fontsize=8)\n'
    '    plt.suptitle(\'Continuous: per-window CV distribution by model\', fontsize=13, y=1.01)\n'
    '    plt.tight_layout(); plt.show()\n'
))

# ── 2.3 IC over CV Windows (temporal trend) ───────────────────────────────────
new_cells.append(md_cell('## 2.3  IC over CV Windows  (temporal trend)'))
new_cells.append(code_cell(
    'if windows_c.empty:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    for horizon in sorted(windows_c[\'horizon\'].unique()):\n'
    '        df_h = windows_c[windows_c[\'horizon\'] == horizon]\n'
    '        fig, ax = plt.subplots(figsize=(13, 4))\n'
    '        for name, grp in df_h.groupby(\'name\'):\n'
    '            ax.plot(grp[\'window\'], grp[\'ic\'], marker=\'o\',\n'
    '                    markersize=4, lw=1.5, label=name)\n'
    '        ax.axhline(0, color=\'red\', lw=0.8, ls=\'--\', label=\'IC=0\')\n'
    '        ax.set_title(f\'IC per CV window — h={horizon}\', fontsize=12)\n'
    '        ax.set_xlabel(\'CV window index\')\n'
    '        ax.set_ylabel(\'Spearman IC\')\n'
    '        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(\'%.3f\'))\n'
    '        ax.legend(loc=\'upper right\', fontsize=8, ncol=2)\n'
    '        plt.tight_layout(); plt.show()\n'
))

# ── 2.4 Head-to-Head Win Rate ─────────────────────────────────────────────────
new_cells.append(md_cell(
    '## 2.4  Head-to-Head Win Rate\n'
    'For each pair of continuous models (same horizon + mode), fraction of CV windows where one beats the other on IC.\n'
))
new_cells.append(code_cell(
    'if windows_c.empty:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    for (horizon, mode), grp in windows_c.groupby([\'horizon\', \'mode\']):\n'
    '        names = grp[\'name\'].unique()\n'
    '        if len(names) < 2:\n'
    '            continue\n'
    '        pivot = grp.pivot(index=\'window\', columns=\'name\', values=\'ic\')\n'
    '        n = len(names)\n'
    '        win_matrix = pd.DataFrame(np.nan, index=names, columns=names)\n'
    '        for a in names:\n'
    '            for b in names:\n'
    '                if a == b: continue\n'
    '                valid = pivot[[a, b]].dropna()\n'
    '                if len(valid):\n'
    '                    win_matrix.loc[a, b] = (valid[a] > valid[b]).mean()\n'
    '        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(4, n)))\n'
    '        sns.heatmap(win_matrix.astype(float), annot=True, fmt=\'.2f\',\n'
    '                    cmap=\'RdYlGn\', vmin=0, vmax=1, linewidths=0.5, ax=ax,\n'
    '                    cbar_kws={\'label\': \'win rate (row beats col on IC)\'})\n'
    '        ax.set_title(f\'IC Win rate — h={horizon}, mode={mode}\', fontsize=12)\n'
    '        plt.tight_layout(); plt.show()\n'
))

# ── 2.5 Metadata ──────────────────────────────────────────────────────────────
new_cells.append(md_cell('## 2.5  Model Metadata'))
new_cells.append(code_cell(
    'if not artifacts_cont:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    meta_rows_c = []\n'
    '    for stem, a in artifacts_cont.items():\n'
    '        row = {\n'
    '            \'artifact\':    stem,\n'
    '            \'model\':       a[\'model_key\'],\n'
    '            \'h\':           a[\'horizon\'],\n'
    '            \'mode\':        a[\'mode\'],\n'
    '            \'ft_type\':     a[\'ft_type\'],\n'
    '            \'train_start\': a[\'train_start\'],\n'
    '            \'train_end\':   a[\'train_end\'],\n'
    '            \'window_days\': a.get(\'window_days\', \'?\'),\n'
    '            \'n_features\':  len(a.get(\'features\', [])),\n'
    '            \'cv_windows\':  len(a[\'cv_metrics\']),\n'
    '        }\n'
    '        if \'n_rounds_final\' in a:\n'
    '            row[\'extra\'] = f\'n_rounds={a["n_rounds_final"]}\'\n'
    '        elif \'model_config\' in a:\n'
    '            cfg = a[\'model_config\']\n'
    '            parts = [f\'cell={cfg.get("cell","?")}\',\n'
    '                     f\'hidden={cfg.get("hidden_size","?")}\',\n'
    '                     f\'seq={a.get("seq_len","?")}\',]\n'
    '            if \'num_filters\' in cfg:\n'
    '                parts.append(f\'filters={cfg["num_filters"]}\')\n'
    '            row[\'extra\'] = \'  \'.join(parts)\n'
    '        else:\n'
    '            row[\'extra\'] = \'\'\n'
    '        meta_rows_c.append(row)\n'
    '\n'
    '    pd.DataFrame(meta_rows_c).set_index(\'artifact\').style \\\n'
    '        .set_caption(\'Continuous artifact metadata\') \\\n'
    '        .set_table_styles([{\'selector\': \'caption\', \'props\': \'font-size:13px; font-weight:bold;\'}])\n'
))

# ── 2.6 Sliding vs Expanding ──────────────────────────────────────────────────
new_cells.append(md_cell(
    '## 2.6  Sliding vs Expanding — direct comparison\n'
    'Only shown when the same continuous model was trained under both modes.\n'
))
new_cells.append(code_cell(
    'if summary_c.empty:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    s_c = summary_c[summary_c[\'mode\'] == \'sliding\']\n'
    '    e_c = summary_c[summary_c[\'mode\'] == \'expanding\']\n'
    '    common_c = set(s_c[\'model\']) & set(e_c[\'model\'])\n'
    '    if not common_c:\n'
    '        print("No continuous model has been trained under both modes yet.")\n'
    '    else:\n'
    '        cmp_rows = []\n'
    '        for model in sorted(common_c):\n'
    '            for m in [\'ic\', \'directional_accuracy\', \'mae\']:\n'
    '                s_val = s_c[s_c[\'model\'] == model][f\'{m}_mean\'].values[0]\n'
    '                e_val = e_c[e_c[\'model\'] == model][f\'{m}_mean\'].values[0]\n'
    '                cmp_rows.append({\'model\': model, \'metric\': m,\n'
    '                                  \'sliding\': s_val, \'expanding\': e_val,\n'
    '                                  \'delta (exp-sli)\': e_val - s_val})\n'
    '        pd.DataFrame(cmp_rows).style \\\n'
    '            .background_gradient(subset=[\'delta (exp-sli)\'], cmap=\'RdYlGn\', vmin=-0.02, vmax=0.02) \\\n'
    '            .format({\'sliding\': \'{:.4f}\', \'expanding\': \'{:.4f}\', \'delta (exp-sli)\': \'{:+.4f}\'}) \\\n'
    '            .set_caption(\'Sliding vs Expanding — continuous CV metrics\')\n'
))

# ── 2.7 Complexity vs IC ──────────────────────────────────────────────────────
new_cells.append(md_cell(
    '## 2.7  Is the Extra Complexity Worth It?  (Continuous)\n'
    'Reuses `model_complexity()` from Section 1. Primary metric: IC instead of balanced accuracy.\n'
))
new_cells.append(code_cell(
    'if not artifacts_cont:\n'
    '    print("No continuous artifacts.")\n'
    'else:\n'
    '    cplx_rows_c = []\n'
    '    for stem, a in artifacts_cont.items():\n'
    '        cplx_rows_c.append({\n'
    '            "name":       stem,\n'
    '            "model":      a["model_key"],\n'
    '            "horizon":    a["horizon"],\n'
    '            "complexity": model_complexity(a),\n'
    '            "ic_mean":    a["cv_summary"]["ic"]["mean"],\n'
    '            "ic_std":     a["cv_summary"]["ic"]["std"],\n'
    '            "da_mean":    a["cv_summary"]["directional_accuracy"]["mean"],\n'
    '        })\n'
    '    cplx_c = pd.DataFrame(cplx_rows_c).sort_values("complexity").reset_index(drop=True)\n'
    '    print(cplx_c[["name", "complexity", "ic_mean", "ic_std"]].to_string(index=False))\n'
    '\n'
    '    for horizon in sorted(cplx_c["horizon"].unique()):\n'
    '        df_h = cplx_c[cplx_c["horizon"] == horizon].copy()\n'
    '        if df_h.empty: continue\n'
    '        baseline_ic   = df_h.sort_values("complexity").iloc[0]["ic_mean"]\n'
    '        baseline_name = df_h.sort_values("complexity").iloc[0]["name"]\n'
    '        df_h["delta_ic"] = df_h["ic_mean"] - baseline_ic\n'
    '\n'
    '        fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n'
    '        colors = sns.color_palette("tab10", len(df_h))\n'
    '        ax = axes[0]\n'
    '        for i, (_, row) in enumerate(df_h.iterrows()):\n'
    '            ax.scatter(row["complexity"], row["ic_mean"], s=130,\n'
    '                       color=colors[i % 10], zorder=3)\n'
    '            ax.errorbar(row["complexity"], row["ic_mean"],\n'
    '                        yerr=row["ic_std"], fmt="none",\n'
    '                        color=colors[i % 10], lw=1.2, capsize=4)\n'
    '            ax.annotate(row["name"], (row["complexity"], row["ic_mean"]),\n'
    '                        xytext=(6, 4), textcoords="offset points", fontsize=8)\n'
    '        ax.axhline(0, color="red", ls="--", lw=0.8, label="IC=0")\n'
    '        ax.set_xscale("log")\n'
    '        ax.set_xlabel("Model complexity  (log scale)", fontsize=10)\n'
    '        ax.set_ylabel("IC (CV mean)", fontsize=10)\n'
    '        ax.set_title(f"Complexity vs IC  —  h={horizon}", fontsize=11)\n'
    '        ax.legend(fontsize=8)\n'
    '\n'
    '        ax2 = axes[1]\n'
    '        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_h["delta_ic"]]\n'
    '        bars = ax2.barh(df_h["name"], df_h["delta_ic"],\n'
    '                        color=bar_colors, edgecolor="white", height=0.55)\n'
    '        ax2.axvline(0, color="black", lw=0.8)\n'
    '        ax2.set_xlabel(f"IC gain vs {baseline_name}", fontsize=10)\n'
    '        ax2.set_title(f"Marginal IC gain  —  h={horizon}", fontsize=11)\n'
    '        for bar, val in zip(bars, df_h["delta_ic"]):\n'
    '            ax2.text(val + (0.001 if val >= 0 else -0.001),\n'
    '                     bar.get_y() + bar.get_height() / 2,\n'
    '                     f"{val:+.4f}", va="center",\n'
    '                     ha="left" if val >= 0 else "right", fontsize=8)\n'
    '        plt.tight_layout(); plt.show()\n'
))

# ── Append all new cells ──────────────────────────────────────────────────────
cells.extend(new_cells)
data["cells"] = cells
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Appended {len(new_cells)} cells — model_comparison Section 2 done.")
```

- [ ] **Step 2: Run the patch**

```bash
cd notebooks && python patch_model_comparison.py && cd ..
```

Expected: `Appended 18 cells — model_comparison Section 2 done.`

- [ ] **Step 3: Verify cell count and validate JSON**

```bash
python -c "
import json
data = json.load(open('notebooks/model_comparison.ipynb'))
cells = data['cells']
print(len(cells), 'cells — valid JSON')
# Spot-check Section 2 header is present
headers = [i for i, c in enumerate(cells) if 'Section 2' in ''.join(c.get('source', []))]
print('Section 2 cells at indices:', headers)
"
```

Expected: `44 cells — valid JSON` (26 + 18), Section 2 cells listed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/model_comparison.ipynb notebooks/patch_model_comparison.py
git commit -m "feat(notebook): add Section 2 continuous artifacts to model_comparison"
```

---

## Task 4: Patch confidence_analysis.ipynb — split + refactor Parts A/B

**Files:**
- Create: `notebooks/patch_confidence_analysis.py`
- Modify: `notebooks/confidence_analysis.ipynb`

Parts A and B must operate on `artifacts_discrete` only. The collect loop in Part B (the `for name, a in artifacts.items()` call) must become `artifacts_discrete.items()`.

- [ ] **Step 1: Write the patch script**

Create `notebooks/patch_confidence_analysis.py`:

```python
"""Patch confidence_analysis.ipynb: split at load + use artifacts_discrete in Parts A/B."""
import json, pathlib

NB = pathlib.Path("notebooks/confidence_analysis.ipynb")
data = json.loads(NB.read_text(encoding="utf-8"))
cells = data["cells"]

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def find_cell(cells, snippet):
    for i, c in enumerate(cells):
        if snippet in "".join(c.get("source", [])):
            return i
    raise ValueError(f"Not found: {snippet!r}")

# ── 1. Replace load cell (cell 2) — add target_type to print ─────────────────
idx_load = find_cell(cells, "Load new-format artifacts")
cells[idx_load]["source"] = (
    '# Load new-format artifacts\n'
    'artifacts = {}\n'
    'for f in sorted(ARTIFACTS_PATH.glob(\'*.pkl\')):\n'
    '    a = joblib.load(f)\n'
    '    if \'cv_summary\' not in a:\n'
    '        continue\n'
    '    artifacts[f.stem] = a\n'
    '    tt = a.get("target_type", "discrete")\n'
    '    print(f\'  {f.stem:22s}  h={a["horizon"]}  mode={a["mode"]}  \'\n'
    '          f\'target={tt}  windows={len(a["cv_metrics"])}\')\n'
    'print(f\'\\n{len(artifacts)} artifact(s) loaded.\')\n'
)

# ── 2. Insert split cell after load cell ──────────────────────────────────────
split_cell = code_cell(
    '# Split by target type\n'
    'artifacts_discrete = {k: v for k, v in artifacts.items()\n'
    '                      if v.get("target_type", "discrete") == "discrete"}\n'
    'artifacts_cont     = {k: v for k, v in artifacts.items()\n'
    '                      if v.get("target_type", "continuous") == "continuous"}\n'
    'print(f"Discrete: {len(artifacts_discrete)}  |  Continuous: {len(artifacts_cont)}")\n'
)
cells.insert(idx_load + 1, split_cell)

# ── 3. Part A window_rows cell — use artifacts_discrete ──────────────────────
idx_wrows = find_cell(cells, "window_rows = []")
cells[idx_wrows]["source"] = cells[idx_wrows]["source"].replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# ── 4. Part B collect loop — use artifacts_discrete ──────────────────────────
idx_collect = find_cell(cells, "pred_data = {}   # name -> DataFrame")
cells[idx_collect]["source"] = cells[idx_collect]["source"].replace(
    "for name, a in artifacts.items():",
    "for name, a in artifacts_discrete.items():"
)

# Save
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched confidence_analysis.ipynb — split + discrete refactor done.")
```

- [ ] **Step 2: Run the patch**

```bash
cd notebooks && python patch_confidence_analysis.py && cd ..
```

Expected: `Patched confidence_analysis.ipynb — split + discrete refactor done.`

- [ ] **Step 3: Verify**

```bash
python -c "
import json
data = json.load(open('notebooks/confidence_analysis.ipynb'))
cells = data['cells']
print(len(cells), 'cells — valid JSON')
# Confirm split cell exists
splits = [i for i, c in enumerate(cells) if 'artifacts_discrete' in ''.join(c.get('source', []))]
print('Cells referencing artifacts_discrete:', splits)
"
```

Expected: `20 cells — valid JSON` (19 + 1 split cell), multiple cells referencing artifacts_discrete.

- [ ] **Step 4: Commit**

```bash
git add notebooks/confidence_analysis.ipynb notebooks/patch_confidence_analysis.py
git commit -m "feat(notebook): split artifacts_discrete/cont in confidence_analysis"
```

---

## Task 5: Add Part C to confidence_analysis.ipynb

**Files:**
- Modify: `notebooks/patch_confidence_analysis.py` (rewrite to append Part C)
- Modify: `notebooks/confidence_analysis.ipynb`

Part C has four subsections: C1 (CV window IC vs decisiveness), C2 (inference on held-out data), C3 (|pred_return| confidence filter sweep), C4 (summary table).

- [ ] **Step 1: Rewrite patch script to append Part C**

Overwrite `notebooks/patch_confidence_analysis.py`:

```python
"""Append Part C (continuous confidence) to confidence_analysis.ipynb."""
import json, pathlib

NB = pathlib.Path("notebooks/confidence_analysis.ipynb")
data = json.loads(NB.read_text(encoding="utf-8"))
cells = data["cells"]

def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

new_cells = []

# ── Part C header ─────────────────────────────────────────────────────────────
new_cells.append(md_cell(
    '---\n'
    '## Part C — Continuous Model Confidence\n'
    'For regression artifacts: does prediction magnitude |pred_return| correlate with quality?\n'
    '- **C1 (no data needed):** CV window IC vs decisiveness (mean |pred_return|).\n'
    '- **C2–C4 (requires DB):** inference on held-out data, magnitude threshold sweep, summary.\n'
))

# ── C1: CV Window IC vs Decisiveness ─────────────────────────────────────────
new_cells.append(md_cell(
    '### C1  CV Window: IC vs Decisiveness\n'
    '`mean_abs_pred` = mean of |predicted log return| across the window — a proxy for how strongly\n'
    'the model commits to a direction. Does higher commitment correlate with higher IC?\n'
))
new_cells.append(code_cell(
    'wdf_c_rows = []\n'
    'for stem, a in artifacts_cont.items():\n'
    '    for wi, m in enumerate(a[\'cv_metrics\']):\n'
    '        wdf_c_rows.append({\n'
    '            \'name\':          stem,\n'
    '            \'model\':         a[\'model_key\'],\n'
    '            \'window\':        wi,\n'
    '            \'ic\':            m.get(\'ic\', np.nan),\n'
    '            \'directional_accuracy\': m.get(\'directional_accuracy\', np.nan),\n'
    '            \'mean_abs_pred\': m.get(\'mean_abs_pred\', np.nan),\n'
    '        })\n'
    '\n'
    'wdf_c = pd.DataFrame(wdf_c_rows)\n'
    '\n'
    'if wdf_c.empty:\n'
    '    print("No continuous artifacts — Part C skipped.")\n'
    'else:\n'
    '    print(wdf_c.head())\n'
))
new_cells.append(code_cell(
    'if not wdf_c.empty:\n'
    '    models_c = wdf_c[\'name\'].unique()\n'
    '    ncols = min(3, len(models_c))\n'
    '    nrows = max(1, (len(models_c) + ncols - 1) // ncols)\n'
    '    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)\n'
    '    palette = sns.color_palette(\'tab10\', len(models_c))\n'
    '\n'
    '    for ax, (name, color) in zip(axes.flat, zip(models_c, palette)):\n'
    '        sub = wdf_c[wdf_c[\'name\'] == name].dropna(subset=[\'mean_abs_pred\', \'ic\'])\n'
    '        if sub.empty:\n'
    '            ax.set_visible(False); continue\n'
    '        ax.scatter(sub[\'mean_abs_pred\'], sub[\'ic\'],\n'
    '                   color=color, alpha=0.75, s=60, edgecolors=\'white\', lw=0.5)\n'
    '        if len(sub) > 2:\n'
    '            m_, b_ = np.polyfit(sub[\'mean_abs_pred\'], sub[\'ic\'], 1)\n'
    '            x_ = np.linspace(sub[\'mean_abs_pred\'].min(), sub[\'mean_abs_pred\'].max(), 50)\n'
    '            ax.plot(x_, m_ * x_ + b_, color=\'black\', lw=1.2, ls=\'--\')\n'
    '            r, p = stats.pearsonr(sub[\'mean_abs_pred\'], sub[\'ic\'])\n'
    '            ax.set_title(f\'{name}\\nr={r:.2f}, p={p:.3f}\', fontsize=9)\n'
    '        else:\n'
    '            ax.set_title(name, fontsize=9)\n'
    '        ax.axhline(0, color=\'red\', lw=0.7, ls=\':\')\n'
    '        ax.set_xlabel(\'Decisiveness (mean |pred return|)\', fontsize=8)\n'
    '        ax.set_ylabel(\'IC (Spearman)\', fontsize=8)\n'
    '\n'
    '    for ax in axes.flat[len(models_c):]:\n'
    '        ax.set_visible(False)\n'
    '\n'
    '    plt.suptitle(\'Does higher prediction magnitude → better IC?\', fontsize=12, y=1.01)\n'
    '    plt.tight_layout(); plt.show()\n'
    '\n'
    '    # Correlation summary\n'
    '    corr_c_rows = []\n'
    '    for name, sub in wdf_c.groupby(\'name\'):\n'
    '        sub = sub.dropna(subset=[\'mean_abs_pred\', \'ic\'])\n'
    '        if len(sub) < 3: continue\n'
    '        r_ic,  p_ic  = stats.pearsonr(sub[\'mean_abs_pred\'], sub[\'ic\'])\n'
    '        r_da,  p_da  = stats.pearsonr(sub[\'mean_abs_pred\'], sub[\'directional_accuracy\'])\n'
    '        corr_c_rows.append({\n'
    '            \'artifact\': name,\n'
    '            \'model\': sub[\'model\'].iloc[0],\n'
    '            \'n_windows\': len(sub),\n'
    '            \'r(decisiveness, ic)\': f\'{r_ic:.3f} (p={p_ic:.3f})\',\n'
    '            \'r(decisiveness, dir_acc)\': f\'{r_da:.3f} (p={p_da:.3f})\',\n'
    '        })\n'
    '    if corr_c_rows:\n'
    '        display(pd.DataFrame(corr_c_rows).set_index(\'artifact\'))\n'
    '    else:\n'
    '        print("Not enough windows for correlation analysis.")\n'
))

# ── C2: Inference on held-out data (continuous) ───────────────────────────────
new_cells.append(md_cell(
    '---\n'
    '### C2  Inference on held-out data  (continuous)\n'
    'Loads each continuous final model and runs forward pass on dates after `train_end`.\n'
    'Returns `(date, ticker, actual_log_ret, pred_log_ret)`. No sigmoid — raw output is the predicted return.\n'
    '\n'
    '> **Requires SQLite DB access.**\n'
))
new_cells.append(code_cell(
    'def run_inference_continuous(artifact: dict, df_micro_raw, df_macro_raw) -> pd.DataFrame:\n'
    '    """Run continuous final model on dates after train_end.\n'
    '    Returns DataFrame with columns: date, ticker, actual, pred.\n'
    '    """\n'
    '    import torch\n'
    '\n'
    '    key       = artifact[\'model_key\']\n'
    '    horizon   = artifact[\'horizon\']\n'
    '    ft_type   = artifact[\'ft_type\']\n'
    '    train_end = pd.Timestamp(artifact[\'train_end\'])\n'
    '\n'
    '    df_macro_arg = df_macro_raw if ft_type == \'macro\' else None\n'
    '    # ml_ready returns y_discrete as 4th element; we need y_cont (5th)\n'
    '    df, X, y_disc, mask, y_cont = ml_ready(\n'
    '        horizon, df_micro_raw, df_macro=df_macro_arg, ft_type=ft_type\n'
    '    )\n'
    '\n'
    '    dates   = df.loc[mask, \'date\'].reset_index(drop=True)\n'
    '    tickers = df.loc[mask, \'ticker\'].reset_index(drop=True)\n'
    '    X = X.reset_index(drop=True)\n'
    '    y_cont = y_cont.reset_index(drop=True)\n'
    '\n'
    '    eval_mask = pd.to_datetime(dates) > train_end\n'
    '    if eval_mask.sum() == 0:\n'
    '        print(f\'  {key}: no held-out rows after {train_end.date()}\')\n'
    '        return pd.DataFrame()\n'
    '\n'
    '    if key in (\'xgb\',):\n'
    '        X_ev = X.loc[eval_mask]\n'
    '        preds = artifact[\'model\'].predict(X_ev)\n'
    '        return pd.DataFrame({\n'
    '            \'date\':   dates.loc[eval_mask].values,\n'
    '            \'ticker\': tickers.loc[eval_mask].values,\n'
    '            \'actual\': y_cont.loc[eval_mask].values,\n'
    '            \'pred\':   preds,\n'
    '        })\n'
    '\n'
    '    # --- neural (GRU, LSTM, CNN+GRU, CNN+LSTM) ---\n'
    '    from models.neural.lstm import add_cyclic_dow, build_sequences, SEQ_LEN\n'
    '    if \'dow\' in X.columns:\n'
    '        X = add_cyclic_dow(X.copy())\n'
    '\n'
    '    seqs, labs_disc, last_dates = build_sequences(X, y_disc, tickers, dates, SEQ_LEN)\n'
    '    # Rebuild labs using y_cont for the same sequence indices\n'
    '    _, labs_cont, _ = build_sequences(X, y_cont, tickers, dates, SEQ_LEN)\n'
    '\n'
    '    eval_date_set = set(dates.loc[eval_mask].values)\n'
    '    seq_mask = np.array([d in eval_date_set for d in last_dates])\n'
    '    seqs_ev  = seqs[seq_mask]\n'
    '    labs_ev  = labs_cont[seq_mask]\n'
    '    last_ev  = last_dates[seq_mask]\n'
    '\n'
    '    if len(seqs_ev) == 0:\n'
    '        print(f\'  {key}: no eval sequences\')\n'
    '        return pd.DataFrame()\n'
    '\n'
    '    sc = artifact[\'scaler\']\n'
    '    n, T, f = seqs_ev.shape\n'
    '    seqs_s = sc.transform(seqs_ev.reshape(-1, f)).reshape(n, T, f)\n'
    '\n'
    '    cfg = artifact[\'model_config\']\n'
    '    if \'num_filters\' in cfg:\n'
    '        from models.neural.cnn_rnn import StockCNNRNN\n'
    '        model = StockCNNRNN(**cfg)\n'
    '    else:\n'
    '        from models.neural.lstm import StockRNN\n'
    '        model = StockRNN(**cfg)\n'
    '    model.load_state_dict(artifact[\'model_state\'])\n'
    '    model.eval()\n'
    '\n'
    '    with torch.no_grad():\n'
    '        raw_out = model(torch.tensor(seqs_s, dtype=torch.float32))\n'
    '    # No sigmoid — raw output is predicted log return\n'
    '    preds = raw_out.numpy().flatten()\n'
    '\n'
    '    return pd.DataFrame({\n'
    '        \'date\':   last_ev,\n'
    '        \'actual\': labs_ev,\n'
    '        \'pred\':   preds,\n'
    '    })\n'
    '\n'
    '\n'
    '# --- collect ---\n'
    'if not artifacts_cont:\n'
    '    print("No continuous artifacts — skipping C2-C4.")\n'
    '    pred_data_cont = {}\n'
    'else:\n'
    '    from config import load_env\n'
    '    load_env()\n'
    '    # Reuse df_micro_raw / df_macro_raw if already loaded by Part B;\n'
    '    # otherwise load them here.\n'
    '    try:\n'
    '        df_micro_raw\n'
    '    except NameError:\n'
    '        df_micro_raw, df_macro_raw = _load_ohlcv()\n'
    '\n'
    '    pred_data_cont = {}\n'
    '    for name, a in artifacts_cont.items():\n'
    '        print(f\'Running continuous inference: {name} ...\')\n'
    '        df_pred = run_inference_continuous(a, df_micro_raw, df_macro_raw)\n'
    '        if len(df_pred) > 0:\n'
    '            pred_data_cont[name] = df_pred\n'
    '            print(f\'  -> {len(df_pred)} rows, \'\n'
    '                  f\'date range: {df_pred["date"].min()} -- {df_pred["date"].max()}\')\n'
    '\n'
    '    print(f\'\\nContinuous predictions collected for: {list(pred_data_cont.keys())}\')\n'
))

# ── C3: |pred_return| confidence filter sweep ────────────────────────────────
new_cells.append(md_cell(
    '### C3  |pred_return| as Confidence Filter\n'
    'Act only when |predicted return| ≥ δ. Sweep δ from 0 to the 90th percentile of |pred| in 20 steps.\n'
    'Shows coverage–directional_accuracy and coverage–IC curves.\n'
))
new_cells.append(code_cell(
    'from scipy.stats import spearmanr\n'
    '\n'
    'if not pred_data_cont:\n'
    '    print("No continuous prediction data.")\n'
    'else:\n'
    '    fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n'
    '\n'
    '    for name, df in pred_data_cont.items():\n'
    '        actuals  = df[\'actual\'].values\n'
    '        preds    = df[\'pred\'].values\n'
    '        abs_pred = np.abs(preds)\n'
    '\n'
    '        p90     = np.percentile(abs_pred, 90)\n'
    '        deltas  = np.linspace(0, p90, 20)\n'
    '        covs, dir_accs, ics = [], [], []\n'
    '\n'
    '        for delta in deltas:\n'
    '            keep = abs_pred >= delta\n'
    '            if keep.sum() < 10:\n'
    '                break\n'
    '            covs.append(keep.mean())\n'
    '            dir_accs.append(float(np.mean(np.sign(actuals[keep]) == np.sign(preds[keep]))))\n'
    '            ic_val, _ = spearmanr(actuals[keep], preds[keep])\n'
    '            ics.append(float(ic_val) if np.isfinite(ic_val) else 0.0)\n'
    '\n'
    '        axes[0].plot(covs, dir_accs, marker=\'o\', ms=4, lw=1.5, label=name)\n'
    '        axes[1].plot(covs, ics,      marker=\'o\', ms=4, lw=1.5, label=name)\n'
    '\n'
    '    axes[0].axhline(0.5, color=\'red\', ls=\'--\', lw=0.8, label=\'random\')\n'
    '    axes[0].set_title(\'Coverage–Directional Accuracy\', fontsize=11)\n'
    '    axes[0].set_xlabel(\'Coverage (fraction kept)\')\n'
    '    axes[0].set_ylabel(\'Directional accuracy\')\n'
    '    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))\n'
    '    axes[0].legend(fontsize=8)\n'
    '\n'
    '    axes[1].axhline(0, color=\'red\', ls=\'--\', lw=0.8, label=\'IC=0\')\n'
    '    axes[1].set_title(\'Coverage–IC (Spearman)\', fontsize=11)\n'
    '    axes[1].set_xlabel(\'Coverage (fraction kept)\')\n'
    '    axes[1].set_ylabel(\'Spearman IC\')\n'
    '    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))\n'
    '    axes[1].legend(fontsize=8)\n'
    '\n'
    '    plt.suptitle(\'Does filtering on |pred_return| improve quality?\', fontsize=12, y=1.01)\n'
    '    plt.tight_layout(); plt.show()\n'
))

# ── C4: Summary metrics table ─────────────────────────────────────────────────
new_cells.append(md_cell('### C4  Summary — Continuous Inference Metrics'))
new_cells.append(code_cell(
    'from scipy.stats import spearmanr\n'
    'from sklearn.metrics import mean_absolute_error\n'
    '\n'
    'if not pred_data_cont:\n'
    '    print("No continuous prediction data.")\n'
    'else:\n'
    '    sum_rows_c = []\n'
    '    for name, df in pred_data_cont.items():\n'
    '        actuals = df[\'actual\'].values\n'
    '        preds   = df[\'pred\'].values\n'
    '        ic_val, _ = spearmanr(actuals, preds)\n'
    '        sum_rows_c.append({\n'
    '            \'artifact\':            name,\n'
    '            \'n_preds\':             len(df),\n'
    '            \'MAE\':                 f\'{mean_absolute_error(actuals, preds):.6f}\',\n'
    '            \'RMSE\':                f\'{np.sqrt(np.mean((actuals - preds)**2)):.6f}\',\n'
    '            \'IC (Spearman)\':       f\'{float(ic_val):.4f}\' if np.isfinite(ic_val) else \'nan\',\n'
    '            \'Directional Acc\':     f\'{np.mean(np.sign(actuals) == np.sign(preds)):.4f}\',\n'
    '        })\n'
    '\n'
    '    pd.DataFrame(sum_rows_c).set_index(\'artifact\').style \\\n'
    '        .set_caption(\'Continuous inference metrics on held-out data\') \\\n'
    '        .set_table_styles([{\'selector\': \'caption\',\n'
    '                            \'props\': \'font-size:13px; font-weight:bold;\'}])\n'
))

# ── Append all new cells ──────────────────────────────────────────────────────
cells.extend(new_cells)
data["cells"] = cells
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Appended {len(new_cells)} cells — confidence_analysis Part C done.")
```

- [ ] **Step 2: Run the patch**

```bash
cd notebooks && python patch_confidence_analysis.py && cd ..
```

Expected: `Appended 11 cells — confidence_analysis Part C done.`

- [ ] **Step 3: Verify**

```bash
python -c "
import json
data = json.load(open('notebooks/confidence_analysis.ipynb'))
cells = data['cells']
print(len(cells), 'cells — valid JSON')
part_c = [i for i, c in enumerate(cells) if 'Part C' in ''.join(c.get('source', []))]
print('Part C cells at:', part_c)
"
```

Expected: `31 cells — valid JSON` (20 + 11), Part C cells listed.

- [ ] **Step 4: Commit**

```bash
git add notebooks/confidence_analysis.ipynb notebooks/patch_confidence_analysis.py
git commit -m "feat(notebook): add Part C continuous confidence analysis to confidence_analysis"
```

---

## Task 6: Cleanup and final push

**Files:**
- Delete: `notebooks/patch_model_comparison.py`
- Delete: `notebooks/patch_confidence_analysis.py`

- [ ] **Step 1: Remove scratch patch scripts**

```bash
rm notebooks/patch_model_comparison.py notebooks/patch_confidence_analysis.py
```

- [ ] **Step 2: Final JSON sanity check on both notebooks**

```bash
python -c "
import json
for nb in ['notebooks/model_comparison.ipynb', 'notebooks/confidence_analysis.ipynb']:
    data = json.load(open(nb))
    print(f'{nb}: {len(data[\"cells\"])} cells — OK')
"
```

Expected:
```
notebooks/model_comparison.ipynb: 44 cells — OK
notebooks/confidence_analysis.ipynb: 31 cells — OK
```

- [ ] **Step 3: Commit cleanup and push**

```bash
git add -A
git commit -m "chore: remove notebook patch scripts after application"
git push -u origin notebook-continuous-support
```

---

## Self-review

**Spec coverage:**
- ✅ Artifact split at load (`artifacts_discrete` / `artifacts_cont`) — Tasks 2, 4
- ✅ `model_comparison.ipynb` Section 1 operates on `artifacts_discrete` — Task 2
- ✅ `model_comparison.ipynb` Section 2: CV Summary, Boxplots, Temporal Trend, Head-to-Head, Metadata, Sliding/Expanding, Complexity — Task 3
- ✅ `confidence_analysis.ipynb` Parts A/B use `artifacts_discrete` — Task 4
- ✅ `confidence_analysis.ipynb` Part C: C1 IC vs decisiveness, C2 inference, C3 filter sweep, C4 summary — Task 5
- ✅ `evaluate.py` `mean_abs_pred` verified — Task 1
- ✅ `model_performance.ipynb` untouched — not in any task (correct)

**Placeholder scan:** No TBDs, no "similar to Task N", all code blocks complete.

**Type consistency:**
- `artifacts_discrete` / `artifacts_cont` defined in Task 2 (model_comparison) and Task 4 (confidence_analysis) independently in each notebook — correct, they are per-notebook locals.
- `model_complexity()` reused in Task 3 Section 2.7 — defined earlier in the same notebook by existing cell, so it's in scope.
- `_load_ohlcv()` referenced in C2 — defined in Part B's existing cell 8, in scope.
- `ml_ready` in `run_inference_continuous` returns 5 values `(df, X, y_disc, mask, y_cont)` — matches the signature in `models/trees/features.py::ml_ready()`.
- `build_sequences` called twice to get `labs_disc` and `labs_cont` — correct since the function takes `y` as a parameter.
