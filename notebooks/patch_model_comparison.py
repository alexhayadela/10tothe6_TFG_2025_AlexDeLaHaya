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

# ── 4. Fix tree_arts to use artifacts_discrete ────────────────────────────────
idx_tree = find_cell(cells, "tree_arts = {k: v for k, v in artifacts.items()")
src = "".join(cells[idx_tree]["source"])
cells[idx_tree]["source"] = src.replace(
    "tree_arts = {k: v for k, v in artifacts.items()",
    "tree_arts = {k: v for k, v in artifacts_discrete.items()"
)

# ── 5. Fix meta_rows to use artifacts_discrete ────────────────────────────────
idx_meta = find_cell(cells, "meta_rows = []")
src = "".join(cells[idx_meta]["source"])
cells[idx_meta]["source"] = src.replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# ── 6. Fix model_complexity cell — loop over artifacts_discrete ───────────────
# The model_complexity function and the cplx_rows loop are in the SAME cell
idx_cplx1 = find_cell(cells, "def model_complexity(")
src = "".join(cells[idx_cplx1]["source"])
cells[idx_cplx1]["source"] = src.replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# Save
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched model_comparison.ipynb — split + discrete refactor done.")
