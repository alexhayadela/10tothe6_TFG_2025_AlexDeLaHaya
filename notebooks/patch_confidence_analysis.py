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

# ── 1. Replace load cell — add target_type to print ───────────────────────────
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
# source may be a list — join, replace, store as string
src = "".join(cells[idx_wrows].get("source", []))
cells[idx_wrows]["source"] = src.replace(
    "for stem, a in artifacts.items():",
    "for stem, a in artifacts_discrete.items():"
)

# ── 4. Part B collect loop — use artifacts_discrete ──────────────────────────
idx_collect = find_cell(cells, "pred_data = {}   # name -> DataFrame")
src = "".join(cells[idx_collect].get("source", []))
cells[idx_collect]["source"] = src.replace(
    "for name, a in artifacts.items():",
    "for name, a in artifacts_discrete.items():"
)

# Save
NB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched confidence_analysis.ipynb — split + discrete refactor done.")
