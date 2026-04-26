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
