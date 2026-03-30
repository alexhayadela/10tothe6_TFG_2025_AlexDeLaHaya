"""Append complexity section to model_comparison.ipynb."""
import json, pathlib

nb_path = pathlib.Path("notebooks/model_comparison.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

COMPLEXITY_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10  Is the Extra Complexity Worth It?\n",
            "Model complexity is defined as trainable parameter count (neural) or effective "
            "node count (tree ensembles). We ask: does each step up in complexity deliver "
            "a proportional gain in balanced accuracy?"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """\
def model_complexity(a: dict) -> int:
    \"\"\"Rough complexity score comparable across model families.

    Trees  : n_trees x nodes_per_tree  (effective leaf count)
    Neural : trainable parameter count
    \"\"\"
    key = a["model_key"]
    p   = a.get("params", {})
    if key == "rf":
        return p.get("n_estimators", 500) * (2 ** p.get("max_depth", 5) - 1)
    if key == "xgb":
        n_rounds = a.get("n_rounds_final", p.get("n_estimators", 150))
        return n_rounds * (2 ** p.get("max_depth", 3) - 1)
    # neural
    cfg  = a["model_config"]
    h, inp = cfg["hidden_size"], cfg["input_size"]
    mult = 4 if cfg.get("cell", "gru") == "lstm" else 3
    if "num_filters" in cfg:
        f, k = cfg["num_filters"], cfg["kernel_size"]
        conv = inp * f * k + f + 2 * f
        rnn  = mult * (f * h + h * h + h)
    else:
        conv = 0
        rnn  = mult * (inp * h + h * h + h)
    return conv + rnn + h + 1

cplx_rows = []
for stem, a in artifacts.items():
    cplx_rows.append({
        "name":       stem,
        "model":      a["model_key"],
        "horizon":    a["horizon"],
        "complexity": model_complexity(a),
        "ba_mean":    a["cv_summary"]["balanced_accuracy"]["mean"],
        "ba_std":     a["cv_summary"]["balanced_accuracy"]["std"],
        "auc_mean":   a["cv_summary"]["roc_auc"]["mean"],
    })
cplx_df = pd.DataFrame(cplx_rows).sort_values("complexity").reset_index(drop=True)
print(cplx_df[["name", "complexity", "ba_mean", "ba_std"]].to_string(index=False))
"""
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """\
for horizon in sorted(cplx_df["horizon"].unique()):
    df_h = cplx_df[cplx_df["horizon"] == horizon].copy()
    if df_h.empty:
        continue

    baseline_ba   = df_h.sort_values("complexity").iloc[0]["ba_mean"]
    baseline_name = df_h.sort_values("complexity").iloc[0]["name"]
    df_h["delta_ba"] = df_h["ba_mean"] - baseline_ba

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: scatter complexity vs accuracy ---
    ax = axes[0]
    colors = sns.color_palette("tab10", len(df_h))
    for i, (_, row) in enumerate(df_h.iterrows()):
        ax.scatter(row["complexity"], row["ba_mean"], s=130,
                   color=colors[i % 10], zorder=3)
        ax.errorbar(row["complexity"], row["ba_mean"],
                    yerr=row["ba_std"], fmt="none",
                    color=colors[i % 10], lw=1.2, capsize=4)
        ax.annotate(row["name"], (row["complexity"], row["ba_mean"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.5, color="red", ls="--", lw=0.8, label="random")
    ax.set_xscale("log")
    ax.set_xlabel("Model complexity  (log scale)", fontsize=10)
    ax.set_ylabel("Balanced accuracy (CV mean)", fontsize=10)
    ax.set_title(f"Complexity vs Performance  —  h={horizon}", fontsize=11)
    ax.legend(fontsize=8)

    # --- right: marginal gain bars ---
    ax2 = axes[1]
    df_s = df_h.sort_values("complexity")
    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_s["delta_ba"]]
    bars = ax2.barh(df_s["name"], df_s["delta_ba"] * 100,
                    color=bar_colors, edgecolor="white", height=0.55)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel(f"Balanced accuracy gain vs {baseline_name} (pp)", fontsize=10)
    ax2.set_title(f"Marginal gain over simplest model  —  h={horizon}", fontsize=11)
    for bar, val in zip(bars, df_s["delta_ba"] * 100):
        ax2.text(val + (0.05 if val >= 0 else -0.05),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.2f}pp", va="center",
                 ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    plt.show()
"""
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": """\
# Efficiency table: balanced-accuracy gain per unit of log-complexity increase
eff_rows = []
for horizon in sorted(cplx_df["horizon"].unique()):
    df_h = cplx_df[cplx_df["horizon"] == horizon].sort_values("complexity")
    base = df_h.iloc[0]
    for _, row in df_h.iterrows():
        ratio    = max(row["complexity"], 1) / max(base["complexity"], 1)
        log_r    = np.log10(ratio)
        delta_pp = (row["ba_mean"] - base["ba_mean"]) * 100
        eff      = delta_pp / log_r if log_r > 0.01 else float("nan")
        verdict  = (
            "baseline" if row["name"] == base["name"]
            else "YES"      if delta_pp >= 0.5
            else "MARGINAL" if delta_pp >= 0.1
            else "NO"
        )
        eff_rows.append({
            "artifact":        row["name"],
            "h":               horizon,
            "complexity":      f"{row['complexity']:,}",
            "ba_mean":         f"{row['ba_mean']:.4f}",
            "delta_ba (pp)":   f"{delta_pp:+.2f}",
            "pp / log10(C/C0)": f"{eff:.2f}" if not np.isnan(eff) else "—",
            "worth it?":       verdict,
        })

eff_df = pd.DataFrame(eff_rows)

def _verdict_color(val):
    return {
        "YES":      "background-color:#c8f7c5",
        "MARGINAL": "background-color:#fef9c3",
        "NO":       "background-color:#ffd5d5",
    }.get(val, "")

eff_df.style \\
    .applymap(_verdict_color, subset=["worth it?"]) \\
    .set_caption("Complexity efficiency — is the extra complexity justified?") \\
    .set_table_styles([{"selector": "caption",
                        "props": "font-size:13px; font-weight:bold;"}])
"""
    }
]

nb["cells"].extend(COMPLEXITY_CELLS)
nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Done — total cells: {len(nb['cells'])}")
