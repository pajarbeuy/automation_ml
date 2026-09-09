from __future__ import annotations

"""
HTML Report Generator.

Produces a self-contained HTML report (single file, no external deps) that
summarises the entire AutoML workflow for a project:
- Dataset profile summary
- Preprocessing decisions made by the user
- Experiment results + comparison table
- Confusion matrix (if available)
- Selected candidate model metrics
"""

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_float(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _status_badge(status: str) -> str:
    color = {"completed": "#10b981", "failed": "#ef4444", "running": "#f59e0b"}.get(status, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75rem;">{status}</span>'


def _build_decisions_table(decisions: list[dict]) -> str:
    if not decisions:
        return "<p style='color:#6b7280'>No decisions recorded.</p>"
    rows = ""
    for d in decisions:
        status_color = {"accepted": "#10b981", "overridden": "#f59e0b", "rejected": "#ef4444"}.get(
            d.get("status", ""), "#6b7280"
        )
        rows += f"""
        <tr>
          <td style="font-family:monospace">{d.get('column_name', '—')}</td>
          <td>{d.get('category', '—')}</td>
          <td style="font-weight:600">{d.get('selected_action', '—')}</td>
          <td><span style="color:{status_color};font-weight:600">{d.get('status','—')}</span></td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <thead>
        <tr style="background:#1e1e2e;color:#a6adc8">
          <th style="text-align:left;padding:8px">Column</th>
          <th style="text-align:left;padding:8px">Category</th>
          <th style="text-align:left;padding:8px">Action</th>
          <th style="text-align:left;padding:8px">Status</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _build_experiments_table(experiments: list[dict], is_regression: bool = False) -> str:
    if not experiments:
        return "<p style='color:#6b7280'>No experiments recorded.</p>"

    def _get_metric(exp: dict, name: str) -> float | None:
        for m in exp.get("metrics", []):
            if m.get("name") == name and m.get("split") == "cv":
                return m.get("value")
        return None

    rows = ""
    if is_regression:
        for exp in experiments:
            r2 = _get_metric(exp, "r2_mean")
            rmse = _get_metric(exp, "rmse_mean")
            mae = _get_metric(exp, "mae_mean")
            t = exp.get("training_time_seconds")
            rows += f"""
            <tr>
              <td>{_status_badge(exp.get('status',''))}</td>
              <td style="font-weight:600">{exp.get('model_name','')}</td>
              <td style="font-family:monospace">{_fmt_float(r2)}</td>
              <td style="font-family:monospace">{_fmt_float(rmse)}</td>
              <td style="font-family:monospace">{_fmt_float(mae)}</td>
              <td style="color:#6b7280">{f"{t:.1f}s" if t else "—"}</td>
            </tr>"""

        return f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
          <thead>
            <tr style="background:#1e1e2e;color:#a6adc8">
              <th style="text-align:left;padding:8px">Status</th>
              <th style="text-align:left;padding:8px">Model</th>
              <th style="text-align:left;padding:8px">R² (CV)</th>
              <th style="text-align:left;padding:8px">RMSE</th>
              <th style="text-align:left;padding:8px">MAE</th>
              <th style="text-align:left;padding:8px">Train Time</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""
    else:
        for exp in experiments:
            f1 = _get_metric(exp, "f1_mean")
            acc = _get_metric(exp, "accuracy_mean")
            auc = _get_metric(exp, "roc_auc_mean")
            t = exp.get("training_time_seconds")
            rows += f"""
            <tr>
              <td>{_status_badge(exp.get('status',''))}</td>
              <td style="font-weight:600">{exp.get('model_name','')}</td>
              <td style="font-family:monospace">{_fmt_pct(f1)}</td>
              <td style="font-family:monospace">{_fmt_pct(acc)}</td>
              <td style="font-family:monospace">{_fmt_pct(auc)}</td>
              <td style="color:#6b7280">{f"{t:.1f}s" if t else "—"}</td>
            </tr>"""

        return f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
          <thead>
            <tr style="background:#1e1e2e;color:#a6adc8">
              <th style="text-align:left;padding:8px">Status</th>
              <th style="text-align:left;padding:8px">Model</th>
              <th style="text-align:left;padding:8px">F1 (CV)</th>
              <th style="text-align:left;padding:8px">Accuracy</th>
              <th style="text-align:left;padding:8px">ROC-AUC</th>
              <th style="text-align:left;padding:8px">Train Time</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""


def _build_confusion_matrix(cm: list[list[int]] | None) -> str:
    if not cm:
        return "<p style='color:#6b7280'>Confusion matrix not available.</p>"
    n = len(cm)
    max_val = max((cm[i][j] for i in range(n) for j in range(n)), default=1)
    header = "<th></th>" + "".join(f"<th style='padding:8px;text-align:center'>Pred {i}</th>" for i in range(n))
    rows = ""
    for i, row in enumerate(cm):
        cells = f"<td style='font-weight:600;padding:8px;color:#a6adc8'>Act {i}</td>"
        for j, val in enumerate(row):
            intensity = val / max_val if max_val > 0 else 0
            if i == j:
                bg = f"rgba(99,102,241,{0.2 + intensity * 0.7:.2f})"
                color = "#fff" if intensity > 0.4 else "#cdd6f4"
            else:
                bg = f"rgba(239,68,68,{0.05 + intensity * 0.5:.2f})"
                color = "#fff" if intensity > 0.5 else "#cdd6f4"
            cells += f"<td style='padding:12px;text-align:center;background:{bg};color:{color};font-weight:700;border-radius:4px'>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    return f"""
    <table style="border-collapse:separate;border-spacing:3px;margin:0 auto">
      <thead><tr style="color:#a6adc8;font-size:0.8rem">{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <p style="font-size:0.75rem;color:#6b7280;margin-top:0.5rem">
      <span style="display:inline-block;width:10px;height:10px;background:rgba(99,102,241,0.6);border-radius:2px;margin-right:4px"></span>Correct (diagonal)
      &nbsp;
      <span style="display:inline-block;width:10px;height:10px;background:rgba(239,68,68,0.3);border-radius:2px;margin-right:4px"></span>Misclassification
    </p>"""


def _build_feature_table(profile: dict) -> str:
    features = profile.get("features", [])
    if not features:
        return "<p style='color:#6b7280'>No features.</p>"
    rows = ""
    for f in features:
        miss = f.get("missing_count", 0)
        miss_str = f"{miss} ({f.get('missing_rate', 0)*100:.1f}%)" if miss else "None"
        miss_color = "#f59e0b" if miss else "#10b981"
        rows += f"""
        <tr>
          <td style="font-family:monospace;font-weight:500">{f.get('name','')}</td>
          <td><span style="background:#313244;padding:2px 6px;border-radius:8px;font-size:0.75rem">{f.get('dtype','')}</span></td>
          <td style="color:{miss_color}">{miss_str}</td>
          <td>{f.get('unique_count','')}</td>
          <td>{'⚠ Yes' if f.get('constant') else '—'}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <thead>
        <tr style="background:#1e1e2e;color:#a6adc8">
          <th style="text-align:left;padding:8px">Column</th>
          <th style="text-align:left;padding:8px">Type</th>
          <th style="text-align:left;padding:8px">Missing</th>
          <th style="text-align:left;padding:8px">Unique</th>
          <th style="text-align:left;padding:8px">Constant</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_html_report(
    project_name: str,
    project_id: str,
    target_column: str,
    dataset_filename: str,
    dataset_rows: int,
    dataset_columns: int,
    profile: dict,
    decisions: list[dict],
    experiments: list[dict],
    candidate_experiment: dict | None,
    confusion_matrix: list[list[int]] | None = None,
    generated_at: str = "",
    task_type: str | None = None,
    residuals_summary: dict[str, float] | None = None,
) -> str:
    """
    Generate a self-contained HTML report for the project.
    Returns the full HTML as a string.
    """

    def _metric(exp: dict, name: str) -> float | None:
        for m in exp.get("metrics", []):
            if m.get("name") == name and m.get("split") == "cv":
                return m.get("value")
        return None

    if task_type is None:
        has_reg_metric = any(
            any(m.get("name") in ("r2_mean", "rmse_mean") for m in exp.get("metrics", []))
            for exp in experiments
        )
        if not has_reg_metric and candidate_experiment:
            has_reg_metric = any(
                m.get("name") in ("r2_mean", "rmse_mean") for m in candidate_experiment.get("metrics", [])
            )
        is_regression = has_reg_metric
    else:
        is_regression = (task_type == "regression")

    candidate_section = ""
    if candidate_experiment:
        t = candidate_experiment.get("training_time_seconds")
        if is_regression:
            r2 = _metric(candidate_experiment, "r2_mean")
            rmse = _metric(candidate_experiment, "rmse_mean")
            mae = _metric(candidate_experiment, "mae_mean")
            mse = _metric(candidate_experiment, "mse_mean")

            if residuals_summary:
                res_rows = "".join(
                    f"<tr><td style='padding:6px 12px;font-weight:600;border-bottom:1px solid #313244'>{k}</td>"
                    f"<td style='padding:6px 12px;font-family:monospace;border-bottom:1px solid #313244'>{v:.4f}</td></tr>"
                    for k, v in residuals_summary.items()
                )
                diag_html = f"""
                <h3>Residuals & Error Summary</h3>
                <table style="border-collapse:collapse;margin:0.5rem 0;font-size:0.85rem">
                  <tbody>{res_rows}</tbody>
                </table>"""
            else:
                diag_html = """
                <h3>Regression Evaluation Summary</h3>
                <p style='color:#a6adc8;font-size:0.85rem'>
                  Continuous target evaluated using cross-validated R², Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
                </p>"""

            candidate_section = f"""
            <section class="section">
              <h2>🏆 Selected Candidate Model</h2>
              <div style="background:#1e1b4b;border:1px solid rgba(99,102,241,0.4);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem">
                <div style="font-size:1.4rem;font-weight:700;margin-bottom:0.5rem">{candidate_experiment.get('model_name','')}</div>
                <div style="color:#a6adc8;font-size:0.8rem">Training time: {f"{t:.1f}s" if t else "—"} &middot; Seed: {candidate_experiment.get('random_seed','')}</div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem;margin-bottom:1.5rem">
                <div class="metric-card"><div class="metric-label">R² Score (CV)</div><div class="metric-val">{_fmt_float(r2)}</div></div>
                <div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-val">{_fmt_float(rmse)}</div></div>
                <div class="metric-card"><div class="metric-label">MAE</div><div class="metric-val">{_fmt_float(mae)}</div></div>
                {"" if mse is None else f'<div class="metric-card"><div class="metric-label">MSE</div><div class="metric-val">{_fmt_float(mse)}</div></div>'}
              </div>
              {diag_html}
            </section>"""
        else:
            f1 = _metric(candidate_experiment, "f1_mean")
            acc = _metric(candidate_experiment, "accuracy_mean")
            auc = _metric(candidate_experiment, "roc_auc_mean")
            prec = _metric(candidate_experiment, "precision_mean")
            rec = _metric(candidate_experiment, "recall_mean")

            cm_html = _build_confusion_matrix(confusion_matrix)
            candidate_section = f"""
            <section class="section">
              <h2>🏆 Selected Candidate Model</h2>
              <div style="background:#1e1b4b;border:1px solid rgba(99,102,241,0.4);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem">
                <div style="font-size:1.4rem;font-weight:700;margin-bottom:0.5rem">{candidate_experiment.get('model_name','')}</div>
                <div style="color:#a6adc8;font-size:0.8rem">Training time: {f"{t:.1f}s" if t else "—"} &middot; Seed: {candidate_experiment.get('random_seed','')}</div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem;margin-bottom:1.5rem">
                <div class="metric-card"><div class="metric-label">F1 Score (CV)</div><div class="metric-val">{_fmt_pct(f1)}</div></div>
                <div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-val">{_fmt_pct(acc)}</div></div>
                <div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val">{_fmt_pct(prec)}</div></div>
                <div class="metric-card"><div class="metric-label">Recall</div><div class="metric-val">{_fmt_pct(rec)}</div></div>
                {"" if auc is None else f'<div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-val">{_fmt_pct(auc)}</div></div>'}
              </div>
              <h3>Confusion Matrix</h3>
              <div style="overflow-x:auto">{cm_html}</div>
            </section>"""

    profile_section = f"""
    <section class="section">
      <h2>📊 Dataset Profile</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1rem;margin-bottom:1.5rem">
        <div class="metric-card"><div class="metric-label">Rows</div><div class="metric-val">{dataset_rows:,}</div></div>
        <div class="metric-card"><div class="metric-label">Columns</div><div class="metric-val">{dataset_columns}</div></div>
        <div class="metric-card"><div class="metric-label">Target</div><div class="metric-val" style="font-size:1rem">{target_column}</div></div>
        <div class="metric-card"><div class="metric-label">Duplicate Rows</div><div class="metric-val">{profile.get('duplicate_rows', 0)}</div></div>
      </div>
      {_build_feature_table(profile)}
    </section>"""

    decisions_section = f"""
    <section class="section">
      <h2>🔧 Preprocessing Decisions</h2>
      {_build_decisions_table(decisions)}
    </section>"""

    experiments_section = f"""
    <section class="section">
      <h2>🧪 Experiment Results</h2>
      {_build_experiments_table(experiments, is_regression=is_regression)}
    </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AutoML Report — {project_name}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f17;color:#cdd6f4;line-height:1.6;padding:2rem}}
    .container{{max-width:960px;margin:0 auto}}
    header{{border-bottom:1px solid #313244;padding-bottom:1.5rem;margin-bottom:2rem}}
    header h1{{font-size:1.8rem;font-weight:700;color:#cba6f7;margin-bottom:0.25rem}}
    header .meta{{color:#6b7280;font-size:0.85rem}}
    .section{{background:#181825;border:1px solid #313244;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
    .section h2{{font-size:1.15rem;font-weight:700;margin-bottom:1rem;color:#cdd6f4}}
    .section h3{{font-size:0.95rem;font-weight:600;margin:1rem 0 0.5rem;color:#a6adc8}}
    table td,table th{{padding:8px;border-bottom:1px solid #313244;vertical-align:middle}}
    table tr:last-child td{{border-bottom:none}}
    .metric-card{{background:#1e1e2e;border:1px solid #313244;border-radius:10px;padding:1rem;text-align:center}}
    .metric-label{{font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.35rem}}
    .metric-val{{font-size:1.6rem;font-weight:700;color:#cba6f7}}
    footer{{text-align:center;color:#4a4f69;font-size:0.75rem;margin-top:2rem}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>AutoML HIL — Project Report</h1>
      <div class="meta">
        Project: <strong>{project_name}</strong> &middot;
        File: <strong>{dataset_filename}</strong> &middot;
        Generated: {generated_at}
      </div>
    </header>

    {candidate_section}
    {profile_section}
    {decisions_section}
    {experiments_section}

    <footer>
      Generated by AutoML Human-in-the-Loop &middot; Project ID: {project_id}
    </footer>
  </div>
</body>
</html>"""

    return html
