"""Etapa 3: Dashboard de observabilidade (HTML estático, autocontido, offline).

Consolida em uma única página:
    • saúde do pipeline (status das tarefas, latência, volume, CPU/memória)
    • data contracts (lotes aprovados × bloqueados)
    • drift de dados (PSI por feature × mês, share de features com drift, drift do score)
    • performance do modelo ao longo do tempo (AUC, KS, taxa de default)
    • fairness (disparate impact) e alertas
    • análise causal (root cause) do último lote

Fonte de dados: logs/*.jsonl + reports/*.csv/json  (gerados pelo pipeline).
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from credit_monitoring.config import load_settings
from credit_monitoring.monitoring.telemetry import log, read_jsonl

PALETTE = {"ok": "#1a7f37", "warn": "#bf8700", "crit": "#cf222e", "blue": "#0969da",
           "gray": "#6e7781", "purple": "#8250df"}
STATUS_COLOR = {"success": PALETTE["ok"], "blocked": PALETTE["crit"], "failed": "#82071e",
                "healthy": PALETTE["ok"], "attention": PALETTE["warn"], "degraded": PALETTE["crit"]}
LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), height=360,
              font=dict(family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
              legend=dict(orientation="h", y=-0.2))


def _fig_html(fig: go.Figure, batch_axis: bool = True) -> str:
    if batch_axis:  # nomes de lote ("2026-01") não devem virar datas no eixo
        fig.update_xaxes(type="category")
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})


def _card(title: str, body: str, wide: bool = False, note: str = "") -> str:
    n = f"<p class='note'>{note}</p>" if note else ""
    return f"<section class='card{' wide' if wide else ''}'><h3>{title}</h3>{n}{body}</section>"


def _kpi(label: str, value: str, tone: str = "") -> str:
    return f"<div class='kpi {tone}'><span>{label}</span><b>{value}</b></div>"


def build_dashboard() -> str:
    s = load_settings()
    reports = s.path("reports_dir")
    stages = pd.DataFrame(read_jsonl("stages.jsonl"))
    alerts = pd.DataFrame(read_jsonl("alerts.jsonl"))
    hist = pd.read_csv(reports / "monitoring_history.csv")
    psi_matrix = pd.read_csv(reports / "drift" / "psi_matrix.csv", index_col=0)
    summary = json.loads((reports / "monitoring_summary.json").read_text(encoding="utf-8"))
    meta = json.loads((s.path("model_dir") / "model_metadata.json").read_text(encoding="utf-8"))
    ref = summary["reference_metrics"]
    mcfg = s.section("monitoring")
    scored = hist[hist["status"] != "blocked"].copy()

    # ------------------------------------------------------------------ KPIs
    n_runs = len(stages)
    ok_rate = (stages["status"] == "success").mean() if n_runs else 0
    n_crit = int((alerts["severity"] == "critical").sum()) if not alerts.empty else 0
    n_warn = int((alerts["severity"] == "warning").sum()) if not alerts.empty else 0
    last = scored.iloc[-1]
    auc_drop = ref["auc"] - last.get("perf_auc", ref["auc"])
    kpis = "".join([
        _kpi("Execuções de tarefas", f"{n_runs}"),
        _kpi("Taxa de sucesso", f"{ok_rate:.0%}", "ok" if ok_rate > .95 else "warn"),
        _kpi("Lotes bloqueados (contrato)", f"{int((hist['status'] == 'blocked').sum())}",
             "crit" if (hist["status"] == "blocked").any() else "ok"),
        _kpi("Alertas críticos", f"{n_crit}", "crit" if n_crit else "ok"),
        _kpi("Alertas de atenção", f"{n_warn}", "warn" if n_warn else "ok"),
        _kpi(f"AUC atual ({last['batch']})", f"{last.get('perf_auc', float('nan')):.3f}",
             "crit" if auc_drop >= mcfg["auc_drop_critical"] else "ok"),
        _kpi("AUC referência", f"{ref['auc']:.3f}"),
        _kpi("Features com drift (último lote)", f"{int(last['n_drifted'])}/{len(psi_matrix)}",
             "crit" if last["drift_share"] >= mcfg["dataset_drift_share"] else "ok"),
    ])

    cards: list[str] = []

    # ------------------------------------------------------------------ 1. Saúde do pipeline
    status_counts = stages.groupby(["stage", "status"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for status in status_counts.columns:
        fig.add_bar(y=status_counts.index, x=status_counts[status], name=status, orientation="h",
                    marker_color=STATUS_COLOR.get(status, PALETTE["gray"]))
    fig.update_layout(**LAYOUT, barmode="stack", title="Status de execução por tarefa")
    cards.append(_card("Saúde do pipeline: status das tarefas", _fig_html(fig, batch_axis=False),
                       note="Cada barra conta execuções registradas em logs/stages.jsonl."))

    lat = stages.groupby("stage")["duration_ms"].agg(["mean", "max"]).sort_values("mean")
    fig = go.Figure()
    fig.add_bar(y=lat.index, x=lat["mean"], name="média", orientation="h", marker_color=PALETTE["blue"])
    fig.add_bar(y=lat.index, x=lat["max"], name="máxima", orientation="h", marker_color="#9ec5fe")
    fig.update_layout(**LAYOUT, barmode="group", title="Latência por tarefa (ms)")
    cards.append(_card("Latência das tarefas", _fig_html(fig, batch_axis=False)))

    per_batch = stages[stages["batch"].isin(hist["batch"])]
    vol = hist.set_index("batch")["rows_received"]
    scored_rows = (per_batch[per_batch["stage"] == "inference"].set_index("batch")["rows_out"]
                   .reindex(vol.index).fillna(0))
    fig = go.Figure()
    fig.add_bar(x=vol.index, y=vol.values, name="recebidos", marker_color=PALETTE["gray"])
    fig.add_bar(x=scored_rows.index, y=scored_rows.values, name="pontuados", marker_color=PALETTE["ok"])
    fig.update_layout(**LAYOUT, barmode="group", title="Volume de dados por lote")
    cards.append(_card("Volume processado", _fig_html(fig),
                       note="Lote do incidente: recebido, mas 0 registros pontuados (bloqueado)."))

    infra = stages.dropna(subset=["memory_rss_mb"]).reset_index(drop=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=infra.index, y=infra["memory_rss_mb"], name="memória RSS (MB)",
                    mode="lines+markers", line_color=PALETTE["purple"])
    fig.add_bar(x=infra.index, y=infra["cpu_time_s"], name="CPU (s)", marker_color="#c8b6f5",
                secondary_y=True)
    fig.update_layout(**LAYOUT, title="Infraestrutura: memória e CPU por execução")
    fig.update_xaxes(title="execução (ordem)")
    cards.append(_card("Estabilidade da infraestrutura", _fig_html(fig, batch_axis=False)))

    # ------------------------------------------------------------------ 2. Contratos
    rows = []
    for r in hist.itertuples():
        badge = "ok" if r.contract_passed else "crit"
        link = f"<a href='data_contracts/{r.contract_report}'>relatório</a>" if isinstance(
            r.contract_report, str) else "-"
        rows.append(f"<tr><td>{html.escape(r.batch)}</td><td>{r.rows_received:,}</td>"
                    f"<td><span class='pill {badge}'>{'APROVADO' if r.contract_passed else 'BLOQUEADO'}"
                    f"</span></td><td>{int(r.contract_violations)}</td><td>{link}</td></tr>")
    cards.append(_card("Data Contracts (Pandera): quality gate por lote",
                       "<table><tr><th>Lote</th><th>Registros</th><th>Contrato</th><th>Violações</th>"
                       "<th>Detalhe</th></tr>" + "".join(rows) + "</table>"))

    # ------------------------------------------------------------------ 3. Drift
    fig = go.Figure(go.Heatmap(
        z=psi_matrix.values, x=psi_matrix.columns, y=psi_matrix.index, colorscale=[
            [0, "#dafbe1"], [0.1 / 0.6, "#fff8c5"], [0.25 / 0.6, "#ffb77c"], [1, "#cf222e"]],
        zmin=0, zmax=0.6, colorbar=dict(title="PSI"),
        text=psi_matrix.round(2).values, texttemplate="%{text}", textfont=dict(size=9)))
    fig.update_layout(**{**LAYOUT, "height": 620}, title="PSI por feature × mês (0,10 atenção | 0,25 crítico)")
    cards.append(_card("Data Drift: PSI por feature ao longo do tempo", _fig_html(fig), wide=True))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=scored["batch"], y=scored["drift_share"], name="% features com drift",
                marker_color=[PALETTE["crit"] if v >= mcfg["dataset_drift_share"] else PALETTE["ok"]
                              for v in scored["drift_share"]])
    fig.add_scatter(x=scored["batch"], y=scored["prediction_psi"], name="PSI do score (prediction drift)",
                    mode="lines+markers", line_color=PALETTE["purple"], secondary_y=True)
    fig.add_hline(y=mcfg["dataset_drift_share"], line_dash="dot", line_color=PALETTE["crit"],
                  annotation_text="limite dataset drift")
    fig.update_yaxes(tickformat=".0%", secondary_y=False)
    fig.update_layout(**LAYOUT, title="Dataset drift e drift do score")
    cards.append(_card("Dataset drift", _fig_html(fig)))

    # ------------------------------------------------------------------ 4. Performance
    perf = scored.dropna(subset=["perf_auc"])
    fig = go.Figure()
    fig.add_scatter(x=perf["batch"], y=perf["perf_auc"], name="AUC", mode="lines+markers",
                    line_color=PALETTE["blue"])
    fig.add_scatter(x=perf["batch"], y=perf["perf_ks"], name="KS", mode="lines+markers",
                    line_color=PALETTE["purple"])
    fig.add_hline(y=ref["auc"], line_dash="dash", line_color=PALETTE["gray"], annotation_text="AUC ref.")
    fig.add_hline(y=ref["auc"] - mcfg["auc_drop_critical"], line_dash="dot", line_color=PALETTE["crit"],
                  annotation_text="limite crítico")
    fig.update_layout(**LAYOUT, title="Performance do modelo (rótulos com atraso)")
    cards.append(_card("Degradação do modelo", _fig_html(fig)))

    fig = go.Figure()
    fig.add_scatter(x=perf["batch"], y=perf["perf_default_rate"], name="default observado",
                    mode="lines+markers", line_color=PALETTE["crit"])
    fig.add_scatter(x=perf["batch"], y=perf["mean_score"], name="PD média prevista",
                    mode="lines+markers", line_color=PALETTE["blue"])
    fig.add_hline(y=ref["default_rate"], line_dash="dash", line_color=PALETTE["gray"],
                  annotation_text="default ref.")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(**LAYOUT, title="Inadimplência real × prevista (calibração)")
    cards.append(_card("Risco previsto × realizado", _fig_html(fig),
                       note="O descolamento entre as curvas é a assinatura do concept drift."))

    # ------------------------------------------------------------------ 5. Fairness
    fig = go.Figure()
    for col, name, color in [("fair_di_sex", "DI sexo", PALETTE["blue"]),
                             ("fair_di_age_band", "DI faixa etária", PALETTE["purple"])]:
        fig.add_scatter(x=scored["batch"], y=scored[col], name=name, mode="lines+markers",
                        line_color=color)
    fig.add_hline(y=mcfg["fairness_di_min"], line_dash="dot", line_color=PALETTE["crit"],
                  annotation_text="regra dos 4/5")
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(**LAYOUT, title="Disparate Impact (aprovação mín./máx. entre grupos)")
    cards.append(_card("Fairness: vieses", _fig_html(fig)))

    # ------------------------------------------------------------------ 6. Root cause
    rc = summary.get("root_cause", {})
    if rc.get("features"):
        rca = pd.DataFrame(rc["features"]).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=rca["score_shift_caused"], y=rca["feature"], orientation="h",
                               marker_color=[PALETTE["crit"] if v > 0 else PALETTE["ok"]
                                             for v in rca["score_shift_caused"]]))
        fig.update_layout(**LAYOUT, title=f"Efeito causal no score médio: lote {rc['batch']}")
        fig.update_xaxes(title="Δ PD média atribuída à feature (intervenção)")
        sm = rc["summary"]
        decomp = (f"<table><tr><th>Componente</th><th>ΔAUC</th></tr>"
                  f"<tr><td>Queda total (referência para produção)</td><td>{sm['auc_drop_total']:.3f}</td></tr>"
                  f"<tr><td>Atribuível a data drift (recuperável)</td><td>{sm['auc_drop_covariate']:.3f}</td></tr>"
                  f"<tr><td>Atribuível a concept drift (residual)</td><td>{sm['auc_drop_concept']:.3f}</td></tr>"
                  f"</table>")
        cards.append(_card("Análise causal: causa raiz da degradação", _fig_html(fig, batch_axis=False) + decomp,
                           wide=True, note="Intervenção que troca a distribuição de cada feature pela de referência "
                           "com quantile mapping; ver docs/causal_analysis.md."))

    # ------------------------------------------------------------------ 7. Alertas
    if not alerts.empty:
        a = alerts.sort_values("timestamp", ascending=False)
        rows = "".join(
            f"<tr><td>{html.escape(str(r.batch))}</td><td><span class='pill "
            f"{'crit' if r.severity == 'critical' else 'warn'}'>{r.severity}</span></td>"
            f"<td>{html.escape(r.kind)}</td><td>{html.escape(r.message)}</td></tr>"
            for r in a.itertuples())
        cards.append(_card(f"Alertas ({len(a)})", "<div class='scroll'><table><tr><th>Lote</th>"
                           "<th>Severidade</th><th>Tipo</th><th>Mensagem</th></tr>" + rows
                           + "</table></div>", wide=True))

    # ------------------------------------------------------------------ 8. Links
    drift_dir = reports / "drift"
    links = "".join(f"<li><a href='drift/{p.name}'>{p.name}</a></li>"
                    for p in sorted(drift_dir.glob("evidently_*.html")))
    cards.append(_card("Relatórios Evidently AI", f"<ul>{links}</ul>",
                       note="Relatórios interativos com testes estatísticos por feature."))
    status_rows = "".join(
        f"<tr><td>{html.escape(r.batch)}</td><td><span class='pill "
        f"{ {'healthy': 'ok', 'attention': 'warn'}.get(r.status, 'crit') }'>{r.status}</span></td>"
        f"<td>{'' if pd.isna(r.drift_share) else f'{r.drift_share:.0%}'}</td>"
        f"<td>{'' if pd.isna(r.perf_auc) else f'{r.perf_auc:.3f}'}</td></tr>"
        for r in hist.itertuples())
    cards.append(_card("Status do modelo por lote", "<table><tr><th>Lote</th><th>Status</th>"
                       "<th>Drift</th><th>AUC</th></tr>" + status_rows + "</table>"))

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Scoring Monitoring</title>
<script>{get_plotlyjs()}</script>
<style>
:root{{--bg:#f6f8fa;--card:#fff;--fg:#1f2328;--muted:#59636e;--line:#d0d7de}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
 font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
header{{background:#0d1117;color:#fff;padding:20px 24px}} header h1{{margin:0 0 4px;font-size:22px}}
header p{{margin:0;color:#9da7b3;font-size:13px}}
main{{max-width:1400px;margin:auto;padding:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-left:5px solid {PALETTE['blue']};
 border-radius:8px;padding:10px 14px}} .kpi span{{font-size:12px;color:var(--muted)}}
.kpi b{{display:block;font-size:24px;margin-top:4px}}
.kpi.ok{{border-left-color:{PALETTE['ok']}}} .kpi.warn{{border-left-color:{PALETTE['warn']}}}
.kpi.crit{{border-left-color:{PALETTE['crit']}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}}
.card.wide{{grid-column:1/-1}} .card h3{{margin:0 0 6px;font-size:15px}}
.note{{margin:0 0 6px;color:var(--muted);font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eaeef2;vertical-align:top}} th{{background:#f6f8fa}}
.pill{{padding:2px 8px;border-radius:999px;color:#fff;font-size:11px;font-weight:600}}
.pill.ok{{background:{PALETTE['ok']}}} .pill.warn{{background:{PALETTE['warn']}}} .pill.crit{{background:{PALETTE['crit']}}}
.scroll{{max-height:420px;overflow:auto}} a{{color:{PALETTE['blue']}}}
@media (max-width:600px){{.grid{{grid-template-columns:1fr}} main{{padding:12px}}}}
</style></head><body>
<header><h1>Credit Scoring: Observabilidade do Modelo em Produção</h1>
<p>Modelo {html.escape(meta['model_name'])} · versão {html.escape(meta['training_run_id'])} ·
threshold {meta['threshold']:.3f} · execução {html.escape(summary['run_id'])} · gerado em {generated}</p></header>
<main><div class="kpis">{kpis}</div><div class="grid">{''.join(cards)}</div></main>
</body></html>"""
    out = reports / "monitoring_dashboard.html"
    out.write_text(page, encoding="utf-8")
    log.info(f"Dashboard gerado em {out}")
    return str(out)
