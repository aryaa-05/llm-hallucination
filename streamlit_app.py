"""
Hallucination Detector — Streamlit UI
Multi-source cross-verification: Web+NLI + QA Consistency + XGBoost fusion.
"""


import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HalluciDetect",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
}

.verdict-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 1px;
}

.verdict-hallucination {
    background: #2d0a0a;
    border: 2px solid #ff3333;
    color: #ff6666;
}

.verdict-uncertain {
    background: #2d2200;
    border: 2px solid #ffaa00;
    color: #ffcc44;
}

.verdict-factual {
    background: #0a2d14;
    border: 2px solid #00cc55;
    color: #44ee77;
}

.verdict-unable {
    background: #1a1a2e;
    border: 2px solid #6666aa;
    color: #9999cc;
}

.claim-card {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
    border-left: 4px solid;
    background: #1a1a1a;
}

.claim-hallucinated { border-color: #ff3333; }
.claim-uncertain    { border-color: #ffaa00; }
.claim-factual      { border-color: #00cc55; }

.claim-text {
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 0.4rem;
    color: #e8e8e8;
}

.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    margin-right: 6px;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}

.mini-metric {
    background: #252525;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.78rem;
    color: #aaa;
}

.mini-metric span {
    color: #fff;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin: 1.5rem 0 0.8rem 0;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.4rem;
}

.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    color: #e8e8e8 !important;
}

.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    background: #1a1a1a !important;
    border: 2px solid #444 !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    border-color: #00cc55 !important;
    color: #00cc55 !important;
}

[data-testid="stSidebar"] {
    background: #0d0d0d;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
def verdict_class(verdict: str) -> str:
    v = verdict.upper()
    if "HALLUCINATION" in v:    return "verdict-hallucination"
    if "UNCERTAIN"     in v:    return "verdict-uncertain"
    if "FACTUAL"       in v:    return "verdict-factual"
    return "verdict-unable"

def claim_class(verdict: str) -> str:
    v = verdict.upper()
    if "HALLUCINATION" in v:    return "claim-hallucinated"
    if "UNCERTAIN"     in v:    return "claim-uncertain"
    return "claim-factual"

def risk_color(risk: float) -> str:
    if risk >= 0.70: return "#ff3333"
    if risk >= 0.35: return "#ffaa00"
    return "#00cc55"

def risk_emoji(verdict: str) -> str:
    v = verdict.upper()
    if "HALLUCINATION" in v: return "[!]"
    if "UNCERTAIN"     in v: return "[?]"
    if "FACTUAL"       in v: return "[+]"
    return "[-]"


def make_gauge(risk: float, verdict: str) -> go.Figure:
    color = risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(risk * 100, 1),
        title = {"text": "Hallucination Risk %", "font": {"family": "Space Mono", "size": 13, "color": "#888"}},
        number= {"suffix": "%", "font": {"family": "Space Mono", "size": 32, "color": color}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickfont": {"color": "#555", "size": 10}},
            "bar"       : {"color": color, "thickness": 0.25},
            "bgcolor"   : "#1a1a1a",
            "bordercolor": "#2a2a2a",
            "steps"     : [
                {"range": [0,  35], "color": "#0a2d14"},
                {"range": [35, 70], "color": "#2d2200"},
                {"range": [70, 100],"color": "#2d0a0a"},
            ],
            "threshold" : {
                "line" : {"color": color, "width": 3},
                "thickness": 0.8,
                "value": risk * 100,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="#0d0d0d", font_color="#e8e8e8",
    )
    return fig


def make_shap_chart(shap_values: dict) -> go.Figure:
    items   = sorted(shap_values.items(), key=lambda x: abs(x[1]))
    labels  = [k.replace("_", " ") for k, _ in items]
    values  = [v for _, v in items]
    # Positive SHAP = pushes risk UP (increases hallucination likelihood) = red.
    # Negative SHAP = pushes risk DOWN (supports factuality)         = green.
    colors  = ["#ff4444" if v > 0 else "#00cc55" for v in values]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = labels,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:+.3f}" for v in values],
        textposition= "outside",
        textfont    = {"family": "Space Mono", "size": 10, "color": "#aaa"},
    ))
    fig.update_layout(
        title       = {"text": "Feature Contributions (SHAP)", "font": {"family": "Space Mono", "size": 12, "color": "#888"}},
        xaxis       = {"title": "", "gridcolor": "#2a2a2a", "zerolinecolor": "#444"},
        yaxis       = {"title": "", "tickfont": {"family": "DM Sans", "size": 11, "color": "#aaa"}},
        height      = 280,
        margin      = dict(l=10, r=60, t=40, b=10),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor = "#111",
        font_color  = "#e8e8e8",
        showlegend  = False,
    )
    return fig


def make_claims_risk_chart(claim_results: list) -> go.Figure:
    labels = [f"Claim {i+1}" for i in range(len(claim_results))]
    risks  = [r["meta"]["hallucination_risk"] for r in claim_results]
    colors = [risk_color(r) for r in risks]

    fig = go.Figure(go.Bar(
        x            = labels,
        y            = risks,
        marker_color = colors,
        text         = [f"{r:.2f}" for r in risks],
        textposition = "outside",
        textfont     = {"family": "Space Mono", "size": 11},
    ))
    fig.add_hline(y=0.70, line_dash="dot", line_color="#ff3333", opacity=0.6)
    fig.add_hline(y=0.35, line_dash="dot", line_color="#ffaa00", opacity=0.6)
    fig.update_layout(
        title      = {"text": "Per-Claim Hallucination Risk", "font": {"family": "Space Mono", "size": 12, "color": "#888"}},
        yaxis      = {"range": [0, 1.1], "gridcolor": "#2a2a2a", "tickfont": {"color": "#555"}},
        xaxis      = {"tickfont": {"color": "#aaa", "family": "Space Mono"}},
        height     = 240,
        margin     = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor ="#111",
        font_color = "#e8e8e8",
        showlegend = False,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### HalluciDetect")
    st.markdown("---")
    st.markdown("**Pipeline stages**")
    st.markdown("""
- **Stage 1** — Claim Decomposer (LLM)
- **Stage 2a** — Semantic Web Search + DeBERTa-v3 NLI
- **Stage 2b** — QA Entropy / Variance (DeBERTa-v3)
- **Stage 3** — Hybrid Meta-Classifier (ML + LLM Judge)
""")
    st.markdown("---")

    show_raw = st.checkbox("Show raw scores per claim", value=False)
    show_web = st.checkbox("Show web snippets", value=False)
    show_qa  = st.checkbox("Show QA pairs", value=False)

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.75rem; color:#555; font-family: Space Mono, monospace;">
LLM ROUTER<br>Gemini primary → Groq → OpenRouter<br>SEMANTIC search · DeBERTa-v3 NLI<br>SelfCheckGPT QA · SHAP
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem; color:#444; font-family: DM Sans, sans-serif;">
<b>Research demo.</b> Full methodology in <code>documentation.md</code>.
Components: claim decomposition (LLM router) · web evidence + bidirectional NLI · QA self-consistency · hybrid XGBoost/heuristic + LLM judge · SHAP explainability.
</div>
""", unsafe_allow_html=True)


# ── Main UI ──────────────────────────────────────────────────
st.markdown('<div class="main-title">HalluciDetect</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-source hallucination detection · Claim decomposition → Semantic Search × QA Entropy → Hybrid Fusion</div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────
EXAMPLES = {
    "Factual (Water)":
        "Water boils at 100 degrees Celsius at sea level.",
    "Hallucinated (Titanic)":
        "The RMS Titanic was a British passenger liner operated by the White Star Line. It sank in the North Atlantic Ocean on April 15, 1912, after being struck by a German submarine during its maiden voyage.",
    "Mixed (Apollo 11)":
        "The Apollo 11 mission was the first crewed moon landing, commanded by Neil Armstrong in 1969. They spent 12 days on the lunar surface and discovered water ice.",
}

col_inp, col_ex = st.columns([3, 1])
with col_ex:
    st.markdown("**Load example:**")
    for label, text in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["input_text"] = text

with col_inp:
    input_text = st.text_area(
        "Paste LLM-generated text to verify:",
        value  = st.session_state.get("input_text", ""),
        height = 160,
        placeholder = "Enter any LLM response here...",
        key    = "input_text",
    )

col_run, col_slider = st.columns([1, 2])
with col_slider:
    max_claims = st.slider(
        "Max claims to verify",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Higher = more thorough but slower. Use 2-3 for single sentences, 6-10 for dense multi-sentence paragraphs.",
    )
    st.caption("Increase for longer or denser text (e.g. historical facts, medical descriptions).")
with col_run:
    run_btn = st.button("RUN DETECTION", use_container_width=True)

# ── Run pipeline ─────────────────────────────────────────────
if run_btn:
    if not input_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    # Lazy import after env is ready
    from pipeline.orchestrator import run_pipeline

    progress_bar  = st.progress(0)
    status_text   = st.empty()

    def progress_cb(stage: str, pct: int):
        progress_bar.progress(pct / 100)
        status_text.markdown(f"<small style='color:#555;font-family:Space Mono'>{stage}</small>", unsafe_allow_html=True)

    with st.spinner(""):
        t0     = time.time()
        report = run_pipeline(input_text.strip(), progress_cb=progress_cb, max_claims=max_claims)
        elapsed = time.time() - t0

    progress_bar.empty()
    status_text.empty()

    st.session_state["report"]  = report
    st.session_state["elapsed"] = elapsed

# ── Display results ──────────────────────────────────────────
if "report" in st.session_state:
    report  = st.session_state["report"]
    elapsed = st.session_state.get("elapsed", 0)

    verdict = report["verdict"]
    risk    = report["overall_risk"]

    # ── Top verdict ──────────────────────────────────────────
    st.markdown(
        f'<div class="verdict-box {verdict_class(verdict)}">'
        f'{risk_emoji(verdict)}&nbsp;&nbsp;{verdict}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Summary row ──────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Risk",     f"{risk:.0%}")
    c2.metric("Total Claims",     report["total_claims"])
    c3.metric("Hallucinated",  report["confirmed_hallucinations"])
    c4.metric("Uncertain",     report["uncertain_claims"])
    c5.metric("Factual",       report["factual_claims"])

    st.caption(
        f"Completed in {elapsed:.1f}s · "
        "Thresholds: risk < 0.35 = FACTUAL, 0.35–0.70 = UNCERTAIN, ≥ 0.70 = HALLUCINATION"
    )

    if not report["claim_results"]:
        st.info("No verifiable claims were extracted.")
        st.stop()

    # ── Charts row ───────────────────────────────────────────
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(make_gauge(risk, verdict), use_container_width=True)
    with g2:
        st.plotly_chart(make_claims_risk_chart(report["claim_results"]), use_container_width=True)

    # ── Per-claim breakdown ──────────────────────────────────
    st.markdown('<div class="section-header">Claim-by-Claim Analysis</div>', unsafe_allow_html=True)

    for i, cr in enumerate(report["claim_results"]):
        claim    = cr["claim"]
        meta     = cr["meta"]
        web      = cr["web_nli"]
        qa       = cr["qa"]
        cr_class = claim_class(meta["verdict"])
        r_color  = risk_color(meta["hallucination_risk"])

        with st.container():
            st.markdown(
                f'<div class="claim-card {cr_class}">'
                f'<div class="claim-text"><b>Claim {i+1}:</b> {claim}</div>'
                f'<div class="metric-row">'
                f'<div class="mini-metric">Risk <span style="color:{r_color}">{meta["hallucination_risk"]:.2f}</span></div>'
                f'<div class="mini-metric">Verdict <span>{meta["verdict"]}</span></div>'
                f'<div class="mini-metric">Conf <span>{meta["confidence"]:.2f}</span></div>'
                f'<div class="mini-metric">Web snippets <span>{web["num_snippets"]}</span></div>'
                f'<div class="mini-metric">QA score <span>{qa["consistency_score"]:.2f}</span></div>'
                f'<div class="mini-metric">Model <span>{meta["model_used"]}</span></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # SHAP chart
            with st.expander(f"SHAP feature contributions — Claim {i+1}"):
                col_shap, col_branch = st.columns([2, 1])
                with col_shap:
                    st.plotly_chart(make_shap_chart(meta["shap_values"]), use_container_width=True)
                with col_branch:
                    st.markdown("**Branch weights**")
                    bw = meta["branch_weights"]
                    st.markdown(f"- Web+NLI: `{bw['web']:.0%}`")
                    st.markdown(f"- QA:      `{bw['qa']:.0%}`")

                    if show_raw:
                        st.markdown("**Raw features**")
                        feat_df = pd.DataFrame([meta["features"]]).T.rename(columns={0: "value"})
                        feat_df["value"] = feat_df["value"].round(3)
                        st.dataframe(feat_df, use_container_width=True)

            # Web snippets
            if show_web and web["snippet_scores"]:
                with st.expander(f"Web evidence — Claim {i+1}"):
                    st.caption(f"Search query: `{web['search_query']}`")
                    for j, s in enumerate(web["snippet_scores"]):
                        ecol, ccol = st.columns([1, 1])
                        ecol.metric(f"Snippet {j+1} entailment",    f"{s['entailment']:.2f}")
                        ccol.metric(f"Snippet {j+1} contradiction", f"{s['contradiction']:.2f}")
                        st.caption(s.get("snippet", "")[:300])
                        st.markdown("---")

            # QA pairs
            if show_qa and qa.get("qa_pairs"):
                with st.expander(f"QA pairs — Claim {i+1}"):
                    for pair in qa["qa_pairs"]:
                        st.markdown(f"**Q:** {pair['question']}")
                        st.markdown(f"**A:** {pair['answer']}")
                        cols = st.columns(3)
                        cols[0].metric("NLI Support", f"{pair['electra_score']:.3f}")
                        cols[1].metric("BERTScore", f"{pair['bertscore_f1']:.3f}")
                        cols[2].metric("Contradiction", f"{pair['contradiction']:.3f}")
                        st.markdown("---")

    # ── Full JSON export ─────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    import json
    st.download_button(
        label    = "Download full report (JSON)",
        data     = json.dumps(report, indent=2),
        file_name= "hallucidetect_report.json",
        mime     = "application/json",
    )
