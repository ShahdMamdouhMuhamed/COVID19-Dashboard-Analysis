import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
CSV_PATH = "data/clean_data.csv"

try:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
except FileNotFoundError:
    raise FileNotFoundError("Dataset not found. Check CSV_PATH.")
    np.random.seed(42)
    continents = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"]
    countries  = {
        "Asia":          ["China", "India", "Japan", "South Korea", "Indonesia"],
        "Europe":        ["Germany", "France", "United Kingdom", "Italy", "Spain"],
        "North America": ["United States", "Canada", "Mexico", "Cuba", "Honduras"],
        "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
        "Africa":        ["South Africa", "Nigeria", "Ethiopia", "Egypt", "Kenya"],
        "Oceania":       ["Australia", "New Zealand", "Papua New Guinea", "Fiji", "Samoa"],
    }
    rows = []
    for cont, locs in countries.items():
        for loc in locs:
            rows.append({
                "continent": cont, "location": loc,
                "total_cases_per_million":      np.random.uniform(5_000,  300_000),
                "total_deaths_per_million":     np.random.uniform(10,     3_000),
                "people_fully_vaccinated_per_hundred": np.random.uniform(5, 95),
                "positive_rate":                np.random.uniform(0.01,  0.35),
                "case_fatality_rate":            np.random.uniform(0.1,   5.0),
                "vaccination_rate":             np.random.uniform(5,     95),
            })
    df = pd.DataFrame(rows)
    # ── BUG 4 FIX: add a date column to fallback DataFrame so the callback
    #    never raises KeyError on filtered_df["date"] ──────────────────────
    dates = pd.date_range("2020-01-01", "2024-07-31", periods=len(df))
    df["date"] = dates

# Ensure date is datetime
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

# Global min/max for date picker
MIN_DATE = df["date"].min() if "date" in df.columns else pd.to_datetime("2020-01-01")
MAX_DATE = df["date"].max() if "date" in df.columns else pd.to_datetime("2024-07-31")

# ── Latest snapshot per country (will be updated dynamically in callback) ─────
if "date" in df.columns:
    latest = (
        df.sort_values("date")
          .groupby("location", as_index=False)
          .last()
    )
else:
    latest = df.copy()

# Drop World / income aggregates if present
agg_tokens = ["world", "income", "union", "international"]
latest = latest[~latest["location"].str.lower().str.contains("|".join(agg_tokens), na=False)]

CONTINENTS = sorted(latest["continent"].dropna().unique())

METRICS = {
    "Total Cases per Million":              "total_cases_per_million",
    "Total Deaths per Million":             "total_deaths_per_million",
    "People Fully Vaccinated (%)":          "people_fully_vaccinated_per_hundred",
    "Positive Rate":                        "positive_rate",
    "Case Fatality Rate (%)":               "case_fatality_rate",
    "Vaccination Rate (%)":                 "vaccination_rate",
}

STACKED_METRICS = {
    "Cases & Deaths per Million": {
        "Cases per Million":  "total_cases_per_million",
        "Deaths per Million": "total_deaths_per_million",
    },
    "Vaccinated vs Unvaccinated (%)": {
        "Fully Vaccinated (%)":   "people_fully_vaccinated_per_hundred",
        "Not Vaccinated (%)":     "_unvaccinated",
    },
}

# Pre-compute derived columns
latest["_unvaccinated"] = 100 - latest.get("people_fully_vaccinated_per_hundred", pd.Series(50, index=latest.index)).fillna(50)

# ─────────────────────────────────────────────
# COLOUR PALETTE & LAYOUT (unchanged)
# ─────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "surface":   "#161b22",
    "border":    "#30363d",
    "accent1":   "#58a6ff",
    "accent2":   "#f78166",
    "accent3":   "#3fb950",
    "accent4":   "#d2a8ff",
    "text":      "#e6edf3",
    "subtext":   "#8b949e",
}

CONT_COLORS = ["#58a6ff","#f78166","#3fb950","#d2a8ff","#ffa657","#39d353"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="'IBM Plex Mono', monospace", color=PALETTE["text"], size=12),
    title_font   =dict(family="'IBM Plex Sans', sans-serif", color=PALETTE["text"], size=15),
    margin       =dict(l=60, r=30, t=60, b=60),
    xaxis        =dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"],
                       tickfont=dict(color=PALETTE["subtext"])),
    yaxis        =dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"],
                       tickfont=dict(color=PALETTE["subtext"])),
    legend       =dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["subtext"])),
    hoverlabel   =dict(bgcolor=PALETTE["surface"], font_color=PALETTE["text"]),
)

# ─────────────────────────────────────────────
# HELPERS (unchanged)
# ─────────────────────────────────────────────

def top_n(data: pd.DataFrame, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    return (
        data[["location", "continent", metric]]
        .dropna()
        .sort_values(metric, ascending=ascending)
        .head(n)
    )

def apply_layout(fig, title: str, xlab: str = "", ylab: str = "") -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title)

    if xlab:
        fig.update_xaxes(title_text=xlab, title_font_color=PALETTE["subtext"])

    if ylab:
        fig.update_yaxes(title_text=ylab, title_font_color=PALETTE["subtext"])

    return fig

# ─────────────────────────────────────────────
# CHART BUILDERS — Existing (unchanged except they now use dynamic "latest")
# ─────────────────────────────────────────────
# (All make_ functions below remain exactly as your teammates wrote them)

def make_column_chart(continent: str, metric_label: str) -> go.Figure:
    col  = METRICS[metric_label]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = top_n(data, col, n=10)
    fig  = px.bar(
        data, x="location", y=col,
        color_discrete_sequence=[PALETTE["accent1"]],
        text_auto=".2s",
    )
    fig.update_traces(marker_line_width=0, textfont_color=PALETTE["text"])
    return apply_layout(
        fig,
        f"Top 10 Countries — {metric_label}",
        "Country", metric_label,
    )

def make_bar_chart(continent: str, metric_label: str) -> go.Figure:
    col  = METRICS[metric_label]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = top_n(data, col, n=10, ascending=True)
    fig  = px.bar(
        data, x=col, y="location",
        orientation="h",
        color_discrete_sequence=[PALETTE["accent2"]],
        text_auto=".2s",
    )
    fig.update_traces(marker_line_width=0, textfont_color=PALETTE["text"])
    fig.update_yaxes(autorange="reversed")
    return apply_layout(
        fig,
        f"Lowest 10 Countries — {metric_label}",
        metric_label, "Country",
    )

def make_stacked_column(continent: str, stack_label: str) -> go.Figure:
    meta  = STACKED_METRICS[stack_label]
    data  = latest[latest["continent"] == continent] if continent != "All" else latest
    first_col = list(meta.values())[0]
    all_cols  = ["location", "continent"] + [c for c in meta.values() if c in latest.columns]
    data = (
        data[all_cols]
        .dropna(subset=[first_col])
        .sort_values(first_col, ascending=False)
        .head(10)
    )

    fig = go.Figure()
    for i, (label, col) in enumerate(meta.items()):
        if col not in data.columns:
            continue
        fig.add_trace(go.Bar(
            name=label, x=data["location"], y=data[col],
            marker_color=CONT_COLORS[i % len(CONT_COLORS)], marker_line_width=0,
        ))
    fig.update_layout(barmode="stack")
    return apply_layout(fig, f"Stacked Column — {stack_label}", "Country", "Value")

def make_stacked_bar(continent: str, stack_label: str) -> go.Figure:
    meta  = STACKED_METRICS[stack_label]
    data  = latest[latest["continent"] == continent] if continent != "All" else latest
    first_col = list(meta.values())[0]
    all_cols  = ["location", "continent"] + [c for c in meta.values() if c in latest.columns]
    data = (
        data[all_cols]
        .dropna(subset=[first_col])
        .sort_values(first_col, ascending=False)
        .head(10)
    )

    fig = go.Figure()
    for i, (label, col) in enumerate(meta.items()):
        if col not in data.columns:
            continue
        fig.add_trace(go.Bar(
            name=label, y=data["location"], x=data[col],
            orientation="h", marker_color=CONT_COLORS[i % len(CONT_COLORS)], marker_line_width=0,
        ))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return apply_layout(fig, f"Stacked Bar — {stack_label}", "Value", "Country")

def make_clustered_column(continent: str) -> go.Figure:
    cols = {
        "Cases / Million":  "total_cases_per_million",
        "Deaths / Million": "total_deaths_per_million",
    }
    if continent != "All":
        data = latest[latest["continent"] == continent]
        group_col = "location"
        data = data.nlargest(8, "total_cases_per_million")
    else:
        data = latest.copy()
        group_col = "continent"
        data = data.groupby("continent", as_index=False)[list(cols.values())].mean()

    fig = go.Figure()
    for i, (label, col) in enumerate(cols.items()):
        fig.add_trace(go.Bar(
            name=label,
            x=data[group_col], y=data[col],
            marker_color=CONT_COLORS[i],
            marker_line_width=0,
        ))
    fig.update_layout(barmode="group")
    return apply_layout(
        fig,
        "Clustered Column — Cases vs Deaths per Million",
        group_col.title(), "Per Million",
    )

def make_clustered_bar(continent: str) -> go.Figure:
    cols = {
        "Cases / Million":  "total_cases_per_million",
        "Deaths / Million": "total_deaths_per_million",
    }
    if continent != "All":
        data = latest[latest["continent"] == continent]
        group_col = "location"
        data = data.nlargest(8, "total_cases_per_million")
    else:
        data = latest.copy()
        group_col = "continent"
        data = data.groupby("continent", as_index=False)[list(cols.values())].mean()

    fig = go.Figure()
    for i, (label, col) in enumerate(cols.items()):
        fig.add_trace(go.Bar(
            name=label,
            y=data[group_col], x=data[col],
            orientation="h",
            marker_color=CONT_COLORS[i],
            marker_line_width=0,
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    return apply_layout(
        fig,
        "Clustered Bar — Cases vs Deaths per Million",
        "Per Million", group_col.title(),
    )

def make_scatter_chart(continent: str, x_metric: str, y_metric: str) -> go.Figure:
    x_col = METRICS[x_metric]
    y_col = METRICS[y_metric]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = data.dropna(subset=[x_col, y_col])
    
    fig = px.scatter(
        data, x=x_col, y=y_col, color="continent" if continent == "All" else None,
        hover_name="location",
        color_discrete_sequence=CONT_COLORS,
    )
    return apply_layout(
        fig,
        f"Scatter Chart — {x_metric} vs {y_metric}",
        x_metric, y_metric,
    )

def make_bubble_chart(continent: str, x_metric: str, y_metric: str, size_metric: str) -> go.Figure:
    x_col = METRICS[x_metric]
    y_col = METRICS[y_metric]
    size_col = METRICS[size_metric]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = data.dropna(subset=[x_col, y_col, size_col])
    
    fig = px.scatter(
        data, x=x_col, y=y_col, size=size_col, color="continent" if continent == "All" else None,
        hover_name="location",
        color_discrete_sequence=CONT_COLORS,
        size_max=60,
    )
    return apply_layout(
        fig,
        f"Bubble Chart — {x_metric} vs {y_metric} (Size: {size_metric})",
        x_metric, y_metric,
    )

def make_histogram(continent: str, metric: str) -> go.Figure:
    col = METRICS[metric]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = data.dropna(subset=[col])
    
    fig = px.histogram(
        data, x=col, color="continent" if continent == "All" else None,
        color_discrete_sequence=CONT_COLORS,
        nbins=20,
    )
    return apply_layout(
        fig,
        f"Histogram — Distribution of {metric}",
        metric, "Frequency",
    )

# ─────────────────────────────────────────────
# NEW CHARTS — YOUR PART (Week 6, 7, 8)
# ─────────────────────────────────────────────

def make_box_chart(continent: str, metric_label: str) -> go.Figure:
    """Week 6 — Box Chart"""
    col = METRICS[metric_label]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = data.dropna(subset=[col])
    fig = px.box(
        data,
        x="continent" if continent == "All" else "location",
        y=col,
        color="continent" if continent == "All" else None,
        color_discrete_sequence=CONT_COLORS,
        points="outliers",
    )
    return apply_layout(
        fig,
        f"Box Chart — {metric_label}",
        "Group" if continent == "All" else "Country",
        metric_label,
    )

def make_violin_chart(continent: str, metric_label: str) -> go.Figure:
    """Week 7 — Violin Chart (follows guideline exactly)"""
    col = METRICS[metric_label]
    data = latest[latest["continent"] == continent] if continent != "All" else latest
    data = data.dropna(subset=[col])
    fig = px.violin(
        data,
        x="continent" if continent == "All" else "location",
        y=col,
        color="continent" if continent == "All" else None,
        color_discrete_sequence=CONT_COLORS,
        box=True,          # inner box (as in guideline)
        points="outliers",
    )
    fig.update_traces(spanmode="hard")   # CRITICAL: prevents KDE extending beyond data
    return apply_layout(
        fig,
        f"Violin Chart — {metric_label}",
        "Group" if continent == "All" else "Country",
        metric_label,
    )

def make_line_chart(continent: str, metric_label: str, time_df: pd.DataFrame) -> go.Figure:
    """Week 8 — Line Chart + Moving Average (follows guideline)"""
    col = METRICS[metric_label]
    # ── BUG 5 FIX: guard against missing date column ──────────────────────
    if "date" not in time_df.columns or col not in time_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No time-series data available", showarrow=False, font_size=16)
        return apply_layout(fig, f"Line Chart — {metric_label} Over Time", "Date", metric_label)
    if continent != "All":
        time_df = time_df[time_df["continent"] == continent].copy()
    # Aggregate by date
    time_series = time_df.groupby("date")[col].mean().reset_index()
    time_series = time_series.dropna(subset=[col])
    if len(time_series) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data in selected date range", showarrow=False, font_size=16)
        return fig
    fig = px.line(
        time_series,
        x="date",
        y=col,
        color_discrete_sequence=[PALETTE["accent3"]],
    )
    fig.update_traces(line_width=3)
    # Add 7-day moving average (as shown in guideline)
    if len(time_series) > 7:
        time_series["MA_7"] = time_series[col].rolling(window=7, center=True, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=time_series["date"],
            y=time_series["MA_7"],
            name="7-day Moving Average",
            line=dict(color=PALETTE["accent2"], width=2.5, dash="dash"),
        ))
    return apply_layout(
        fig,
        f"Line Chart — {metric_label} Over Time",
        "Date",
        metric_label,
    )

def make_area_chart(continent: str, metric_label: str, time_df: pd.DataFrame) -> go.Figure:
    """Week 9 — Area Chart (time-series cumulative trend per continent)"""
    col = METRICS[metric_label]

    # Filter by continent if selected
    if continent != "All":
        time_df = time_df[time_df["continent"] == continent].copy()

    # Drop aggregates from the time-series slice
    time_df = time_df[
        ~time_df["location"].str.lower().str.contains("|".join(agg_tokens), na=False)
    ]

    if "date" not in time_df.columns or col not in time_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No time-series data available", showarrow=False, font_size=16)
        return apply_layout(fig, f"Area Chart — {metric_label} Over Time", "Date", metric_label)

    # Aggregate by date and continent (or just date when a single continent is chosen)
    if continent == "All":
        group_cols = ["date", "continent"]
        time_series = (
            time_df.dropna(subset=[col])
            .groupby(group_cols)[col]
            .mean()
            .reset_index()
        )
        fig = px.area(
            time_series,
            x="date",
            y=col,
            color="continent",
            color_discrete_sequence=CONT_COLORS,
            line_group="continent",
        )
    else:
        time_series = (
            time_df.dropna(subset=[col])
            .groupby("date")[col]
            .mean()
            .reset_index()
        )
        if time_series.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data in selected date range", showarrow=False, font_size=16)
            return apply_layout(fig, f"Area Chart — {metric_label} Over Time", "Date", metric_label)

        fig = px.area(
            time_series,
            x="date",
            y=col,
            color_discrete_sequence=[PALETTE["accent4"]],
        )

    fig.update_traces(line_width=1.5)
    return apply_layout(
        fig,
        f"Area Chart — {metric_label} Over Time",
        "Date",
        metric_label,
    )

# ─────────────────────────────────────────────
# LAYOUT HELPERS (unchanged)
# ─────────────────────────────────────────────
CARD = {
    "backgroundColor": PALETTE["surface"],
    "border":          f"1px solid {PALETTE['border']}",
    "borderRadius":    "8px",
    "padding":         "16px",
    "marginBottom":    "20px",
}

SECTION_LABEL = {
    "fontFamily": "'IBM Plex Mono', monospace",
    "fontSize":   "11px",
    "color":      PALETTE["subtext"],
    "letterSpacing": "0.12em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
}

DROPDOWN_STYLE = {
    "backgroundColor": PALETTE["bg"],
    "color":           PALETTE["text"],
    "border":          f"1px solid {PALETTE['border']}",
    "borderRadius":    "6px",
}

CHART_SECTION = lambda label, chart_id: html.Div([
    html.P(label, style=SECTION_LABEL),
    dcc.Graph(id=chart_id, config={"displayModeBar": False}, style={"height": "380px"}),
], style=CARD)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = dash.Dash(__name__, title="COVID-19 Comparison Charts")

app.layout = html.Div(
    style={
        "backgroundColor": PALETTE["bg"],
        "minHeight":       "100vh",
        "padding":         "28px 32px",
        "fontFamily":      "'IBM Plex Sans', sans-serif",
    },
    children=[
        # HEADER (unchanged)
        html.Div([
            html.Span("COVID-19", style={
                "fontFamily":    "'IBM Plex Mono', monospace",
                "fontSize":      "11px",
                "color":         PALETTE["accent1"],
                "letterSpacing": "0.15em",
                "textTransform": "uppercase",
            }),
            html.H1("Global Pandemic — Comparison Charts", style={
                "color":      PALETTE["text"],
                "fontSize":   "28px",
                "margin":     "4px 0 4px",
                "fontWeight": "700",
                "lineHeight": "1.2",
            }),
            html.P(" Column · Bar · Stacked · Clustered · Box · Violin · Line · Area", style={
                "color":    PALETTE["subtext"],
                "fontSize": "13px",
                "margin":   "0 0 28px",
            }),
        ]),

        # CONTROLS (existing + NEW DATE RANGE)
        html.Div([
            html.Div([
                html.P("Continent", style=SECTION_LABEL),
                dcc.Dropdown(
                    id="dd-continent",
                    options=[{"label": "🌍 All", "value": "All"}] +
                            [{"label": c, "value": c} for c in CONTINENTS],
                    value="All",
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                html.P("Primary Metric", style=SECTION_LABEL),
                dcc.Dropdown(
                    id="dd-metric",
                    options=[{"label": k, "value": k} for k in METRICS],
                    value=list(METRICS.keys())[0],
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                html.P("Stacked Metric Pair", style=SECTION_LABEL),
                dcc.Dropdown(
                    id="dd-stack",
                    options=[{"label": k, "value": k} for k in STACKED_METRICS],
                    value=list(STACKED_METRICS.keys())[0],
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "28px"}),

        # RELATIONSHIP CONTROLS (unchanged)
        html.Div([
            html.Div([
                html.P("X-Axis Metric", style=SECTION_LABEL),
                dcc.RadioItems(
                    id="radio-x",
                    options=[{"label": k, "value": k} for k in METRICS],
                    value=list(METRICS.keys())[0],
                    labelStyle={"display": "block", "color": PALETTE["text"], "fontSize": "12px"},
                ),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                html.P("Y-Axis Metric", style=SECTION_LABEL),
                dcc.RadioItems(
                    id="radio-y",
                    options=[{"label": k, "value": k} for k in METRICS],
                    value=list(METRICS.keys())[1],
                    labelStyle={"display": "block", "color": PALETTE["text"], "fontSize": "12px"},
                ),
            ], style={"flex": "1", "marginRight": "16px"}),

            html.Div([
                html.P("Size Metric (Bubble)", style=SECTION_LABEL),
                dcc.RadioItems(
                    id="radio-size",
                    options=[{"label": k, "value": k} for k in METRICS],
                    value=list(METRICS.keys())[2],
                    labelStyle={"display": "block", "color": PALETTE["text"], "fontSize": "12px"},
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "28px"}),

        # ── NEW: DATE RANGE SLIDER / PICKER (your required interactive element) ──
        html.Div([
            html.P("Date Range (filter all charts)", style=SECTION_LABEL),
            dcc.DatePickerRange(
                id="date-range",
                start_date=MIN_DATE,
                end_date=MAX_DATE,
                display_format="YYYY-MM-DD",
                style={
                    "backgroundColor": PALETTE["surface"],
                    "border": f"1px solid {PALETTE['border']}",
                    "borderRadius": "6px",
                    "padding": "8px",
                }
            ),
        ], style={"marginBottom": "28px"}),

        # WEEK 1 — COMPARISON CHARTS (unchanged)
        html.Div([
            html.P(" COMPARISON CHARTS", style={
                **SECTION_LABEL, "color": PALETTE["accent1"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Column Chart — Top 10 Countries", "chart-column"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Bar Chart (Horizontal) — Lowest 10 Countries", "chart-bar"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # WEEK 2 — STACKED & CLUSTERED (unchanged)
        html.Div([
            html.P(" COMPARISON CHARTS (STACKED & CLUSTERED)", style={
                **SECTION_LABEL, "color": PALETTE["accent3"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Stacked Column Chart", "chart-stacked-col"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Stacked Bar Chart (Horizontal)", "chart-stacked-bar"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
            html.Div([
                html.Div(CHART_SECTION("Clustered Column Chart", "chart-clustered-col"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Clustered Bar Chart (Horizontal)", "chart-clustered-bar"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # RELATIONSHIP CHARTS (unchanged)
        html.Div([
            html.P(" RELATIONSHIP CHARTS", style={
                **SECTION_LABEL, "color": PALETTE["accent4"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Scatter Chart", "chart-scatter"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Bubble Chart", "chart-bubble"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # DISTRIBUTION CHARTS — NOW INCLUDES BOX + VIOLIN (your part)
        html.Div([
            html.P(" DISTRIBUTION CHARTS (Box + Violin + Histogram)", style={
                **SECTION_LABEL, "color": PALETTE["accent2"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Histogram", "chart-histogram"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Box Chart", "chart-box"),
                         style={"flex": "1", "marginRight": "12px"}),
                html.Div(CHART_SECTION("Violin Chart", "chart-violin"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # TIME-SERIES CHARTS — Line Chart (your part)
        html.Div([
            html.P(" TIME-SERIES CHARTS", style={
                **SECTION_LABEL, "color": PALETTE["accent3"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Line Chart — Evolution Over Time", "chart-line"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # WEEK 9 — AREA CHART (NEW)
        html.Div([
            html.P(" TIME-SERIES CHARTS (AREA)", style={
                **SECTION_LABEL, "color": PALETTE["accent4"], "marginBottom": "16px",
            }),
            html.Div([
                html.Div(CHART_SECTION("Area Chart — Cumulative Metric Trend Over Time", "chart-area"),
                         style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),

        # FOOTER
        html.Div(
            "COVID-19 · Our World in Data · Dash + Plotly • Box + Violin + Line + Area + Date Range added",
            style={"color": PALETTE["subtext"], "fontSize": "11px",
                   "textAlign": "center", "marginTop": "16px"},
        ),
    ],
)

# ─────────────────────────────────────────────
# CALLBACK (updated with date range + new charts)
# ─────────────────────────────────────────────

@app.callback(
    Output("chart-column",        "figure"),
    Output("chart-bar",           "figure"),
    Output("chart-stacked-col",   "figure"),
    Output("chart-stacked-bar",   "figure"),
    Output("chart-clustered-col", "figure"),
    Output("chart-clustered-bar", "figure"),
    Output("chart-scatter",       "figure"),
    Output("chart-bubble",        "figure"),
    Output("chart-histogram",     "figure"),
    Output("chart-box",           "figure"),      # Week 6
    Output("chart-violin",        "figure"),      # Week 7
    Output("chart-line",          "figure"),      # Week 8
    Output("chart-area",          "figure"),      # Week 9
    Input("dd-continent", "value"),
    Input("dd-metric",    "value"),
    Input("dd-stack",     "value"),
    Input("radio-x",      "value"),
    Input("radio-y",      "value"),
    Input("radio-size",   "value"),
    Input("date-range",   "start_date"),          # Area chart
    Input("date-range",   "end_date"),            # Area chart
)
def update_charts(continent, metric_label, stack_label, x_metric, y_metric, size_metric,
                  start_date, end_date):
    # 1. Filter the full time-series data by selected date range
    filtered_df = df.copy()
    if start_date is not None and end_date is not None:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df["date"] >= start_date) &
            (filtered_df["date"] <= end_date)
        ]

    # 2. Compute latest snapshot WITHIN the selected date range
    if "date" in filtered_df.columns and not filtered_df.empty:
        filtered_latest = (
            filtered_df.sort_values("date")
              .groupby("location", as_index=False)
              .last()
        )
    else:
        filtered_latest = filtered_df.copy()

    # Drop aggregates
    filtered_latest = filtered_latest[
        ~filtered_latest["location"].str.lower().str.contains("|".join(agg_tokens), na=False)
    ]

    # ── BUG 2 FIX: recompute _unvaccinated on the fresh filtered snapshot ─
    filtered_latest["_unvaccinated"] = (
        100 - filtered_latest["people_fully_vaccinated_per_hundred"].fillna(50)
    )

    # ── BUG 1 FIX: update module-level latest safely for this request ─────
    global latest
    latest = filtered_latest

    return (
        make_column_chart(continent, metric_label),
        make_bar_chart(continent, metric_label),
        make_stacked_column(continent, stack_label),
        make_stacked_bar(continent, stack_label),
        make_clustered_column(continent),
        make_clustered_bar(continent),
        make_scatter_chart(continent, x_metric, y_metric),
        make_bubble_chart(continent, x_metric, y_metric, size_metric),
        make_histogram(continent, metric_label),
        make_box_chart(continent, metric_label),
        make_violin_chart(continent, metric_label),
        make_line_chart(continent, metric_label, filtered_df),   # passes full time-series data
        make_area_chart(continent, metric_label, filtered_df),   # Week 9 — Area Chart
    )


if __name__ == "__main__": 
    print(" Starting Dash app with Box + Violin + Line + Date Range...")
    print(f"Date range available: {MIN_DATE.date()} → {MAX_DATE.date()}")
    app.run(debug=False, port=8050)