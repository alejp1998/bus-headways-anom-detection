import dash
import plotly.graph_objects as go
import plotly.io as pio

# Plotly template with transparent background so the CSS theme shows through.
clean_template = go.layout.Template()
clean_template.layout = go.Layout(
    font={"family": "DM Sans, sans-serif", "color": "#64748B"},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis={
        "gridcolor": "rgba(100, 116, 139, 0.15)",
        "zerolinecolor": "rgba(100, 116, 139, 0.25)",
        "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
    },
    yaxis={
        "gridcolor": "rgba(100, 116, 139, 0.15)",
        "zerolinecolor": "rgba(100, 116, 139, 0.25)",
        "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
    },
    legend={"font": {"family": "DM Sans, sans-serif", "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
    hoverlabel={
        "font": {"family": "Space Grotesk, sans-serif", "size": 12, "color": "#F8FAFC"},
        "bgcolor": "#1E293B",
        "bordercolor": "rgba(139, 92, 246, 0.6)",
    },
)

pio.templates["clean_modern"] = clean_template
pio.templates.default = "clean_modern"


def theme_layout(theme="dark", uirevision=None):
    """Return plotly layout overrides tuned for the active frontend theme with high-contrast tooltips."""
    dark = theme == "dark"
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, sans-serif", "color": "#94A3B8" if dark else "#334155"},
        "xaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.12)" if dark else "rgba(100, 116, 139, 0.12)",
            "zerolinecolor": "rgba(148, 163, 184, 0.2)" if dark else "rgba(100, 116, 139, 0.2)",
            "tickfont": {
                "family": "JetBrains Mono, monospace",
                "size": 11,
                "color": "#94A3B8" if dark else "#475569",
            },
        },
        "yaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.12)" if dark else "rgba(100, 116, 139, 0.12)",
            "zerolinecolor": "rgba(148, 163, 184, 0.2)" if dark else "rgba(100, 116, 139, 0.2)",
            "tickfont": {
                "family": "JetBrains Mono, monospace",
                "size": 11,
                "color": "#94A3B8" if dark else "#475569",
            },
        },
        "title": {
            "font": {
                "family": "Space Grotesk, sans-serif",
                "size": 14,
                "color": "#F8FAFC" if dark else "#0F172A",
            }
        },
        "hoverlabel": {
            "font": {
                "family": "Space Grotesk, sans-serif",
                "size": 13,
                "color": "#F8FAFC" if dark else "#0F172A",
            },
            "bgcolor": "#1E293B" if dark else "#FFFFFF",
            "bordercolor": "rgba(139, 92, 246, 0.8)",
        },
    }
    if uirevision is not None:
        layout["uirevision"] = uirevision
    return layout


# Initialize Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True, update_title="")
app.title = "Bus Headways | Real-Time Anomaly Detection"
server = app.server
