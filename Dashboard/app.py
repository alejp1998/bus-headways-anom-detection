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
        "font": {"family": "JetBrains Mono, monospace", "size": 12},
        "bgcolor": "#1E293B",
        "bordercolor": "rgba(139, 92, 246, 0.4)",
    },
)

pio.templates["clean_modern"] = clean_template
pio.templates.default = "clean_modern"


def theme_layout(theme="dark"):
    """Return plotly layout overrides tuned for the active frontend theme."""
    dark = theme == "dark"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, sans-serif", "color": "#94A3B8" if dark else "#475569"},
        "xaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.15)" if dark else "rgba(100, 116, 139, 0.15)",
            "zerolinecolor": "rgba(148, 163, 184, 0.25)" if dark else "rgba(100, 116, 139, 0.25)",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
        },
        "yaxis": {
            "gridcolor": "rgba(148, 163, 184, 0.15)" if dark else "rgba(100, 116, 139, 0.15)",
            "zerolinecolor": "rgba(148, 163, 184, 0.25)" if dark else "rgba(100, 116, 139, 0.25)",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 11},
        },
        "title": {
            "font": {
                "family": "Space Grotesk, sans-serif",
                "size": 14,
                "color": "#F8FAFC" if dark else "#0F172A",
            }
        },
        "hoverlabel": {
            "font": {"family": "JetBrains Mono, monospace", "size": 12},
            "bgcolor": "#1E293B" if dark else "#FFFFFF",
            "bordercolor": "rgba(139, 92, 246, 0.4)",
            "font_color": "#F8FAFC" if dark else "#0F172A",
        },
    }


# Initialize Dash application
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Bus Headways | Real-Time Anomaly Detection"
server = app.server
