import os
import sys

from dash import Input, Output, dcc, html

# Ensure Dashboard directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from apps import app_credits, app_home, app_realtime_london, app_realtime_madrid

# Custom HTML index template with fonts & styling
app.index_string = """
<!DOCTYPE html>
<html lang="en">
    <head>
        {%metas%}
        <title>Bus Headways | Real-Time Anomaly Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        {%favicon%}
        {%css%}
    </head>
    <body>
        <header class="app-navbar">
            <div class="flex-between">
                <div class="flex-gap">
                    <a class="brand-logo" href="/home">
                        <i class="fa-solid fa-bus-simple"></i>
                        <span>HEADWAYS</span>
                    </a>
                    <span class="badge-pill primary" style="font-size: 0.7rem; padding: 0.2rem 0.5rem;">
                        <i class="fa-solid fa-circle" style="font-size: 0.45rem; color: #10B981;"></i> LIVE MONITOR
                    </span>
                </div>
                <nav class="flex-gap">
                    <a class="nav-link" href="/home"><i class="fa-solid fa-house"></i> Overview</a>
                    <div style="position: relative; display: inline-block;">
                        <a class="nav-link" href="/realtime/madrid/1"><i class="fa-solid fa-location-dot"></i> Madrid EMT</a>
                    </div>
                    <div style="position: relative; display: inline-block;">
                        <a class="nav-link" href="/realtime/london/25"><i class="fa-solid fa-location-dot"></i> London TfL</a>
                    </div>
                    <a class="nav-link" href="/credits"><i class="fa-solid fa-graduation-cap"></i> Research & Credits</a>
                </nav>
            </div>
        </header>

        <main style="padding: 1.5rem 2rem; max-width: 1600px; margin: 0 auto;">
            {%app_entry%}
        </main>

        <footer style="padding: 2rem; text-align: center; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border-color); margin-top: 3rem;">
            <p>© 2026 Universidad Politécnica de Madrid (UPM) & Cátedra Cabify — IEEE Transactions on Intelligent Transportation Systems</p>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# App Layout Div
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ]
)


# Routing callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    try:
        if pathname in ["/home", "/", None]:
            return app_home.layout
        elif pathname.startswith("/realtime/madrid"):
            line = pathname.split("/")[-1] if len(pathname.split("/")) > 3 else ""
            if line not in ["1", "44", "82", "132", "133", "F", "G"]:
                return html.Div(
                    className="modern-card",
                    style={"textAlign": "center", "padding": "3rem"},
                    children=[
                        html.H2("Madrid Line Not Available", style={"color": "#F87171"}),
                        html.P(f"Line '{line}' is not currently active in the real-time pipeline."),
                        dcc.Link(
                            "← Back to Madrid Line 1",
                            href="/realtime/madrid/1",
                            className="btn-primary-gradient",
                            style={"marginTop": "1rem"},
                        ),
                    ],
                )
            return app_realtime_madrid.layout
        elif pathname.startswith("/realtime/london"):
            line = pathname.split("/")[-1] if len(pathname.split("/")) > 3 else ""
            if line not in ["18", "25"]:
                return html.Div(
                    className="modern-card",
                    style={"textAlign": "center", "padding": "3rem"},
                    children=[
                        html.H2("London Line Not Available", style={"color": "#F87171"}),
                        html.P(f"Line '{line}' is not currently active in the real-time pipeline."),
                        dcc.Link(
                            "← Back to London Line 18",
                            href="/realtime/london/18",
                            className="btn-primary-gradient",
                            style={"marginTop": "1rem"},
                        ),
                    ],
                )
            return app_realtime_london.layout
        elif pathname == "/credits":
            return app_credits.layout
        else:
            return app_home.layout
    except Exception:
        return app_home.layout


server = app.server

if __name__ == "__main__":
    app.run(port=8050, host="0.0.0.0", debug=False)
