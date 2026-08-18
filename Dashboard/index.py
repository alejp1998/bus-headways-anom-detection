import os
import sys

from dash import Input, Output, dcc, html

# Ensure Dashboard directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from apps import app_credits, app_history, app_home, app_realtime_london, app_realtime_madrid

# Custom HTML index template with theme script & typography
app.index_string = """
<!DOCTYPE html>
<html lang="en">
    <head>
        {%metas%}
        <title>Bus Headways | Real-Time Anomaly Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
        <link rel="alternate icon" type="image/x-icon" href="/assets/favicon.ico">
        <link rel="apple-touch-icon" href="/assets/favicon.png">
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
                    <span class="badge-pill primary" style="font-size: 0.72rem; padding: 0.25rem 0.6rem;">
                        <span class="pulse-indicator"></span> LIVE
                    </span>
                </div>
                <div class="flex-gap">
                    <nav class="flex-gap">
                        <a class="nav-link" href="/home"><i class="fa-solid fa-house"></i> Overview</a>
                        <a class="nav-link" href="/realtime/madrid/1"><i class="fa-solid fa-location-dot"></i> Madrid EMT</a>
                        <a class="nav-link" href="/realtime/london/25"><i class="fa-solid fa-location-dot"></i> London TfL</a>
                        <a class="nav-link" href="/history"><i class="fa-solid fa-clock-rotate-left"></i> History & Models</a>
                        <a class="nav-link" href="/credits"><i class="fa-solid fa-graduation-cap"></i> Credits</a>
                    </nav>
                    <button class="theme-btn" id="open-guide-btn" onclick="openGuideModal()" title="Help & System Guide" style="padding: 0.35rem 0.75rem; gap: 0.35rem; display: inline-flex; align-items: center; font-size: 0.82rem; font-weight: 600;">
                        <i class="fa-solid fa-circle-question" style="color: var(--primary-color);"></i> Guide
                    </button>
                    <div class="theme-switch-container">
                        <button class="theme-btn" data-set-theme="light" onclick="setDashboardTheme('light')" title="Light Theme">
                            <i class="fa-solid fa-sun"></i>
                        </button>
                        <button class="theme-btn" data-set-theme="dark" onclick="setDashboardTheme('dark')" title="Dark Theme">
                            <i class="fa-solid fa-moon"></i>
                        </button>
                        <button class="theme-btn active" data-set-theme="system" onclick="setDashboardTheme('system')" title="System Theme">
                            <i class="fa-solid fa-desktop"></i> Auto
                        </button>
                    </div>
                </div>
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

        <!-- Help & System Guide Modal -->
        <div id="guide-modal-backdrop" class="guide-modal-backdrop" style="display: none;" onclick="if(event.target === this) closeGuideModal()">
            <div class="guide-modal-card">
                <div class="guide-modal-header">
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <div class="guide-icon-badge"><i class="fa-solid fa-book-open"></i></div>
                        <div>
                            <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">Transit Intelligence Guide</h2>
                            <p style="margin: 0; font-size: 0.78rem; color: var(--text-muted);">How Real-Time Bus Headways & Anomaly Detection Work</p>
                        </div>
                    </div>
                    <button class="guide-close-btn" onclick="closeGuideModal()" title="Close (Esc)">&times;</button>
                </div>

                <div class="guide-modal-body">
                    <!-- Section 1 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-bus" style="color: var(--primary-color);"></i> 1. What is a "Headway"?</div>
                        <p>In public transit, <strong>headway</strong> is the time interval (in minutes or seconds) between two consecutive buses arriving at the same stop along a route. When buses are evenly spaced, passenger wait times are minimized and transit capacity is maximized.</p>
                    </div>

                    <!-- Section 2 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-triangle-exclamation" style="color: var(--danger-color);"></i> 2. What is "Bus Bunching"?</div>
                        <p>When a lead bus gets delayed by traffic or boarding passengers, it falls behind. The trailing bus catches up because fewer passengers are waiting for it, eventually running right behind the lead bus like a twin. This is called <strong>bus bunching</strong>. The corridor highlights these in <span style="color: var(--danger-color); font-weight: 600;">red warning brackets (gap &lt; 2 min)</span>.</p>
                    </div>

                    <!-- Section 3 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-route" style="color: var(--accent-color);"></i> 3. Reading the Headway Corridor</div>
                        <p>The corridor diagram is a real-time linear stringline map of the entire bus line:</p>
                        <ul>
                            <li><strong>Direction 1 (Top) &amp; Direction 2 (Bottom):</strong> Show buses currently traveling between route termini.</li>
                            <li><strong>Colored Circles:</strong> Individual buses reporting live GPS telemetry. Each bus has a unique matching color on the spatial map and time series plots.</li>
                            <li><strong>Spacing Bridges:</strong> The colored lines connecting consecutive buses show the exact spacing (e.g. <code>5.4 min</code>).</li>
                        </ul>
                    </div>

                    <!-- Section 4 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-chart-line" style="color: #10B981;"></i> 4. 1D vs 2D Headway Dynamics</div>
                        <ul>
                            <li><strong>1D Headway Series:</strong> Plots the time evolution of headways between pairs of consecutive buses (<em>Bus A &rarr; Bus B</em>). The red dashed lines mark nominal tolerance thresholds.</li>
                            <li><strong>2D Dynamics (Phase Space):</strong> Tracks triplets of consecutive buses (<em>Bus A &rarr; Bus B &rarr; Bus C</em>) simultaneously. The <strong>red dashed confidence ellipse</strong> represents nominal operation. When a trajectory drifts outside the ellipse, a multi-bus regularity disturbance is occurring.</li>
                        </ul>
                    </div>

                    <!-- Section 5 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-brain" style="color: #8B5CF6;"></i> 5. What is the Mahalanobis Distance?</div>
                        <p>Rather than using fixed thresholds, our AI engine learns nominal Gaussian distributions (&mu;, &Sigma;) for every hour of the day and day-type (weekday, Saturday, Sunday). The <strong>Mahalanobis distance</strong> measures how many statistical standard deviations (&sigma;) a live headway deviates from expected traffic patterns, accounting for correlation between consecutive vehicles.</p>
                    </div>

                    <!-- Section 6 -->
                    <div class="guide-section">
                        <div class="guide-section-title"><i class="fa-solid fa-sliders" style="color: #F59E0B;"></i> 6. How to Use the Controls</div>
                        <ul>
                            <li><strong>Confidence (1 - &alpha;):</strong> Adjusts the statistical alarm sensitivity. 98% (default) flags only the 2% most extreme deviations as anomalies.</li>
                            <li><strong>Filter Window (k):</strong> Requires an anomaly to persist for <em>k</em> consecutive collection cycles before triggering an alert, preventing false alarms caused by temporary red lights or dwell spikes.</li>
                        </ul>
                    </div>
                </div>

                <div class="guide-modal-footer">
                    <span style="font-size: 0.76rem; color: var(--text-muted);"><i class="fa-solid fa-shield-halved"></i> Autonomous IEEE TITS Real-Time Pipeline</span>
                    <button class="btn-primary-gradient" style="padding: 0.35rem 1rem; font-size: 0.82rem;" onclick="closeGuideModal()">Got it</button>
                </div>
            </div>
        </div>

    </body>
</html>
"""

# App Layout Div
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="theme-store", storage_type="memory"),
        dcc.Interval(id="theme-poll", interval=800, n_intervals=0),
        html.Div(id="page-content"),
    ]
)


# Clientside callback: sync the active theme from the DOM into the Dash store
app.clientside_callback(
    """
    function(n) {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
    """,
    Output("theme-store", "data"),
    Input("theme-poll", "n_intervals"),
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
                        html.H2(
                            "Madrid Line Not Available", style={"color": "var(--danger-color)"}
                        ),
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
            if line not in ["18", "24", "25", "73"]:
                return html.Div(
                    className="modern-card",
                    style={"textAlign": "center", "padding": "3rem"},
                    children=[
                        html.H2(
                            "London Line Not Available", style={"color": "var(--danger-color)"}
                        ),
                        html.P(f"Line '{line}' is not currently active in the real-time pipeline."),
                        dcc.Link(
                            "← Back to London Line 25",
                            href="/realtime/london/25",
                            className="btn-primary-gradient",
                            style={"marginTop": "1rem"},
                        ),
                    ],
                )
            return app_realtime_london.layout
        elif pathname == "/history":
            return app_history.layout
        elif pathname == "/credits":
            return app_credits.layout
        else:
            return app_home.layout
    except Exception:
        return app_home.layout


server = app.server

if __name__ == "__main__":
    app.run(port=8050, host="0.0.0.0", debug=False)
