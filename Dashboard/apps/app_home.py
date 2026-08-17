from dash import dcc, html

layout = html.Div(
    children=[
        # Hero Banner
        html.Div(
            className="modern-card no-hover",
            style={
                "background": "linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(79, 70, 229, 0.1) 100%)",
                "borderColor": "rgba(139, 92, 246, 0.3)",
                "marginBottom": "2rem",
                "padding": "2.5rem",
            },
            children=[
                html.Div(
                    className="flex-between flex-wrap",
                    children=[
                        html.Div(
                            style={"maxWidth": "800px"},
                            children=[
                                html.Span(
                                    "IEEE Transactions on Intelligent Transportation Systems",
                                    className="badge-pill primary",
                                    style={"marginBottom": "0.75rem"},
                                ),
                                html.H1(
                                    "Real-Time Bus Headway Analysis & Anomaly Detection",
                                    style={
                                        "fontSize": "2.2rem",
                                        "fontWeight": "700",
                                        "marginTop": "0.5rem",
                                        "marginBottom": "1rem",
                                    },
                                ),
                                html.P(
                                    "Online statistical modeling and unsupervised anomaly detection for urban public transit networks in Madrid (EMT) and London (TfL). Quantifying headway regularity, bus bunching, and service quality in real time.",
                                    style={
                                        "color": "var(--text-muted)",
                                        "fontSize": "1.1rem",
                                        "lineHeight": "1.6",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="flex-gap flex-wrap",
                            style={"marginTop": "1rem"},
                            children=[
                                dcc.Link(
                                    children=[
                                        html.I(
                                            className="fa-solid fa-bolt",
                                            style={"marginRight": "0.5rem"},
                                        ),
                                        "Madrid Line 1",
                                    ],
                                    className="btn-primary-gradient",
                                    href="/realtime/madrid/1",
                                ),
                                dcc.Link(
                                    children=[
                                        html.I(
                                            className="fa-solid fa-city",
                                            style={"marginRight": "0.5rem"},
                                        ),
                                        "London Line 25",
                                    ],
                                    className="route-btn",
                                    href="/realtime/london/25",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # KPI Stats Grid
        html.Div(
            className="grid-4",
            style={"marginBottom": "2rem"},
            children=[
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span(
                                    "Madrid Lines",
                                    style={"color": "var(--text-muted)", "fontSize": "0.85rem"},
                                ),
                                html.I(className="fa-solid fa-route", style={"color": "#8B5CF6"}),
                            ],
                        ),
                        html.H2(
                            "7 Monitored", style={"fontSize": "1.6rem", "margin": "0.5rem 0 0.2rem"}
                        ),
                        html.Span(
                            "EMT Madrid Network",
                            style={"fontSize": "0.8rem", "color": "var(--text-muted)"},
                        ),
                    ],
                ),
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span(
                                    "London Corridors",
                                    style={"color": "var(--text-muted)", "fontSize": "0.85rem"},
                                ),
                                html.I(
                                    className="fa-solid fa-train-subway", style={"color": "#06B6D4"}
                                ),
                            ],
                        ),
                        html.H2(
                            "2 Active", style={"fontSize": "1.6rem", "margin": "0.5rem 0 0.2rem"}
                        ),
                        html.Span(
                            "TfL Unified Open Data",
                            style={"fontSize": "0.8rem", "color": "var(--text-muted)"},
                        ),
                    ],
                ),
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span(
                                    "Statistical Model",
                                    style={"color": "var(--text-muted)", "fontSize": "0.85rem"},
                                ),
                                html.I(
                                    className="fa-solid fa-chart-line", style={"color": "#10B981"}
                                ),
                            ],
                        ),
                        html.H2(
                            "Mahalanobis", style={"fontSize": "1.6rem", "margin": "0.5rem 0 0.2rem"}
                        ),
                        html.Span(
                            "Unsupervised Multivariate",
                            style={"fontSize": "0.8rem", "color": "#10B981"},
                        ),
                    ],
                ),
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            children=[
                                html.Span(
                                    "Service Metrics",
                                    style={"color": "var(--text-muted)", "fontSize": "0.85rem"},
                                ),
                                html.I(
                                    className="fa-solid fa-gauge-high", style={"color": "#F59E0B"}
                                ),
                            ],
                        ),
                        html.H2(
                            "Real-Time QoS",
                            style={"fontSize": "1.6rem", "margin": "0.5rem 0 0.2rem"},
                        ),
                        html.Span(
                            "Bunching & Gaps Detection",
                            style={"fontSize": "0.8rem", "color": "var(--text-muted)"},
                        ),
                    ],
                ),
            ],
        ),
        # City Route Selectors
        html.Div(
            className="grid-2",
            style={"marginBottom": "2rem"},
            children=[
                # Madrid Panel
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "1.25rem"},
                            children=[
                                html.H3(
                                    "Madrid EMT Lines", style={"fontSize": "1.3rem", "margin": 0}
                                ),
                                html.Span("Spain", className="badge-pill primary"),
                            ],
                        ),
                        html.P(
                            "Select an EMT Madrid bus line to inspect live vehicle telemetry, spatial distributions along routes, and multi-dimensional headway anomalies.",
                            style={
                                "color": "var(--text-muted)",
                                "fontSize": "0.9rem",
                                "marginBottom": "1.5rem",
                            },
                        ),
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 1"],
                                    className="route-btn",
                                    href="/realtime/madrid/1",
                                ),
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 44"],
                                    className="route-btn",
                                    href="/realtime/madrid/44",
                                ),
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 82"],
                                    className="route-btn",
                                    href="/realtime/madrid/82",
                                ),
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 132"],
                                    className="route-btn",
                                    href="/realtime/madrid/132",
                                ),
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 133"],
                                    className="route-btn",
                                    href="/realtime/madrid/133",
                                ),
                            ],
                        ),
                    ],
                ),
                # London Panel
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "1.25rem"},
                            children=[
                                html.H3(
                                    "London TfL Lines", style={"fontSize": "1.3rem", "margin": 0}
                                ),
                                html.Span("United Kingdom", className="badge-pill primary"),
                            ],
                        ),
                        html.P(
                            "Select a London bus corridor to evaluate real-time headway behavior, stop arrival estimates, and confidence ellipses over historical distributions.",
                            style={
                                "color": "var(--text-muted)",
                                "fontSize": "0.9rem",
                                "marginBottom": "1.5rem",
                            },
                        ),
                        html.Div(
                            className="flex-gap flex-wrap",
                            children=[
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 18"],
                                    className="route-btn",
                                    href="/realtime/london/18",
                                ),
                                dcc.Link(
                                    [html.I(className="fa-solid fa-bus"), "Line 25"],
                                    className="route-btn",
                                    href="/realtime/london/25",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)
