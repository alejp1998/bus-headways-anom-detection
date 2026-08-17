from dash import html

layout = html.Div(
    children=[
        html.Div(
            className="modern-card no-hover",
            style={
                "background": "linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(79, 70, 229, 0.1) 100%)",
                "borderColor": "rgba(139, 92, 246, 0.3)",
                "marginBottom": "2rem",
                "padding": "2rem 2.5rem",
            },
            children=[
                html.Span(
                    "Research & Credits",
                    className="badge-pill primary",
                    style={"marginBottom": "0.5rem"},
                ),
                html.H1(
                    "Publication & Academic Context",
                    style={"fontSize": "2rem", "marginTop": "0.5rem"},
                ),
                html.P(
                    "This research develops a data processing and unsupervised statistical anomaly detection system for bus headways in Madrid and London.",
                    style={"color": "var(--text-muted)", "fontSize": "1.05rem"},
                ),
            ],
        ),
        html.Div(
            className="grid-2",
            style={"marginBottom": "2rem"},
            children=[
                # Publication Card
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "1rem"},
                            children=[
                                html.H3(
                                    "IEEE TITS Journal Paper",
                                    style={"fontSize": "1.25rem", "margin": 0},
                                ),
                                html.Span("Published", className="badge-pill success"),
                            ],
                        ),
                        html.P(
                            [
                                html.Strong(
                                    "A. Jarabo-Peñas, P. J. Zufiria and C. García-Mauriño, "
                                ),
                                html.Em('"Bus Headways Analysis for Anomaly Detection," '),
                                "in ",
                                html.Strong(
                                    "IEEE Transactions on Intelligent Transportation Systems"
                                ),
                                ", vol. 23, no. 10, pp. 18975-18988, Oct. 2022.",
                            ],
                            style={
                                "fontSize": "0.95rem",
                                "lineHeight": "1.6",
                                "color": "var(--text-main)",
                            },
                        ),
                        html.Div(
                            style={"marginTop": "1.5rem"},
                            children=[
                                html.A(
                                    [
                                        html.I(
                                            className="fa-solid fa-arrow-up-right-from-square",
                                            style={"marginRight": "0.5rem"},
                                        ),
                                        "View on IEEE Xplore",
                                    ],
                                    href="https://doi.org/10.1109/TITS.2022.3155180",
                                    target="_blank",
                                    className="btn-primary-gradient",
                                ),
                            ],
                        ),
                    ],
                ),
                # Academic Affiliations
                html.Div(
                    className="modern-card",
                    children=[
                        html.Div(
                            className="flex-between",
                            style={"marginBottom": "1rem"},
                            children=[
                                html.H3(
                                    "Academic Institutions",
                                    style={"fontSize": "1.25rem", "margin": 0},
                                ),
                                html.Span("Affiliations", className="badge-pill primary"),
                            ],
                        ),
                        html.Ul(
                            style={
                                "listStyle": "none",
                                "padding": 0,
                                "margin": 0,
                                "lineHeight": "2",
                            },
                            children=[
                                html.Li(
                                    [
                                        html.I(
                                            className="fa-solid fa-building-columns",
                                            style={"color": "#8B5CF6", "marginRight": "0.75rem"},
                                        ),
                                        "Universidad Politécnica de Madrid (UPM)",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.I(
                                            className="fa-solid fa-car",
                                            style={"color": "#06B6D4", "marginRight": "0.75rem"},
                                        ),
                                        "Cátedra Cabify - ETSIT UPM",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.I(
                                            className="fa-solid fa-flask",
                                            style={"color": "#10B981", "marginRight": "0.75rem"},
                                        ),
                                        "Information Processing and Telecommunications Center (IPTC)",
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)
