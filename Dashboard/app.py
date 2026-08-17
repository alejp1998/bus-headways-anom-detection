import dash
import plotly.io as pio

# Global dark theme to match the dashboard design system
pio.templates.default = "plotly_dark"

# We setup the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Bus Headways | Real-Time Anomaly Detection"
server = app.server
