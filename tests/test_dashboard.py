import json
import os
import sys

# Add Dashboard and root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Dashboard"))
sys.path.insert(0, ROOT)

from index import display_page

from app import app
from apps import app_credits, app_home, app_realtime_london, app_realtime_madrid


def test_app_initialization():
    assert app is not None
    assert app.server is not None


def test_display_page_routing():
    # Test home
    home_layout = display_page("/home")
    assert home_layout == app_home.layout

    # Test root
    root_layout = display_page("/")
    assert root_layout == app_home.layout

    # Test credits
    credits_layout = display_page("/credits")
    assert credits_layout == app_credits.layout

    # Test valid madrid
    madrid_layout = display_page("/realtime/madrid/1")
    assert madrid_layout == app_realtime_madrid.layout

    # Test valid london
    london_layout = display_page("/realtime/london/18")
    assert london_layout == app_realtime_london.layout

    # Test invalid line
    invalid_layout = display_page("/realtime/madrid/999")
    assert invalid_layout != app_realtime_madrid.layout


def test_static_data_files():
    # Check Madrid static data
    assert os.path.exists(os.path.join(ROOT, "Madrid", "Data", "Static", "lines_dict.json"))
    with open(os.path.join(ROOT, "Madrid", "Data", "Static", "lines_dict.json")) as f:
        data = json.load(f)
        assert "1" in data

    # Check London static data
    assert os.path.exists(os.path.join(ROOT, "London", "Data", "Static", "lines_dict.json"))
    with open(os.path.join(ROOT, "London", "Data", "Static", "lines_dict.json")) as f:
        data = json.load(f)
        assert "18" in data
