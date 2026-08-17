"""End-to-End Automated Browser & UI Verification using Playwright."""

import time

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8050"


def test_dashboard_full_flow():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        # 1. Test Homepage
        print("Testing Homepage...")
        page.goto(f"{BASE_URL}/home", wait_until="domcontentloaded")
        time.sleep(1)
        assert page.locator(".brand-logo").is_visible()
        page.screenshot(path="/tmp/pw_home.png")
        print("  ✅ Homepage renders cleanly.")

        # 2. Test London Line 24 (Live Data)
        print("Testing London Line 24 (Live Route)...")
        page.goto(f"{BASE_URL}/realtime/london/24", wait_until="domcontentloaded")
        time.sleep(3)

        # Check route pills
        active_pill = page.locator(".route-btn.active").inner_text()
        print(f"  Active route pill: {active_pill}")
        assert "24" in active_pill

        # Check KPIs
        fleet_val = page.locator("#kpi-fleetLondon").inner_text()
        print(f"  Live fleet KPI: {fleet_val}")
        assert fleet_val != "—" and fleet_val != "0"

        # Check flat headway graph has rendered svg traces
        flat_graph = page.locator("#flat-hwsLondon")
        assert flat_graph.is_visible()
        page.screenshot(path="/tmp/pw_london24.png")
        print("  ✅ London Line 24 live data verified in browser.")

        # 3. Test London Line 73 (Switching routes)
        print("Testing London Line 73...")
        page.goto(f"{BASE_URL}/realtime/london/73", wait_until="domcontentloaded")
        time.sleep(3)
        active_pill_73 = page.locator(".route-btn.active").inner_text()
        print(f"  Active route pill for 73: {active_pill_73}")
        assert "73" in active_pill_73
        page.screenshot(path="/tmp/pw_london73.png")
        print("  ✅ London Line 73 route switching verified.")

        # 4. Test Madrid Line 1
        print("Testing Madrid Line 1...")
        page.goto(f"{BASE_URL}/realtime/madrid/1", wait_until="domcontentloaded")
        time.sleep(3)
        active_pill_m = page.locator(".route-btn.active").inner_text()
        print(f"  Active Madrid route pill: {active_pill_m}")
        assert "1" in active_pill_m
        page.screenshot(path="/tmp/pw_madrid1.png")
        print("  ✅ Madrid Line 1 verified.")

        # 5. Test History Page
        print("Testing History Page...")
        page.goto(f"{BASE_URL}/history", wait_until="domcontentloaded")
        time.sleep(2)
        assert page.locator("#hist-kpi-records").is_visible()
        page.screenshot(path="/tmp/pw_history.png")
        print("  ✅ History Page verified.")

        browser.close()
        print("\n🎉 ALL PLAYWRIGHT BROWSER VERIFICATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_dashboard_full_flow()
