#!/usr/bin/env python3
"""EMT Madrid MobilityLabs Authentication Poller & Watchdog.

Periodically tests credentials against the EMT OpenAPI login endpoint:
- If 403 (Code 84), logs waiting status.
- When 200 (Code 00/01) is returned with an active accessToken, triggers the Madrid collector.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent

try:
    from api_credentials import XClientId, passKey
except ImportError:
    XClientId = ""
    passKey = ""


def check_madrid_auth() -> dict:
    """Test EMT Madrid login endpoint."""
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not XClientId or not passKey:
        return {"status": "unconfigured", "message": "Missing credentials in api_credentials.py"}

    try:
        url = "https://openapi.emtmadrid.es/v2/mobilitylabs/user/login/"
        resp = requests.get(
            url,
            headers={"X-ClientId": XClientId.strip(), "passKey": passKey.strip()},
            timeout=10,
        )
        data = resp.json() if resp.status_code in (200, 403) else {}
        code = str(data.get("code", ""))

        if resp.status_code == 200 and code in ("00", "01"):
            token = data.get("data", [{}])[0].get("accessToken", "")
            print(f"[{now_str}] 🎉 Madrid EMT MobilityLabs activated! Access token acquired.")
            return {"status": "active", "token": token}
        elif code == "84":
            print(
                f"[{now_str}] ⏳ Madrid EMT Status: Pending portal review/activation (Code 84). Will recheck in 2m."
            )
            return {"status": "pending_review", "code": 84}
        else:
            print(
                f"[{now_str}] ⚠️ EMT Response: HTTP {resp.status_code} (Code {code}): {data.get('description', '')}"
            )
            return {"status": "error", "code": code}
    except Exception as e:
        print(f"[{now_str}] Error contacting EMT Madrid: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="EMT Madrid MobilityLabs Watchdog")
    parser.add_argument(
        "--once", action="store_true", help="Perform single authentication test and exit"
    )
    parser.add_argument(
        "--interval", type=int, default=120, help="Polling interval in seconds (default: 120)"
    )
    args = parser.parse_args()

    if args.once:
        check_madrid_auth()
        return

    print(f"Starting EMT Madrid Watchdog (checking every {args.interval}s)...")
    while True:
        res = check_madrid_auth()
        if res.get("status") == "active":
            print("Access granted! Ready to start Madrid live collection.")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
