from playwright.sync_api import sync_playwright
import time
import os

def open_browser():
    with sync_playwright() as p:
        user_data_dir = os.path.join(os.getcwd(), "amazon_profile")
        print(f"Opening persistent profile from: {user_data_dir}", flush=True)
        
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            # Ensure it opens normally and doesn't get instantly detected as a bot while browsing
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        print("Browser is open! You can restore pages, log in, or browse freely.", flush=True)
        print("This window will stay open until you close it manually or 24 hours pass.", flush=True)
        
        # Keep the script alive so the browser stays open
        try:
            time.sleep(86400) # Wait 24 hours
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    open_browser()
