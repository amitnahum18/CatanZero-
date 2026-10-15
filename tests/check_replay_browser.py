"""Manual browser smoke check; uses an existing host Playwright and Edge."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    target = sys.argv[1]
    source = Path(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]).resolve()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            headless=True,
        )
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(target if target.startswith("http://") else source.as_uri())
        assert page.locator("#tiles polygon").count() == 19
        assert page.locator(".player").count() == 4
        page.locator("#next").click()
        assert page.locator("#buildings path").count() == 1
        page.locator("#end").click()
        assert page.locator("#winner").inner_text() == "כחול ניצח"
        assert page.locator("#progress").inner_text() == "1102 / 1102"
        page.screenshot(path=str(source.with_name("replay-final.png")), full_page=True)
        page.locator("#start").click()
        page.locator("#play").click()
        page.wait_for_function("Number(document.getElementById('scrubber').value) >= 3")
        page.locator("#play").click()
        assert page.locator("#play").inner_text() == "▶ ניגון"
        page.screenshot(path=str(source.with_name("replay-preview.png")), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert errors == [], errors
        browser.close()
    print("Browser check passed: board, step, winner, playback, mobile layout; no JavaScript errors")


if __name__ == "__main__":
    main()
