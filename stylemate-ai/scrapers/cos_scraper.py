from playwright.sync_api import sync_playwright
import time


def scrape():
    products = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        url = "https://www.cos.com/en-ww/men/view-all"
        print(f"Scraping: {url}")
        page.goto(url, timeout=60000)

        # Scroll to load everything
        prev_height = 0
        while True:
            page.mouse.wheel(0, 3000)
            time.sleep(1)
            curr_height = page.evaluate("document.body.scrollHeight")
            if curr_height == prev_height:
                break
            prev_height = curr_height

        items = page.query_selector_all("div.group.bg-main")  # container for each product

        for item in items:
            try:
                # Title from <img alt="">
                img_tag = item.query_selector("img")
                title = img_tag.get_attribute("alt").strip() if img_tag else None

                # URL from <a href=...>
                a_tag = item.query_selector("a")
                url = a_tag.get_attribute("href") if a_tag else None
                if url and not url.startswith("http"):
                    url = "https://www.cos.com" + url

                # Price from visible span or text element
                price_el = item.query_selector("div[class*=pt-1\\.25]")
                price = price_el.inner_text().strip() if price_el else None

                # Image: get from srcset (grab highest resolution)
                srcset = img_tag.get_attribute("srcset") if img_tag else None
                image = None
                if srcset:
                    image = srcset.split(",")[-1].split(" ")[0].strip()

                # Color swatch is inside a <span> with inline style (not always present)
                color = None
                swatch_el = item.query_selector("div[class*=bg-main-product-card-image] [style*=background-color]")
                if swatch_el:
                    style = swatch_el.get_attribute("style")
                    if "background-color:" in style:
                        color = style.split("background-color:")[1].split(";")[0].strip()

                if title and image and url:
                    products.append({
                        "title": title,
                        "url": url,
                        "image": image,
                        "price": price,
                        "color": color
                    })

            except Exception as e:
                print(f"❌ Skipping item due to error: {e}")

        browser.close()
    return products
