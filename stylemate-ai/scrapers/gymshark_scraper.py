import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def extract_products_from_page(soup):
    products = []
    for article in soup.find_all("article"):
        try:
            title_tag = article.find("a", class_="product-card_product-title-link__nDARd")
            if not title_tag:
                continue
            title = title_tag.text.strip()
            href = title_tag["href"]
            product_url = href if href.startswith("http") else "https://www.gymshark.com" + href

            # IMAGE URL from <source data-srcset="...">
            image_url = None
            source_tag = article.find("source", attrs={"data-srcset": True})
            if source_tag:
                srcset = source_tag["data-srcset"]
                image_url = srcset.split(",")[0].split(" ")[0]  # First URL from srcset

            # fallback to <img>
            if not image_url:
                img_tag = article.find("img")
                if img_tag and img_tag.get("src"):
                    image_url = img_tag["src"]

            price_tag = article.find("span", class_="product-card_product-price__vFL1l")
            price = price_tag.text.strip() if price_tag else None

            fit_tag = article.find("p", class_="product-card_product-fit__Pe02x")
            fit = fit_tag.text.strip() if fit_tag else None

            color_tag = article.find("p", class_="product-card_product-colour__8R7vq")
            color = color_tag.text.strip() if color_tag else None

            if image_url:
                products.append({
                    "title": title,
                    "url": product_url,
                    "image": image_url,
                    "price": price,
                    "fit": fit,
                    "color": color
                })

        except Exception as e:
            print(f"Skipping product due to error: {e}")
    return products

def extract_products(base_url):
    all_products = []
    page = 1
    MAX_PAGES = 20  # Safety fallback to prevent infinite loops

    while page <= MAX_PAGES:
        paged_url = f"{base_url}?page={page}"
        print(f"Scraping: {paged_url}")
        response = requests.get(paged_url, headers=HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")

        page_products = extract_products_from_page(soup)
        if not page_products:
            print(f"No products found on page {page}. Stopping pagination.")
            break

        all_products.extend(page_products)
        page += 1

    return all_products

def scrape():
    women_url = "https://www.gymshark.com/collections/all-products/womens"
    men_url = "https://www.gymshark.com/collections/all-products/mens"
    return extract_products(women_url) + extract_products(men_url)
