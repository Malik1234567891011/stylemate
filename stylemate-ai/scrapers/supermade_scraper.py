#!/usr/bin/env python3

import requests
import json
import logging
from urllib.parse import urljoin

BASE_URL = "https://www.thesupermade.com"
COLLECTION_HANDLE = "best-seller"
OUTPUT_FILE = "supermade_products.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def scrape() -> list:
    products = []
    page = 1

    while True:
        url = f"{BASE_URL}/collections/{COLLECTION_HANDLE}/products.json?page={page}"
        logger.info(f"Fetching page {page}: {url}")

        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            logger.warning(f"Failed to fetch page {page}")
            break

        data = resp.json()
        items = data.get("products", [])
        if not items:
            break

        for item in items:
            title = item.get("title")
            handle = item.get("handle")
            image_url = item.get("images", [{}])[0].get("src")
            price = item.get("variants", [{}])[0].get("price")
            product_url = urljoin(BASE_URL, f"/products/{handle}")

            if title and price and image_url:
                products.append({
                    "title": title,
                    "price": f"${price}",
                    "url": product_url,
                    "image_url": image_url,
                    "tags": item.get("tags", []),
                    "sizes": [v.get("title") for v in item.get("variants", [])]
                })

        page += 1
        if page > 24:  # safety cap
            break

    return products


if __name__ == "__main__":
    data = scrape()
    logger.info(f"Scraped {len(data)} products.")
    print(f"Scraped {len(data)} products")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Output saved to {OUTPUT_FILE}")
