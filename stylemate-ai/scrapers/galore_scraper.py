#!/usr/bin/env python3
# galore_scraper.py

import requests
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
BASE_URL       = "https://galoreyyz.com"
COLLECTION_URL = urljoin(BASE_URL, "/collections/shop-all")
OUTPUT_FILE    = "galore_products.json"

# ───────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
def get_soup(url: str) -> BeautifulSoup:
    """
    Fetch a URL over HTTP(S) and return a BeautifulSoup object
    with the HTML parsed.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_price(card_content: BeautifulSoup) -> str:
    """
    Given a <div class="card__content"> block, pull out the price text.
    Prioritize .price-item--sale if present, otherwise use .price-item--regular.
    Returns a bare string like "$55.00" (or None if not found).
    """
    # Look for a “sale” price first:
    sale_tag = card_content.select_one(".price-item--sale")
    if sale_tag and sale_tag.get_text(strip=True):
        return sale_tag.get_text(strip=True)

    # Otherwise look for a “regular” price:
    reg_tag = card_content.select_one(".price-item--regular")
    if reg_tag and reg_tag.get_text(strip=True):
        return reg_tag.get_text(strip=True)

    return None


def extract_image_url(card_div: BeautifulSoup) -> str:
    """
    Given the parent <div class="card"> (which wraps each product card),
    find the first <img> and return its src (resolve to absolute if needed).
    If it’s a lazy‐loaded image, we check for srcset/data-srcset too.
    """
    img_tag = card_div.find("img")
    if not img_tag:
        return ""

    # If the <img> has a “srcset” or “data-srcset”, pick the highest-resolution
    # entry after the last comma:
    srcset = img_tag.get("data-srcset") or img_tag.get("srcset")
    if srcset:
        # e.g. "https://…/small.jpg 200w, https://…/medium.jpg 400w, https://…/large.jpg 800w"
        last_piece = srcset.split(",")[-1].strip().split(" ")[0]
        if last_piece.startswith("http"):
            return last_piece
        else:
            return urljoin(BASE_URL, last_piece)

    # Fallback to plain "src":
    src = img_tag.get("src") or ""
    if not src:
        return ""
    if src.startswith("http"):
        return src
    return urljoin(BASE_URL, src)

# ───────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ───────────────────────────────────────────────────────────────────────────────
def scrape() -> list:
    """
    Scrape all products from the Galore YYZ “Shop All” collection page.
    Returns a list of dicts, each containing:
      {
        "title":  str,
        "price":  str or None,
        "url":    str,
        "image_url": str,
        "tags":   [],          # (no tags on Galore’s front page)
        "sizes":  []           # (no sizes on Galore’s front page)
      }
    """
    soup = get_soup(COLLECTION_URL)
    products = []
    seen_urls = set()

    # Each product’s info block sits under <div class="card__content">.
    for card_content in soup.select("div.card__content"):
        try:
            # Walk up to the <div class="card"> wrapper to find the image.
            card_div = card_content.find_parent("div", class_="card")
            if not card_div:
                # If for some reason we can't find a parent with class "card", skip
                continue

            # ─── Extract title & URL ─────────────────────────────────────────
            title_link = card_content.select_one("h3.card__heading a.full-unstyled-link")
            if not title_link:
                logger.debug("Skipping a card because no <h3.card__heading a> found")
                continue

            title = title_link.get_text(strip=True)
            rel_url = title_link.get("href", "").strip()
            if not rel_url:
                logger.debug(f"Skipping {title!r} because href is missing.")
                continue

            product_url = urljoin(BASE_URL, rel_url)
            if product_url in seen_urls:
                continue
            seen_urls.add(product_url)

            # ─── Extract price ──────────────────────────────────────────────
            price_text = extract_price(card_content)

            # ─── Extract image URL ──────────────────────────────────────────
            image_url = extract_image_url(card_div)
            if not image_url:
                logger.warning(f"No image found for product: {title}")

            # Tags and sizes are not exposed in Galore’s “Shop All” page,
            # so we simply leave them empty arrays.
            products.append({
                "title": title,
                "price": price_text,
                "url": product_url,
                "image_url": image_url,
                "tags": [],
                "sizes": []
            })
        except Exception as e:
            logger.error(f"Error parsing a Galore card: {e}")

    return products


if __name__ == "__main__":
    data = scrape()
    logger.info(f"Scraped {len(data)} products from Galore YYZ.")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Output saved to {OUTPUT_FILE}")
