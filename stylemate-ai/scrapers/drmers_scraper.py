import requests
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin, unquote

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
BASE_URL        = "https://drmersclub.com"
COLLECTION_PATH = "/collections/shop-all"
COLLECTION_URL  = urljoin(BASE_URL, COLLECTION_PATH)

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
    """Fetch URL and return BeautifulSoup parser."""
    resp = requests.get(url)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def pick_image_url(card: BeautifulSoup) -> str:
    """Select the highest-res image from primary, secondary, or noscript tags."""
    # primary image
    img = card.select_one("img.grid-product__image")
    if img:
        srcset = img.get("data-srcset") or img.get("srcset")
        if srcset:
            last = srcset.split(',')[-1].strip().split(' ')[0]
            return last if last.startswith('http') else f'https:{last}'
        src = img.get('src')
        if src:
            return src if src.startswith('http') else urljoin(BASE_URL, src)
    # secondary image
    img2 = card.select_one(".grid-product__secondary-image img")
    if img2:
        srcset = img2.get("data-srcset") or img2.get("srcset")
        if srcset:
            last = srcset.split(',')[-1].strip().split(' ')[0]
            return last if last.startswith('http') else f'https:{last}'
        src = img2.get('src')
        if src:
            return src if src.startswith('http') else urljoin(BASE_URL, src)
    # noscript fallback
    nos = card.select_one('noscript img')
    if nos and nos.get('src'):
        src = nos['src']
        return src if src.startswith('http') else urljoin(BASE_URL, src)
    return ""


def parse_product_data(card: BeautifulSoup) -> tuple:
    """Extract tags and variant sizes from embedded JSON."""
    # match banana-container by presence of data-product-data attribute
    bc = card.select_one(".banana-container[data-product-data]")
    if not bc:
        return [], []
    try:
        raw = unquote(bc['data-product-data'])
        obj = json.loads(raw)
        tags = obj.get('tags', [])
        sizes = [
            { 'size': v.get('name'), 'in_stock': v.get('in_stock', False) }
            for v in obj.get('variants', [])
        ]
        return tags, sizes
    except Exception as e:
        logger.error(f"JSON parse error for product-data: {e}")
        return [], []

# ───────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ───────────────────────────────────────────────────────────────────────────────
def scrape_drmers() -> list:
    """Scrape all products from the collection page."""
    soup = get_soup(COLLECTION_URL)
    products = []
    seen_urls = set()

    # only divs with product-id attribute
    for card in soup.select('div.grid__item[data-product-id]'):
        try:
            title_el = card.select_one('.grid-product__title')
            price_el = card.select_one('.grid-product__price .money')
            link_el  = card.select_one('a.grid-product__link')

            title = title_el.get_text(strip=True) if title_el else None
            price = price_el.get_text(strip=True) if price_el else None
            rel   = link_el['href'] if link_el and link_el.has_attr('href') else None
            url   = urljoin(BASE_URL, rel) if rel else None

            if not (title and price and url):
                logger.debug(f"Skipping incomplete: title={title}, price={price}, url={url}")
                continue
            if url in seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue
            seen_urls.add(url)

            image_url = pick_image_url(card)
            if not image_url:
                logger.warning(f"No image for: {title}")

            tags, sizes = parse_product_data(card)

            products.append({
                'title': title,
                'price': price,
                'url': url,
                'image_url': image_url,
                'tags': tags,
                'sizes': sizes
            })

        except Exception as e:
            logger.error(f"Error parsing card: {e}")

    return products

if __name__ == '__main__':
    data = scrape_drmers()
    logger.info(f"Scraped {len(data)} products.")

    print(json.dumps(data, indent=2, ensure_ascii=False))
    with open('drmers_products.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Output saved to drmers_products.json")
