from playwright.sync_api import sync_playwright
import time
import csv
import re
import bs4
import os

def load_existing_reviews(filepath):
    existing = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row.get('Body', '').strip())
    return existing

def parse_html_to_reviews(html):
    soup = bs4.BeautifulSoup(html, 'html.parser')
    all_text = soup.get_text(separator="\n", strip=True)
    lines = all_text.split('\n')
    
    reviews = []
    
    # We look for the author pattern: "Name, Location (X years ago)" or "Name,  (X months ago)"
    author_regex = re.compile(r"^(.*?),\s*(.*?)\s*\(\d+\s*(year|month|day|week)s?\s*ago\)$", re.IGNORECASE)
    
    # Backwards parsing: Find author, then backtrack for body, title, rating
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = author_regex.match(line)
        if match:
            author = match.group(1).strip()
            
            # Backtrack to find rating and text
            # Usually: 
            # Rating
            # Title
            # Body
            # Author
            # But sometimes Title and Body are combined.
            
            body = ""
            title = ""
            rating = ""
            
            # Let's look up to 4 lines above
            for offset in range(1, 5):
                idx = i - offset
                if idx < 0: break
                
                prev_line = lines[idx].strip()
                
                if prev_line in ["1", "2", "3", "4", "5"]:
                    rating = prev_line
                    # Everything between rating and author is text
                    text_lines = lines[idx+1 : i]
                    if len(text_lines) == 1:
                        title = text_lines[0]
                        body = text_lines[0]
                    elif len(text_lines) >= 2:
                        title = text_lines[0]
                        body = " ".join(text_lines[1:])
                    break
            
            if rating and body:
                reviews.append({
                    'Author': author,
                    'Rating': rating,
                    'Title': title,
                    'Body': body
                })
        i += 1
        
    return reviews

def scrape():
    url = "https://www.bigbasket.com/product-reviews/40016465/eveready-carbon-zinc-battery-red-aaa-1012-4-pcs/?page=1"
    filepath = "carbonzinc_aaa.csv"
    existing = load_existing_reviews(filepath)
    
    with sync_playwright() as p:
        print("Launching Headed Browser to bypass Akamai Bot Protection...", flush=True)
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_extra_http_headers({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'})
        
        print(f"Navigating to {url}", flush=True)
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
        
        print("\n*** IMPORTANT ***")
        print("Please solve any captchas if they appear.")
        print("Waiting 15 seconds...\n", flush=True)
        time.sleep(15)
        
        print("Scrolling down to load more reviews...", flush=True)
        # Scroll 10 times to load all reviews
        for _ in range(10):
            page.mouse.wheel(0, 1500)
            time.sleep(1.5)
            
        html = page.content()
        browser.close()
        
    reviews = parse_html_to_reviews(html)
    print(f"Parsed {len(reviews)} reviews from the page.")
    
    new_reviews = []
    for r in reviews:
        if r['Body'] not in existing:
            new_reviews.append(r)
            
    print(f"Found {len(new_reviews)} NEW reviews to append.")
    
    if new_reviews:
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=['Author', 'Rating', 'Title', 'Body'])
            if not file_exists:
                writer.writeheader()
            for r in new_reviews:
                writer.writerow(r)
        print(f"Successfully appended {len(new_reviews)} reviews to {filepath}.")

if __name__ == "__main__":
    scrape()
