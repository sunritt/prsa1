from playwright.sync_api import sync_playwright
import time
import csv
import os
import bs4

def load_existing_reviews(filepath):
    existing = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                body = row.get('Body')
                if body is not None:
                    existing.add(body.strip())
    return existing

def scrape_interactive():
    filepath = "DL101.csv"
    existing = load_existing_reviews(filepath)
    
    with sync_playwright() as p:
        print("Launching Interactive Scraper...", flush=True)
        user_data_dir = os.path.join(os.getcwd(), "amazon_profile")
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        )
        
        page = context.pages[0] if context.pages else context.new_page()
        
        print("\n===========================================")
        print("BROWSER OPENED!")
        print("You have 2 FULL MINUTES to navigate to the exact page you want.")
        print("Get everything set up. I will NOT scroll or interfere during this time.")
        print("After 2 minutes, I will continuously scroll SLOWLY and scrape for exactly 3 MINUTES.")
        print("===========================================\n", flush=True)
        
        # Give the user 2 minutes to navigate
        time.sleep(120)
        
        print("\n*** 2 Minutes Up! Beginning Auto-Scroll and Scrape for 3 Minutes ***\n", flush=True)
        
        start_time = time.time()
        
        while time.time() - start_time < 180:
            try:
                # Scroll down slightly (lower rate so reviews don't get skipped)
                page.mouse.wheel(0, 500)
                time.sleep(2)
                
                # Get current page content without refreshing
                html = page.content()
                soup = bs4.BeautifulSoup(html, 'html.parser')
                
                reviews = soup.find_all(id=lambda i: i and i.startswith('customer_review'))
                new_this_tick = 0
                
                for el in reviews:
                    try:
                        author_el = el.find(class_='a-profile-name')
                        author = author_el.get_text(strip=True) if author_el else "Amazon Customer"
                        
                        rating_el = el.find('i', class_='review-rating') or el.find('span', class_='a-icon-alt')
                        rating = rating_el.get_text(strip=True).split('.')[0] if rating_el else "Unknown"
                        
                        title_el = el.find('a', class_='review-title') or el.find('span', class_='review-title')
                        title = title_el.get_text(strip=True) if title_el else "Unknown"
                        
                        body_el = el.find('span', class_='review-text') or el.find('div', class_='review-text-content')
                        body = body_el.get_text(strip=True) if body_el else "Unknown"
                        
                        if body and body != "Unknown" and body not in existing:
                            existing.add(body)
                            new_this_tick += 1
                            
                            file_exists = os.path.exists(filepath)
                            with open(filepath, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=['Author', 'Rating', 'Title', 'Body'])
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow({
                                    'Author': author,
                                    'Rating': rating,
                                    'Title': title,
                                    'Body': body
                                })
                    except Exception:
                        pass
                
                if new_this_tick > 0:
                    print(f"Scraped {new_this_tick} new reviews! Total in CSV: {len(existing)}", flush=True)
                    
            except Exception as e:
                # Ignore errors like page navigating, just wait and retry
                time.sleep(2)
                
        print("\n*** 3 Minutes Up! Stopping Scraper. ***\n", flush=True)
        context.close()

if __name__ == "__main__":
    scrape_interactive()
