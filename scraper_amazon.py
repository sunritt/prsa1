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

def scrape():
    filepath = "DL95.csv"
    urls_file = "urls.txt"
    
    if not os.path.exists(urls_file):
        print(f"Error: {urls_file} not found. Please create it and add your Amazon links.", flush=True)
        return
        
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
        
    if not urls:
        print(f"Error: {urls_file} is empty.", flush=True)
        return
        
    existing = load_existing_reviews(filepath)
    new_reviews = []
    
    with sync_playwright() as p:
        print("Launching Headed Browser with Persistent Login...", flush=True)
        user_data_dir = os.path.join(os.getcwd(), "amazon_profile")
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        )
        page = context.pages[0] if context.pages else context.new_page()
        
        for idx, url in enumerate(urls, 1):
            print(f"Navigating to Link {idx}/{len(urls)}: {url[:60]}...", flush=True)
            
            html = ""
            for retry in range(3):
                try:
                    page.goto(url, wait_until='domcontentloaded')
                    time.sleep(5) # Wait for reviews to load
                    html = page.content()
                    break
                except Exception as e:
                    print(f"Error getting page content: {e}. Retrying...", flush=True)
                    time.sleep(3)
                    
            if not html:
                print(f"Failed to load Link {idx}. Skipping.", flush=True)
                continue
                
            soup = bs4.BeautifulSoup(html, 'html.parser')
            reviews = soup.find_all(id=lambda i: i and i.startswith('customer_review'))
            
            if not reviews:
                print(f"No reviews found on Link {idx}.", flush=True)
                continue
                
            print(f"Found {len(reviews)} reviews on Link {idx}.")
            
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
                    
                    if body and body != "Unknown":
                        new_reviews.append({
                            'Author': author,
                            'Rating': rating,
                            'Title': title,
                            'Body': body
                        })
                except Exception as e:
                    print(f"Error parsing a review: {e}")
                    
        print(f"\nScraping finished. Found {len(new_reviews)} total reviews across all provided links.")
        
        if new_reviews:
            file_exists = os.path.exists(filepath)
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=['Author', 'Rating', 'Title', 'Body'])
                if not file_exists:
                    writer.writeheader()
                for r in new_reviews:
                    writer.writerow(r)
            print(f"Successfully appended {len(new_reviews)} reviews to {filepath}.")
            
        context.close()

if __name__ == "__main__":
    scrape()
