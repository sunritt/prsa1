import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import re

async def scrape_flipkart_reviews(url, output_file='reviews.csv'):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        print(f"Navigating to base url...")
        
        reviews_data = []
        page_num = 1
        
        while True:
            current_url = f"{url}&page={page_num}"
            print(f"Scraping Flipkart page {page_num}...")
            await page.goto(current_url, wait_until='domcontentloaded')
            
            # Wait a bit to simulate human reading and allow dynamic content
            await page.wait_for_timeout(4000)
            
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all elements that indicate a review, like 'Verified Purchase' text
            verified_elements = soup.find_all(string=re.compile(r'Verified Purchase', re.IGNORECASE))
            
            if not verified_elements:
                print("No more reviews found on this page. Stopping.")
                await page.screenshot(path=f"debug_flipkart_page_{page_num}.png")
                break
        
            for verified_text in verified_elements:
                try:
                    # Find the parent that contains the whole review
                    review_container = verified_text.parent
                    for _ in range(7):  # Go up enough levels to capture the whole review block
                        if review_container and review_container.parent:
                            review_container = review_container.parent
                    
                    # Rating - look for text like 'X.X' or 'X' in a div near the top of the container
                    # Often it's in a div with a green or orange color, but we can just use regex on all divs
                    rating = "N/A"
                    for div in review_container.find_all('div'):
                        if re.match(r'^[1-5](\.[0-9])?$', div.text.strip()):
                            rating = div.text.strip() + " out of 5 stars"
                            break
                    
                    # Title
                    # Heuristic: the title is usually next to the rating, or it's the text immediately after it
                    title = "N/A"
                    for div in review_container.find_all('div'):
                        if div.text.strip() and not re.match(r'^[1-5](\.[0-9])?$', div.text.strip()) and len(div.text.strip()) < 50:
                            # It's a short text, might be the title
                            title = div.text.strip()
                            # We break early, hoping the first non-rating short text is the title
                            if "Review for" not in title and title != "Verified Purchase":
                                break
                    
                    # Body
                    body_spans = review_container.find_all('span')
                    body = "N/A"
                    for span in body_spans:
                        if len(span.text) > len(body) or body == "N/A":
                            body = span.text.strip()
                    if body == "N/A":
                        # Sometimes body is in a div
                        for div in review_container.find_all('div'):
                            if len(div.text) > 30 and "Review for" not in div.text:
                                body = div.text.strip()
                                break
                                
                    # Author
                    author = "N/A"
                    
                    # The author and location are usually in adjacent divs inside a flex-row
                    # We can find the location div which starts with a comma, e.g. ", Durgapur"
                    location_div = None
                    for div in review_container.find_all('div', dir='auto'):
                        if div.text.strip().startswith(','):
                            location_div = div
                            break
                            
                    if location_div:
                        author_div = location_div.find_previous_sibling('div')
                        if author_div:
                            author = author_div.text.strip() + location_div.text.strip()
                    
                    # Fallback if no location comma is found
                    if author == "N/A":
                        author_candidates = []
                        for div in review_container.find_all('div', dir='auto'):
                            text = div.text.strip()
                            if 3 < len(text) < 30 and "Review for" not in text and "Helpful" not in text and text != "Verified Purchase":
                                author_candidates.append(text)
                        if len(author_candidates) >= 3:
                            author = author_candidates[-3]
                        elif len(author_candidates) >= 2:
                            author = author_candidates[-2] 
                        elif len(author_candidates) == 1:
                            author = author_candidates[0]
                    
                    # Date
                    date = "N/A"
                    for div in review_container.find_all('div', dir='auto'):
                        text = div.text.strip()
                        if " ago" in text or re.search(r'\d{4}', text) or re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', text):
                            if len(text) < 20:
                                date = text
                                break
                    
                    reviews_data.append({
                        'Author': author,
                        'Rating': rating,
                        'Title': title,
                        'Date': "Reviewed in India on " + date if date != "N/A" else date,
                        'Body': body
                    })
                except Exception as e:
                    print(f"Error parsing a review: {e}")
                    
            print(f"Scraped {len(verified_elements)} reviews on Flipkart page {page_num}.")
            page_num += 1
            if page_num > 100:  # safety limit
                print("Reached safety limit of 100 pages.")
                break
        
        await browser.close()
        
        # Append to CSV
        if reviews_data:
            df = pd.DataFrame(reviews_data)
            df = df.replace(r'\n',' ', regex=True)
            
            # Read existing CSV
            try:
                existing_df = pd.read_csv(output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                combined_df = df
                
            combined_df.to_csv(output_file, index=False)
            print(f"Successfully scraped {len(reviews_data)} reviews from Flipkart and appended to {output_file}")
        else:
            print("No reviews were scraped.")

if __name__ == "__main__":
    urls = [
        "https://www.flipkart.com/eveready-carbon-zinc-aa-battery/product-reviews/itm64e5a00b03b04?pid=ACCEUHMJYT7HWTV7&lid=LSTACCEUHMJYT7HWTV7ONYTNL&marketplace=FLIPKART"
    ]
    for url in urls:
        print(f"Starting scraping for {url}...")
        asyncio.run(scrape_flipkart_reviews(url, output_file='carbonzinc_aa.csv'))
