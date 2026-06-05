import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import time
import random

async def scrape_amazon_reviews(url, output_file='reviews.csv'):
    async with async_playwright() as p:
        # Use a headed browser to reduce bot detection
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until='domcontentloaded')
        
        # Wait a bit to set cookies
        await page.wait_for_timeout(3000)
        
        # Find "See all reviews" link href and navigate to it to avoid login wall
        try:
            see_all_link = await page.wait_for_selector('a[data-hook="see-all-reviews-link-foot"]', timeout=5000)
            if see_all_link:
                reviews_href = await see_all_link.get_attribute('href')
                if reviews_href:
                    if not reviews_href.startswith('http'):
                        reviews_href = "https://www.amazon.in" + reviews_href
                    print(f"Navigating to reviews page: {reviews_href}")
                    await page.goto(reviews_href, wait_until='domcontentloaded')
                    await page.wait_for_timeout(3000)
        except Exception as e:
            print(f"Could not find 'See all reviews': {e}")
            print("Will attempt to scrape reviews from the current page.")
        
        reviews_data = []
        page_num = 1
        
        while True:
            print(f"Scraping page {page_num}...")
            # Ensure reviews are loaded
            try:
                await page.wait_for_selector('div[data-hook="review"]', timeout=10000)
            except:
                print("No reviews found or timeout waiting for reviews.")
                await page.screenshot(path=f"debug_amazon_page_{page_num}.png")
                break
                
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            review_elements = soup.find_all('div', {'data-hook': 'review'})
            if not review_elements:
                print("No review elements found in HTML.")
                break
                
            for review in review_elements:
                # Extract rating
                rating_elem = review.find('i', {'data-hook': 'review-star-rating'}) or review.find('i', {'data-hook': 'cmps-review-star-rating'})
                rating = rating_elem.text.strip() if rating_elem else "N/A"
                
                # Extract title
                title_elem = review.find('a', {'data-hook': 'reviewTitle'}) or review.find('h5', {'data-hook': 'reviewTitle'}) or review.find('span', {'data-hook': 'review-title'})
                title = title_elem.text.strip() if title_elem else "N/A"
                
                # Extract body
                body_elem = review.find('div', {'data-hook': 'reviewTextContainer'}) or review.find('span', {'data-hook': 'review-body'}) or review.find('div', {'data-hook': 'reviewRichContentContainer'})
                if body_elem:
                    body = body_elem.text.strip()
                    # Clean up Amazon's boilerplate text
                    body = body.replace('Brief content visible, double tap to read full content.', '')
                    body = body.replace('Full content visible, double tap to read brief content.', '')
                    body = body.replace('Read moreRead less', '')
                    body = body.strip()
                else:
                    body = "N/A"
                
                # Extract author
                author_elem = review.find('span', class_='a-profile-name')
                author = author_elem.text.strip() if author_elem else "N/A"
                
                # Extract date
                date_elem = review.find('span', {'data-hook': 'review-date'})
                date = date_elem.text.strip() if date_elem else "N/A"
                
                reviews_data.append({
                    'Author': author,
                    'Rating': rating,
                    'Title': title,
                    'Date': date,
                    'Body': body
                })
            
            print(f"Scraped {len(review_elements)} reviews on page {page_num}.")
            
            # Try to click next page
            try:
                next_button = await page.query_selector('li.a-last a')
                if next_button:
                    print("Navigating to next page...")
                    await next_button.click()
                    await page.wait_for_load_state('domcontentloaded')
                    await page.wait_for_timeout(random.randint(2000, 4000)) # Random delay
                    page_num += 1
                else:
                    print("No more pages found.")
                    break
            except Exception as e:
                print(f"Could not navigate to next page: {e}")
                break
                
            # For this project, limit to 100 pages to avoid taking too long or getting blocked (infinite loops)
            if page_num > 100:
                print("Reached limit of 100 pages. Stopping pagination.")
                break
                
        await browser.close()
        
        # Save to CSV
        if reviews_data:
            df = pd.DataFrame(reviews_data)
            # Clean up newlines in text
            df = df.replace(r'\n',' ', regex=True)
            df.to_csv(output_file, index=False)
            print(f"Successfully saved {len(reviews_data)} reviews to {output_file}")
        else:
            print("No reviews were scraped.")

if __name__ == "__main__":
    url = "https://www.amazon.in/product-reviews/B0D2RQJHCB/ref=cm_cr_arp_d_paging_btm_2?_encoding=UTF8&ie=UTF8&reviewerType=all_reviews&pageNumber=2&nextPageToken=MjAyNi0wNi0wMlQwNzoyNTowNy4zMDA0Mjc1NjZaADEw"
    asyncio.run(scrape_amazon_reviews(url, output_file='DL47Mega.csv'))
