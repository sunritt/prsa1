# Eveready Product Review Sentiment Analyzer (PRSA)

A comprehensive, end-to-end AI pipeline and web dashboard for scraping, analyzing, and visualizing customer reviews across major e-commerce platforms (Amazon, Flipkart, BigBasket) for Eveready products.

## Features

### 1. Web Scraping
- **Multi-Platform Support**: Custom web scrapers utilizing `Playwright` to extract reviews from Amazon, Flipkart, and BigBasket.
- **Interactive Scraping Mode**: Specifically designed interactive scrapers that allow the user to manually navigate through CAPTCHAs or logins before an automated scrolling script takes over to harvest thousands of reviews.

### 2. AI Sentiment Pipeline (`sentiment_analyzer.py`)
- **Deep Learning Classifier**: A custom PyTorch `BiLSTM + Attention` neural network architecture trained to classify raw review text into 3 distinct sentiments: Positive, Neutral, and Negative.
- **Abstractive Summarization**: Utilizes a local `T5-Small` transformer model to read hundreds of reviews and generate a concise, human-readable paragraph summarizing the overall customer sentiment.
- **Spam Detection Filter**: Intelligently identifies and filters out repetitive, low-effort reviews (e.g. "good product good product") so they don't skew the AI summarization or aspect categorization.
- **Amazon-Style Aspect Categorization**: Automatically categorizes reviews by specific product features (e.g., Battery / Power, Brightness, Value for Money, Build Quality) using regex matching, generating topic-specific sentiment counts and AI summaries for each category.

### 3. Modern Interactive Dashboard
- **Vanilla HTML/CSS/JS**: A lightweight, blazing-fast static frontend that doesn't require a backend server to run. Just open `index.html`.
- **Premium Aesthetics**: Features a sleek dark mode design with glassmorphism styling, responsive layouts, and smooth micro-animations.
- **Aspect Pill Tags**: Interactive UI elements that allow users to click on specific aspects (like "Battery (50)") to reveal a sliding card with localized sentiment breakdowns, custom summaries, and direct customer quotes.
- **Raw Review Modal**: A dedicated modal allowing users to read through all verbatim reviews for a selected product, color-coded by sentiment.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sunritt/prsa1.git
   cd prsa1
   ```

2. **Install Dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   Simply open `index.html` in your web browser. No local server required!

## Running the Backend Pipeline

To run the full sentiment analysis pipeline and regenerate the `catalog_data.js` database from your raw CSV files:

```bash
python sentiment_analyzer.py
```

*Note: This will consolidate all `.csv` files in the directory, clean the data, run the PyTorch model, generate T5 summaries, and output the final structured JSON objects into `catalog_data.js`.*

## Data Structure
The application relies on a local array of JSON objects (`catalog_data.js`). The AI pipeline automatically formats and injects properties like:
- `avg_rating`
- `pos_pct`, `neu_pct`, `neg_pct`
- `strengths`, `complaints`, `improvements`
- `aspects` array (Amazon-style categories)
- `raw_reviews` array (All verbatim reviews)

---
*Built as a scalable data analytics tool for uncovering actionable product insights.*
