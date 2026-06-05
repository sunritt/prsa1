import os
from pymongo import MongoClient
from google import genai
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def get_reviews_from_db(product_name, limit=200):
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set")
        
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client.get_database("reviews_db")
    collection = db.get_collection("product_reviews")
    
    # Fetch reviews for the specific product
    cursor = collection.find({"Product_Name": product_name}).limit(limit)
    reviews = [doc['text'] for doc in cursor if 'text' in doc]
    return reviews

def analyze_reviews(product_name, reviews):
    if not reviews:
        return "No reviews found for this product in the database."
        
    combined_reviews = "\n- ".join(reviews)
    
    prompt = f"""Act as a product analyst. Here are the latest {len(reviews)} reviews for {product_name}.
Generate a fresh summary, 3 positive insights, and 3 areas for improvement. Use markdown formatting.

Reviews:
- {combined_reviews}
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"

if __name__ == "__main__":
    # Test the pipeline
    test_product = "10Wbatten"
    print(f"Fetching reviews for {test_product}...")
    reviews = get_reviews_from_db(test_product, limit=5)
    print(f"Found {len(reviews)} reviews. Analyzing...")
    analysis = analyze_reviews(test_product, reviews)
    print("--- Analysis ---")
    print(analysis)
