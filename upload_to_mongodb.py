import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def upload_data():
    if not MONGO_URI:
        print("MONGO_URI is not set in .env")
        return

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client.get_database("reviews_db")
    collection = db.get_collection("product_reviews")
    
    csv_file = "master_reviews.csv"
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return
        
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # We only need relevant columns
    cols = ['Product_Name', 'Author', 'rating_num', 'text']
    df = df[cols].dropna(subset=['Product_Name', 'text'])
    
    records = df.to_dict('records')
    
    print(f"Uploading {len(records)} records to MongoDB...")
    # Drop collection to prevent duplicate uploads during testing
    collection.drop()
    collection.insert_many(records)
    
    print("Upload complete!")

if __name__ == "__main__":
    upload_data()
