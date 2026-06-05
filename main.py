from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import numpy as np
import os
import json

# Import necessary parts from our analyzer script
from sentiment_analyzer import (
    BiLSTMAttention,
    preprocess_dataframe,
    tokens_to_indices,
    generate_product_summaries,
    DEVICE,
    MAX_SEQ_LEN
)

app = FastAPI(title="Eveready Sentiment API")

# Setup CORS so the HTML file can fetch from the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
MODEL = None
WORD2IDX = None
CONFIG = None
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


@app.on_event("startup")
def load_model():
    """Load the PyTorch model and vocabulary into memory on startup."""
    global MODEL, WORD2IDX, CONFIG
    
    model_path = os.path.join(DATA_DIR, 'sentiment_model.pt')
    if not os.path.exists(model_path):
        print("Model file not found. Please run sentiment_analyzer.py first.")
        return
        
    print("Loading PyTorch model into memory...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    WORD2IDX = checkpoint['word2idx']
    CONFIG = checkpoint['config']
    
    # Initialize model architecture
    MODEL = BiLSTMAttention(
        vocab_size=CONFIG['vocab_size'],
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_classes=CONFIG['num_classes']
    ).to(DEVICE)
    
    MODEL.load_state_dict(checkpoint['model_state'])
    MODEL.eval()
    print("Model successfully loaded and ready for inference!")


@app.get("/api/catalog")
def get_dynamic_catalog():
    """
    Endpoint that dynamically loads reviews, runs them through 
    the NLP model, and generates actionable insights.
    """
    if MODEL is None or WORD2IDX is None:
        return {"error": "Model not loaded properly"}
        
    # In a real production environment, this would query a SQL database.
    # Here, we read our pre-scraped CSV file to simulate the database.
    master_path = os.path.join(DATA_DIR, 'master_reviews.csv')
    df = pd.read_csv(master_path)
    
    # Text preprocessing
    df = preprocess_dataframe(df)
    
    # Convert text to padded index sequences for the model
    df['indices'] = df['tokens'].apply(lambda t: tokens_to_indices(t, WORD2IDX, max_len=MAX_SEQ_LEN))
    
    # Run the model inference and extract insights
    # (Since we removed the file saving logic, it now returns the data dictionary)
    catalog_data = generate_product_summaries(df, MODEL, WORD2IDX)
    
    return catalog_data
