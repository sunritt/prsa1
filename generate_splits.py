import pandas as pd
import numpy as np
import json
import torch
import os

from sentiment_analyzer import (
    BiLSTMAttention,
    preprocess_dataframe,
    tokens_to_indices,
    generate_product_summaries,
    DEVICE,
    MAX_SEQ_LEN
)

def main():
    print("Loading PyTorch model...")
    model_path = 'sentiment_model.pt'
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    word2idx = checkpoint['word2idx']
    config = checkpoint['config']
    
    model = BiLSTMAttention(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print("Loading master_reviews.csv...")
    df = pd.read_csv('master_reviews.csv')
    
    # Shuffle the dataframe to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess the entire dataframe once to save time
    print("Preprocessing text...")
    df = preprocess_dataframe(df)
    df['indices'] = df['tokens'].apply(lambda t: tokens_to_indices(t, word2idx, max_len=MAX_SEQ_LEN))

    # Split into 5 chunks using native slicing to preserve DataFrame type
    num_chunks = 5
    chunk_size = int(np.ceil(len(df) / num_chunks))
    chunks = [df[i*chunk_size : (i+1)*chunk_size] for i in range(num_chunks)]
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing Chunk {i}/{num_chunks} (Rows: {len(chunk)})")
        
        # Run inference and summary generation on just this chunk
        # Note: generate_product_summaries calculates stats based on the length of the chunk provided
        catalog_data = generate_product_summaries(chunk, model, word2idx)
        
        # Save to file
        output_file = f'catalog_data_{i}.js'
        with open(output_file, 'w') as f:
            f.write("const catalogData = ")
            json.dump(catalog_data, f, indent=4)
            f.write(";")
        
        print(f"-> Saved distinct catalog to {output_file}")

if __name__ == "__main__":
    main()
