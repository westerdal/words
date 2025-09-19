#!/usr/bin/env python3
"""
Create a basic cat CSV using the existing embeddings and rankings
This will create the initial CSV that we can then process with the queue system
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_cat_csv():
    """Create basic cat CSV with rankings and placeholder clues"""
    
    print("=== Creating Basic Cat CSV ===")
    
    # Check if cat CSV already exists
    cat_csv = "secretword/secretword-easy-animals-cat.csv"
    if os.path.exists(cat_csv):
        print(f"✅ Cat CSV already exists: {cat_csv}")
        return True
    
    secret_word = "cat"
    
    # Load words from enable2.txt
    print("📚 Loading ENABLE2 word list...")
    try:
        with open("data/enable2.txt", 'r', encoding='utf-8') as f:
            words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(words):,} singular words")
    except FileNotFoundError:
        print("❌ ENABLE2 word list not found. Please run create_enable2.py first.")
        return False
    
    # Load embeddings
    print("🧮 Loading embeddings...")
    try:
        with open(".env/embeddings2.json", 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        print(f"✅ Loaded embeddings for {len(embeddings):,} words")
    except FileNotFoundError:
        print("❌ Embeddings2 file not found. Please run create_embeddings2.py first.")
        return False
    
    if secret_word not in embeddings:
        print(f"❌ '{secret_word}' not found in embeddings")
        return False
    
    # Compute semantic rankings
    print("🔍 Computing semantic rankings...")
    secret_embedding = np.array(embeddings[secret_word])
    secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
    
    word_similarities = []
    for word in tqdm(words, desc="Computing similarities"):
        if word in embeddings:
            word_embedding = np.array(embeddings[word])
            word_embedding = word_embedding / np.linalg.norm(word_embedding)
            similarity = np.dot(word_embedding, secret_embedding)
            word_similarities.append((word, similarity))
        else:
            word_similarities.append((word, -1.0))
    
    # Sort by similarity
    word_similarities.sort(key=lambda x: (-x[1], x[0]))
    
    # Create CSV data with basic structure
    print("📄 Creating CSV data...")
    csv_data = []
    for rank, (word, similarity) in enumerate(word_similarities, 1):
        # Basic clue structure - will be updated by queue system
        if rank == 1:
            clue = "This is the *."
        else:
            clue = "placeholder clue"  # Will be replaced by AI processing
        
        csv_data.append({
            'rank': rank,
            'secret_word': secret_word,
            'word': word,
            'clue': clue
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    os.makedirs(os.path.dirname(cat_csv), exist_ok=True)
    df.to_csv(cat_csv, index=False)
    
    file_size = os.path.getsize(cat_csv)
    print(f"✅ Created cat CSV: {cat_csv}")
    print(f"📏 File size: {file_size/1024/1024:.1f} MB")
    print(f"📊 Total rows: {len(df):,}")
    print(f"🎯 '{secret_word}' has rank: 1")
    
    # Show top 10
    print(f"\nTop 10 most similar words to '{secret_word}':")
    for i, (word, sim) in enumerate(word_similarities[:10], 1):
        print(f"  {i:2d}. {word:<15} (similarity: {sim:.6f})")
    
    return True

if __name__ == "__main__":
    create_cat_csv()
