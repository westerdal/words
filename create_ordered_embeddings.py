#!/usr/bin/env python3
"""
Create ordered embedding files for secret words
Format: secretword/[secretword]-embeddings.txt
Each line: rank,word,similarity_score
This provides a checkpoint system for processing
"""

import os
import json
import numpy as np
from tqdm import tqdm

def create_ordered_embeddings(secret_word):
    """Create ordered embeddings file for a secret word"""
    
    print(f"\n=== Creating Ordered Embeddings for '{secret_word.upper()}' ===")
    
    # File paths
    output_file = f"secretword/embeddings-{secret_word}.txt"
    
    # Check if already exists
    if os.path.exists(output_file):
        print(f"✅ Ordered embeddings already exist: {output_file}")
        
        # Show file info
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        file_size = os.path.getsize(output_file)
        
        print(f"📊 File contains {line_count:,} ranked words")
        print(f"📏 File size: {file_size/1024/1024:.1f} MB")
        return True
    
    # Load words from enable2.txt
    print("📚 Loading ENABLE2 word list...")
    try:
        with open("data/enable2.txt", 'r', encoding='utf-8') as f:
            words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(words):,} singular words")
    except FileNotFoundError:
        print("❌ ENABLE2 word list not found. Please run create_enable2.py first.")
        return False
    
    # Check if secret word is in the list
    if secret_word not in words:
        print(f"❌ Secret word '{secret_word}' not found in ENABLE2 word list")
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
    
    # Compute semantic similarities
    print(f"🔍 Computing semantic similarities for '{secret_word}'...")
    
    # Get secret word embedding and normalize
    secret_embedding = np.array(embeddings[secret_word])
    secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
    
    # Compute similarities for all words
    word_similarities = []
    missing_embeddings = 0
    
    for word in tqdm(words, desc="Computing similarities"):
        if word in embeddings:
            word_embedding = np.array(embeddings[word])
            word_embedding = word_embedding / np.linalg.norm(word_embedding)
            similarity = np.dot(word_embedding, secret_embedding)
            word_similarities.append((word, similarity))
        else:
            # Assign very low similarity for missing embeddings
            word_similarities.append((word, -1.0))
            missing_embeddings += 1
    
    if missing_embeddings > 0:
        print(f"⚠️  {missing_embeddings:,} words missing from embeddings")
    
    # Sort by similarity (descending), then alphabetically for ties
    print("📊 Sorting by similarity...")
    word_similarities.sort(key=lambda x: (-x[1], x[0]))
    
    # Verify secret word is at rank 1
    if word_similarities[0][0] != secret_word:
        print(f"⚠️  Warning: '{secret_word}' is not at rank 1!")
        print(f"    Top word: '{word_similarities[0][0]}' (similarity: {word_similarities[0][1]:.6f})")
    else:
        print(f"✅ '{secret_word}' correctly ranked at position 1")
    
    # Create output directory
    os.makedirs("secretword", exist_ok=True)
    
    # Write ordered embeddings file
    print(f"💾 Writing ordered embeddings to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("rank,word,similarity\n")
        
        # Write all ranked words
        for rank, (word, similarity) in enumerate(word_similarities, 1):
            f.write(f"{rank},{word},{similarity:.8f}\n")
    
    # Report results
    file_size = os.path.getsize(output_file)
    print(f"✅ Created ordered embeddings file: {output_file}")
    print(f"📏 File size: {file_size/1024/1024:.1f} MB")
    print(f"📊 Total words ranked: {len(word_similarities):,}")
    
    # Show top 10 for verification
    print(f"\n🔝 Top 10 most similar words to '{secret_word}':")
    for i, (word, similarity) in enumerate(word_similarities[:10], 1):
        print(f"  {i:2d}. {word:<15} (similarity: {similarity:.6f})")
    
    print(f"\n💡 This file can now be used as a checkpoint for processing '{secret_word}'")
    print(f"💡 No need to recompute embeddings - just load this ranked list!")
    
    return True

def load_ordered_embeddings(secret_word):
    """Load ordered embeddings from file"""
    
    embeddings_file = f"secretword/embeddings-{secret_word}.txt"
    
    if not os.path.exists(embeddings_file):
        print(f"❌ Ordered embeddings file not found: {embeddings_file}")
        return None
    
    print(f"📂 Loading ordered embeddings from {embeddings_file}")
    
    ranked_words = []
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                rank = int(parts[0])
                word = parts[1]
                similarity = float(parts[2])
                ranked_words.append((rank, word, similarity))
    
    print(f"✅ Loaded {len(ranked_words):,} ranked words")
    return ranked_words

def main():
    """Create ordered embeddings for words in master list"""
    print("=== Creating Ordered Embeddings for Secret Words ===")
    
    # Read master list
    master_list_path = "secretword/master-list.txt"
    
    if not os.path.exists(master_list_path):
        print(f"❌ Master list not found: {master_list_path}")
        return False
    
    with open(master_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📋 Found {len(lines)} words in master list")
    
    # Process first few words as examples
    words_to_process = ['cat', 'dog']  # Start with these
    
    for secret_word in words_to_process:
        try:
            success = create_ordered_embeddings(secret_word)
            if success:
                print(f"✅ {secret_word.upper()} embeddings created successfully")
            else:
                print(f"❌ Failed to create {secret_word.upper()} embeddings")
        except KeyboardInterrupt:
            print(f"\n⏸️  Process interrupted for {secret_word}")
            break
        except Exception as e:
            print(f"❌ Error processing {secret_word}: {e}")
    
    print("\n🎉 Ordered embeddings creation complete!")
    print("💡 These files can now be used as checkpoints for fast CSV processing")

if __name__ == "__main__":
    main()
