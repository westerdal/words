#!/usr/bin/env python3
"""
Find and generate missing embeddings for words in enable2.txt
"""

import json
import openai
import os
from pathlib import Path
from typing import List, Dict

def load_existing_embeddings() -> Dict:
    """Load existing embeddings"""
    with open('.env/embeddings2.json', 'r') as f:
        return json.load(f)

def load_word_list() -> List[str]:
    """Load word list"""
    with open('data/enable2.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def find_missing_words() -> List[str]:
    """Find words that don't have embeddings"""
    embeddings = load_existing_embeddings()
    words = load_word_list()
    
    missing = []
    for word in words:
        if word not in embeddings:
            missing.append(word)
    
    return missing

def generate_embeddings_batch(words: List[str], batch_size: int = 100) -> Dict:
    """Generate embeddings for missing words"""
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    all_embeddings = {}
    
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        print(f"🤖 Generating embeddings for batch {i//batch_size + 1}: {len(batch)} words")
        
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-large"
            )
            
            for j, embedding_data in enumerate(response.data):
                word = batch[j]
                all_embeddings[word] = embedding_data.embedding
                
        except Exception as e:
            print(f"❌ Error generating embeddings for batch: {e}")
            continue
    
    return all_embeddings

def main():
    print("🔍 Finding missing embeddings...")
    missing_words = find_missing_words()
    
    print(f"📊 Found {len(missing_words)} missing words")
    if len(missing_words) > 0:
        print(f"📝 First 10 missing words: {missing_words[:10]}")
        
        # Generate embeddings for missing words
        print("🤖 Generating missing embeddings...")
        new_embeddings = generate_embeddings_batch(missing_words)
        
        if new_embeddings:
            # Load existing embeddings
            existing_embeddings = load_existing_embeddings()
            
            # Add new embeddings
            existing_embeddings.update(new_embeddings)
            
            # Save updated embeddings
            print("💾 Saving updated embeddings...")
            with open('.env/embeddings2.json', 'w') as f:
                json.dump(existing_embeddings, f)
            
            print(f"✅ Added {len(new_embeddings)} new embeddings")
            print(f"📊 Total embeddings: {len(existing_embeddings)}")
        else:
            print("❌ No new embeddings were generated")
    else:
        print("✅ No missing embeddings found!")

if __name__ == "__main__":
    main()


