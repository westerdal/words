#!/usr/bin/env python3
"""
Add missing embedding for 'fish' to embeddings2.json
"""

import json
import openai
import os
from pathlib import Path

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_embedding(word):
    """Get embedding for a word using OpenAI"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=word
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error getting embedding for '{word}': {e}")
        return None

def add_fish_embedding():
    """Add fish embedding to embeddings2.json"""
    embeddings_file = Path(".env/embeddings2.json")
    
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    print(f"📂 Loading embeddings from {embeddings_file}")
    print(f"⚠️ This may take several minutes for the large file...")
    
    # Load existing embeddings
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    
    print(f"✅ Loaded {len(embeddings):,} existing embeddings")
    
    # Check if fish already exists
    if 'fish' in embeddings:
        print(f"✅ 'fish' already has an embedding")
        return True
    
    print(f"🔄 Getting embedding for 'fish'...")
    fish_embedding = get_embedding('fish')
    
    if fish_embedding is None:
        print(f"❌ Failed to get embedding for 'fish'")
        return False
    
    # Add fish embedding
    embeddings['fish'] = fish_embedding
    
    print(f"💾 Saving updated embeddings to {embeddings_file}")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)
    
    print(f"✅ Successfully added 'fish' embedding to embeddings2.json")
    print(f"📊 Total embeddings: {len(embeddings):,}")
    
    return True

if __name__ == "__main__":
    print("🐟 Adding 'fish' embedding to embeddings2.json")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        exit(1)
    
    success = add_fish_embedding()
    
    if success:
        print("\n🎉 Ready to process 'fish' secretword!")
    else:
        print("\n💥 Failed to add fish embedding")
        exit(1)


