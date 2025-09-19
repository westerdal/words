#!/usr/bin/env python3
"""
Test the embeddings merger for book with the actual OpenAI words
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from embeddings_merger import create_merged_embeddings
from openai_similar_words import get_openai_similar_words

def test_book_merger():
    """Test creating proper merged embeddings for book"""
    secret_word = "book"
    
    print(f"=== Testing Embeddings Merger for '{secret_word}' ===")
    
    # Get the actual OpenAI words
    openai_words = get_openai_similar_words(secret_word)
    
    if not openai_words:
        print("❌ Failed to get OpenAI words")
        return False
    
    print(f"📊 Got {len(openai_words)} OpenAI words")
    print(f"📝 First 5 OpenAI words: {openai_words[:5]}")
    
    # Create merged embeddings
    success = create_merged_embeddings(secret_word, openai_words)
    
    if success:
        # Check the result
        embeddings2_file = Path("secretword/embeddings-book2.txt")
        if embeddings2_file.exists():
            with open(embeddings2_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n✅ Created embeddings-book2.txt with {len(lines)-1:,} words (excluding header)")
            
            # Show first 10 lines
            print("📝 First 10 entries:")
            for i, line in enumerate(lines[:11]):  # Include header
                print(f"   {line.strip()}")
            
            return True
        else:
            print("❌ Merged file was not created")
            return False
    else:
        print("❌ Failed to create merged embeddings")
        return False

if __name__ == "__main__":
    success = test_book_merger()
    sys.exit(0 if success else 1)
