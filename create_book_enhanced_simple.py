#!/usr/bin/env python3
"""
Simple enhanced embeddings creation for 'book' - bypasses full embeddings loading
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from config import Config
from progress_tracker import create_tracker, quick_log
from openai_similar_words import get_openai_similar_words
from plural_converter import convert_plurals_to_singular

def create_book_enhanced_simple():
    """Create enhanced embeddings for 'book' without loading full embeddings cache"""
    secret_word = "book"
    
    print(f"=== Creating Enhanced Embeddings for '{secret_word}' (Simple Mode) ===")
    
    # Step 1: Get OpenAI similar words (already cached)
    quick_log(secret_word, f"🤖 Getting OpenAI similar words for '{secret_word}'")
    openai_words = get_openai_similar_words(secret_word)
    
    if not openai_words:
        quick_log(secret_word, "❌ ERROR: No OpenAI words retrieved")
        return False
    
    # Step 2: Convert plurals and remove duplicates
    processed_words = convert_plurals_to_singular(secret_word, openai_words)
    
    if not processed_words:
        quick_log(secret_word, "❌ ERROR: No words remaining after processing")
        return False
    
    quick_log(secret_word, f"✅ Processed OpenAI words: {len(openai_words)} → {len(processed_words)}")
    
    # Step 3: Create a simple embeddings-book2.txt file with OpenAI words at top
    # For now, just put OpenAI words at the top with fake similarity scores
    output_file = Config.SECRETWORD_DIR / f"embeddings-{secret_word}2.txt"
    
    quick_log(secret_word, f"📝 Creating enhanced embeddings file: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write OpenAI words first (ranks 1-N)
            for i, word in enumerate(processed_words, 1):
                # Format: rank word similarity [notes]
                if word == secret_word:
                    f.write(f"1 {word} 1.0000 [Secret word]\n")
                else:
                    # Fake similarity scores decreasing from 0.95
                    similarity = 0.95 - (i * 0.01)
                    f.write(f"{i} {word} {similarity:.4f} [OpenAI #{i}]\n")
        
        quick_log(secret_word, f"✅ Created enhanced embeddings with {len(processed_words)} OpenAI words")
        quick_log(secret_word, f"📁 File: {output_file}")
        
        # Show first few entries
        quick_log(secret_word, "📝 First 10 entries:")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                quick_log(secret_word, f"   {line.strip()}")
        
        return True
        
    except Exception as e:
        quick_log(secret_word, f"❌ ERROR: Failed to create enhanced embeddings: {e}")
        return False

if __name__ == "__main__":
    success = create_book_enhanced_simple()
    if success:
        print(f"\n🎉 Successfully created enhanced embeddings for 'book'!")
    else:
        print(f"\n💥 Failed to create enhanced embeddings for 'book'")
    sys.exit(0 if success else 1)
