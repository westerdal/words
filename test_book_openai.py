#!/usr/bin/env python3
"""
Test OpenAI similar words for 'book' with timeout
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from openai_similar_words import get_openai_similar_words

def test_book_openai():
    """Test OpenAI word retrieval for 'book' with progress tracking"""
    print("=== Testing OpenAI Similar Words for 'book' ===")
    
    try:
        # Get OpenAI words with timeout handling
        words = get_openai_similar_words("book")
        
        if words:
            print(f"\n🎉 Retrieved {len(words)} words from OpenAI!")
            print("📝 First 10 words:")
            for i, word in enumerate(words[:10], 1):
                print(f"   {i:2}. {word}")
            
            if len(words) > 10:
                print(f"   ... and {len(words) - 10} more words")
            
            return True
        else:
            print("\n💥 Failed to retrieve words from OpenAI (returned empty list)")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during OpenAI retrieval: {e}")
        return False

if __name__ == "__main__":
    success = test_book_openai()
    sys.exit(0 if success else 1)
