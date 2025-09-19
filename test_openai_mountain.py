#!/usr/bin/env python3
"""
Test OpenAI word retrieval for mountain without full embeddings generation
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from openai_similar_words import get_openai_similar_words

def test_mountain_words():
    """Test OpenAI word retrieval for mountain"""
    print("=== Testing OpenAI Word Retrieval for 'mountain' ===")
    
    # Get OpenAI words
    words = get_openai_similar_words("mountain")
    
    if words:
        print(f"\n🎉 Successfully retrieved {len(words)} words!")
        print("\nTop 20 words:")
        for i, word in enumerate(words[:20], 1):
            print(f"  {i:>2}. {word}")
        
        if len(words) > 20:
            print(f"  ... and {len(words) - 20} more")
        
        return True
    else:
        print("\n💥 Failed to retrieve words from OpenAI")
        return False

if __name__ == "__main__":
    success = test_mountain_words()
    sys.exit(0 if success else 1)
