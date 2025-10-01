#!/usr/bin/env python3
"""
Test the enhanced three-method expansion system
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))

from openai_similar_words import OpenAISimilarWords

def test_word(word):
    print(f"\n🧪 Testing enhanced three-method expansion for: '{word}'")
    print("=" * 60)
    
    try:
        openai_words = OpenAISimilarWords(word)
        results = openai_words.get_similar_words()
        
        if results:
            print(f"\n✅ SUCCESS: Generated {len(results)} total words")
            print(f"📋 First 20 words: {', '.join(results[:20])}")
            
            # Check for specific words we expect from simplified method
            test_words = ['smoothie', 'antioxidant', 'vitamin', 'nutrition', 'breakfast']
            found_words = [w for w in test_words if w in results]
            if found_words:
                print(f"🎯 Found expected words: {', '.join(found_words)}")
            else:
                print("⚠️ None of the expected words found")
        else:
            print("❌ FAILED: No results generated")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_word("juice")

