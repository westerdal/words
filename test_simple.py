#!/usr/bin/env python3
"""
Simple test for the three-method expansion
"""

import os
import sys
from pathlib import Path

# Set OpenAI key
os.environ['OPENAI_API_KEY'] = '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'

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from scripts.utilities.openai_similar_words import OpenAISimilarWords
    
    print("🧪 Testing enhanced three-method expansion for: 'juice'")
    print("=" * 60)
    
    # Test the enhanced system
    openai_words = OpenAISimilarWords("juice")
    results = openai_words.get_two_pass_expansion()
    
    if results:
        print(f"\n✅ SUCCESS: Generated {len(results)} total words")
        print(f"📋 First 20 words: {', '.join(results[:20])}")
        
        # Check for specific words we expect from simplified method
        test_words = ['smoothie', 'antioxidant', 'vitamin', 'nutrition', 'breakfast', 'beverage', 'drink']
        found_words = [w for w in test_words if w in results]
        if found_words:
            print(f"🎯 Found expected words: {', '.join(found_words)}")
        else:
            print("⚠️ None of the expected test words found")
        
        print(f"\n📊 All words: {', '.join(results)}")
    else:
        print("❌ FAILED: No results generated")
        
except Exception as e:
    import traceback
    print(f"❌ ERROR: {e}")
    print(f"Traceback: {traceback.format_exc()}")

