#!/usr/bin/env python3
"""
Test the ENABLE2 update functionality with mountain
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from openai_similar_words import get_openai_similar_words

def test_enable2_update():
    """Test OpenAI word retrieval and ENABLE2 update for mountain"""
    print("=== Testing ENABLE2 Update with 'mountain' ===")
    
    # Check current enable2.txt word count
    enable2_file = Path("data/enable2.txt")
    if enable2_file.exists():
        with open(enable2_file, 'r', encoding='utf-8') as f:
            initial_count = len([line for line in f if line.strip()])
        print(f"📊 Initial ENABLE2.txt word count: {initial_count:,}")
    else:
        print("❌ ENABLE2.txt not found")
        return False
    
    # Get OpenAI words (this should add new words to ENABLE2.txt)
    words = get_openai_similar_words("mountain")
    
    if words:
        print(f"\n🎉 Retrieved {len(words)} words from OpenAI!")
        
        # Check updated word count
        with open(enable2_file, 'r', encoding='utf-8') as f:
            final_count = len([line for line in f if line.strip()])
        
        added_count = final_count - initial_count
        print(f"📊 Final ENABLE2.txt word count: {final_count:,} (+{added_count})")
        
        if added_count > 0:
            print(f"✅ Successfully added {added_count} new words to ENABLE2.txt!")
        else:
            print("📝 No new words added (all were already in ENABLE2.txt)")
        
        return True
    else:
        print("\n💥 Failed to retrieve words from OpenAI")
        return False

if __name__ == "__main__":
    success = test_enable2_update()
    sys.exit(0 if success else 1)
