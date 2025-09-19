#!/usr/bin/env python3
"""
Simple test of the ENABLE2 update functionality
"""

import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "scripts" / "utilities"))
from openai_similar_words import OpenAISimilarWords

def test_enable2_update_simple():
    """Test just the ENABLE2 update functionality with mock data"""
    print("=== Testing ENABLE2 Update Functionality ===")
    
    # Check current enable2.txt word count
    enable2_file = Path("data/enable2.txt")
    if enable2_file.exists():
        with open(enable2_file, 'r', encoding='utf-8') as f:
            initial_count = len([line for line in f if line.strip()])
        print(f"📊 Initial ENABLE2.txt word count: {initial_count:,}")
    else:
        print("❌ ENABLE2.txt not found")
        return False
    
    # Create OpenAI module instance
    try:
        openai_module = OpenAISimilarWords("test")
        
        # Test with some mock new words
        test_new_words = ["testword1", "testword2", "uniqueword123"]
        print(f"🧪 Testing with mock words: {test_new_words}")
        
        # Test the _add_words_to_enable2 method directly
        success = openai_module._add_words_to_enable2(test_new_words)
        
        if success:
            # Check updated word count
            with open(enable2_file, 'r', encoding='utf-8') as f:
                final_count = len([line for line in f if line.strip()])
            
            added_count = final_count - initial_count
            print(f"📊 Final ENABLE2.txt word count: {final_count:,} (+{added_count})")
            
            if added_count > 0:
                print(f"✅ Successfully added {added_count} new words to ENABLE2.txt!")
                
                # Show the last few lines to verify
                print("📝 Last few lines of ENABLE2.txt:")
                with open(enable2_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-5:]:
                        print(f"   {line.strip()}")
            else:
                print("📝 No new words added (all were already in ENABLE2.txt)")
            
            return True
        else:
            print("❌ Failed to add words to ENABLE2.txt")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_enable2_update_simple()
    sys.exit(0 if success else 1)
