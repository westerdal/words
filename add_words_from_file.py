#!/usr/bin/env python3
"""
Add secret words to the batch queue from a text file
"""

import sys
from batch_secretword_generator import BatchSecretWordGenerator

def add_words_from_file(filename):
    """Add words from a text file to the processing queue."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines() if line.strip()]
        
        if not words:
            print(f"❌ No words found in {filename}")
            return
        
        print(f"📄 Found {len(words)} words in {filename}")
        
        # Show preview
        print("📝 Preview (first 10 words):")
        for i, word in enumerate(words[:10]):
            print(f"  {i+1:2d}. {word}")
        if len(words) > 10:
            print(f"  ... and {len(words) - 10} more")
        
        # Confirm
        confirm = input(f"\nAdd these {len(words)} words to the queue? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            generator = BatchSecretWordGenerator()
            added_count = generator.add_words_to_queue(words)
            print(f"✅ Successfully added {added_count} new words to the queue")
        else:
            print("❌ Cancelled - no words added")
    
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python add_words_from_file.py <filename>")
        print("\nExample:")
        print("  python add_words_from_file.py word_list.txt")
        sys.exit(1)
    
    filename = sys.argv[1]
    add_words_from_file(filename)

if __name__ == "__main__":
    main()
