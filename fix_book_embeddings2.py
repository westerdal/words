#!/usr/bin/env python3
"""
Direct fix for book embeddings2 file - merge OpenAI words with original embeddings
"""

import csv
from pathlib import Path

def fix_book_embeddings2():
    """Create proper embeddings-book2.txt with OpenAI words at top + all original words"""
    
    print("=== Fixing Book Embeddings2 File ===")
    
    # File paths
    openai_file = Path("secretword/openai-book.txt")
    embeddings_file = Path("secretword/embeddings-book.txt")
    embeddings2_file = Path("secretword/embeddings-book2.txt")
    
    # Step 1: Load OpenAI words
    if not openai_file.exists():
        print(f"❌ OpenAI file not found: {openai_file}")
        return False
    
    print(f"📂 Loading OpenAI words from {openai_file}")
    openai_words = []
    with open(openai_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse numbered format: "1. word" -> "word"
            if '. ' in line:
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    word = parts[1].strip().lower()
                    if word and word not in openai_words:  # Avoid duplicates
                        openai_words.append(word)
            else:
                # Direct word format
                word = line.lower()
                if word and word not in openai_words:
                    openai_words.append(word)
    
    print(f"✅ Loaded {len(openai_words)} OpenAI words")
    
    # Step 2: Load original embeddings
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    print(f"📂 Loading original embeddings from {embeddings_file}")
    original_data = {}  # word -> (rank, similarity)
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                rank = int(row[0])
                word = row[1].lower()
                similarity = float(row[2])
                original_data[word] = (rank, similarity)
    
    print(f"✅ Loaded {len(original_data):,} original embeddings")
    
    # Step 3: Create merged data
    print("🔄 Creating merged embeddings...")
    
    merged_data = []
    used_words = set()
    
    # Phase 1: Add OpenAI words at top
    for i, word in enumerate(openai_words, 1):
        word_lower = word.lower()
        
        if word_lower in original_data:
            original_rank, similarity = original_data[word_lower]
            note = f"OpenAI #{i}, Our rank #{original_rank:,}"
        else:
            similarity = 0.0  # Placeholder for words not in our embeddings
            note = f"OpenAI #{i}, Not in our embeddings"
        
        merged_data.append((i, word_lower, similarity, note))
        used_words.add(word_lower)
    
    # Phase 2: Add remaining original words
    remaining_words = []
    for word, (original_rank, similarity) in original_data.items():
        if word not in used_words:
            remaining_words.append((original_rank, word, similarity))
    
    # Sort remaining by original rank
    remaining_words.sort(key=lambda x: x[0])
    
    # Add to merged data
    next_rank = len(openai_words) + 1
    for original_rank, word, similarity in remaining_words:
        note = f"Our original rank #{original_rank:,}"
        merged_data.append((next_rank, word, similarity, note))
        next_rank += 1
    
    print(f"✅ Created merged data with {len(merged_data):,} total words")
    print(f"   OpenAI words: {len(openai_words)}")
    print(f"   Original words added: {len(remaining_words):,}")
    
    # Step 4: Save merged file
    print(f"💾 Saving to {embeddings2_file}")
    
    with open(embeddings2_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['rank', 'word', 'similarity', 'note'])
        
        # Write all data
        for rank, word, similarity, note in merged_data:
            writer.writerow([rank, word, f"{similarity:.8f}", note])
    
    file_size = embeddings2_file.stat().st_size
    print(f"✅ Saved {len(merged_data):,} words ({file_size/1024/1024:.1f} MB)")
    
    # Show first 10 for verification
    print("📊 First 10 entries:")
    for i, (rank, word, similarity, note) in enumerate(merged_data[:10]):
        print(f"   {rank:>3}: {word:<15} {similarity:.6f} [{note}]")
    
    return True

if __name__ == "__main__":
    success = fix_book_embeddings2()
    if success:
        print(f"\n🎉 Successfully fixed embeddings-book2.txt!")
    else:
        print(f"\n💥 Failed to fix embeddings-book2.txt")
    exit(0 if success else 1)
