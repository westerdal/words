#!/usr/bin/env python3
"""
Create clean embeddings files by filtering out words that contain the secret word
"""

import csv
from pathlib import Path

def contains_secret_word(word, secret_word):
    """Check if word contains the secret word (case insensitive)"""
    return secret_word.lower() in word.lower()

def create_clean_embeddings(secret_word, input_file, output_file):
    """Create clean embeddings by filtering out contaminated words"""
    print(f"🧹 Creating clean embeddings for '{secret_word}'")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    
    contaminated_words = []
    clean_words = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 3:
                    continue
                
                rank, word, similarity = row[0], row[1], row[2]
                note = row[3] if len(row) > 3 else ""
                
                if contains_secret_word(word, secret_word):
                    contaminated_words.append(word)
                else:
                    clean_words.append((rank, word, similarity, note))
        
        print(f"📊 Filtering results:")
        print(f"   Original words: {len(contaminated_words) + len(clean_words):,}")
        print(f"   Contaminated words removed: {len(contaminated_words):,}")
        print(f"   Clean words remaining: {len(clean_words):,}")
        
        # Write clean embeddings with re-ranked positions
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Write header
            if len(clean_words[0]) > 3 and clean_words[0][3]:  # Has note column
                writer.writerow(['rank', 'word', 'similarity', 'note'])
            else:
                writer.writerow(['rank', 'word', 'similarity'])
            
            # Write clean words with new sequential ranks
            for new_rank, (old_rank, word, similarity, note) in enumerate(clean_words, 1):
                if note:
                    writer.writerow([new_rank, word, similarity, note])
                else:
                    writer.writerow([new_rank, word, similarity])
        
        print(f"✅ Clean embeddings saved to {output_file}")
        
        # Show some examples of removed words
        if contaminated_words:
            print(f"\n🗑️ Examples of removed contaminated words:")
            for word in contaminated_words[:20]:
                print(f"   - {word}")
            if len(contaminated_words) > 20:
                print(f"   ... and {len(contaminated_words) - 20} more")
        
        return len(clean_words), len(contaminated_words)
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
        return 0, 0
    except Exception as e:
        print(f"❌ Error processing files: {e}")
        return 0, 0

if __name__ == "__main__":
    secret_word = "fish"
    
    print("=" * 60)
    print(f"🧹 CREATING CLEAN EMBEDDINGS FOR '{secret_word.upper()}'")
    print("=" * 60)
    
    # Process both standard and enhanced embeddings
    files_to_clean = [
        ("secretword/embeddings-fish.txt", "secretword/embeddings-fish-clean.txt"),
        ("secretword/embeddings-fish2.txt", "secretword/embeddings-fish2-clean.txt")
    ]
    
    for input_file, output_file in files_to_clean:
        print(f"\n{'-' * 40}")
        clean_count, contaminated_count = create_clean_embeddings(secret_word, input_file, output_file)
        
        if clean_count > 0:
            contamination_rate = (contaminated_count / (clean_count + contaminated_count)) * 100
            print(f"📈 Contamination rate: {contamination_rate:.1f}%")
    
    print(f"\n🎉 Clean embeddings created!")
    print(f"💡 Use the *-clean.txt files for CSV generation to avoid secret word leakage")


