#!/usr/bin/env python3
"""
Complete the cat CSV by adding all remaining words after the AI cutoff
"""

import os
import pandas as pd

def complete_cat_csv():
    """Add all remaining words to the cat CSV after AI cutoff"""
    
    print("=== Completing Cat CSV with Remaining Words ===")
    
    # Load current CSV
    csv_file = "secretword/secretword-easy-animals-cat.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Cat CSV not found: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"📂 Current CSV has {len(df):,} rows")
    
    # Load ordered embeddings to get all words
    embeddings_file = "secretword/embeddings-cat.txt"
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    # Load all ranked words from embeddings file
    print(f"📂 Loading all ranked words from {embeddings_file}")
    all_ranked_words = []
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                rank = int(parts[0])
                word = parts[1]
                similarity = float(parts[2])
                all_ranked_words.append((rank, word, similarity))
    
    print(f"✅ Loaded {len(all_ranked_words):,} total words")
    
    # Find the highest rank in current CSV
    max_current_rank = df['rank'].max()
    print(f"📊 Current CSV ends at rank {max_current_rank:,}")
    
    # Find where weak connections were inserted
    weak_connections = df[df['clue'] == 'weak connection']
    weak_start_rank = weak_connections['rank'].min() if len(weak_connections) > 0 else None
    weak_end_rank = weak_connections['rank'].max() if len(weak_connections) > 0 else None
    
    if weak_start_rank:
        print(f"🗂️  Weak connections: ranks {weak_start_rank:,} - {weak_end_rank:,} ({len(weak_connections):,} words)")
        next_rank = weak_end_rank + 1
    else:
        next_rank = max_current_rank + 1
    
    # Get words that are missing (after the cutoff)
    existing_words = set(df['word'].values)
    missing_words = []
    
    for rank, word, similarity in all_ranked_words:
        if word not in existing_words:
            missing_words.append((word, similarity))
    
    print(f"📋 Found {len(missing_words):,} missing words to add")
    
    if not missing_words:
        print("✅ No missing words - CSV is already complete")
        return True
    
    # Create entries for missing words
    new_entries = []
    current_rank = next_rank
    
    for word, similarity in missing_words:
        new_entries.append({
            'rank': current_rank,
            'secret_word': 'cat',
            'word': word,
            'clue': None,  # NULL clue for words beyond cutoff
            'connection_strength': 'hard_cutoff'
        })
        current_rank += 1
    
    # Add new entries to DataFrame
    new_df = pd.DataFrame(new_entries)
    complete_df = pd.concat([df, new_df], ignore_index=True)
    complete_df = complete_df.sort_values('rank').reset_index(drop=True)
    
    # Save completed CSV
    backup_file = csv_file.replace('.csv', '_incomplete.csv')
    print(f"💾 Backing up incomplete CSV to {backup_file}")
    df.to_csv(backup_file, index=False)
    
    print(f"💾 Saving completed CSV...")
    complete_df.to_csv(csv_file, index=False)
    
    # Report results
    file_size = os.path.getsize(csv_file)
    ai_clues = len([c for c in complete_df['clue'] if c and c not in ['This is the *.', 'ERROR', 'weak connection'] and pd.notna(c)])
    weak_connection_clues = len([c for c in complete_df['clue'] if c == 'weak connection'])
    null_clues = len([c for c in complete_df['clue'] if c is None or pd.isna(c)])
    
    print(f"\n📊 Completed CSV Results:")
    print(f"   Original rows: {len(df):,}")
    print(f"   Added rows: {len(new_entries):,}")
    print(f"   Final rows: {len(complete_df):,}")
    print(f"   AI clues: {ai_clues:,}")
    print(f"   Weak connections: {weak_connection_clues:,}")
    print(f"   NULL clues: {null_clues:,}")
    print(f"   File size: {file_size/1024/1024:.1f} MB")
    
    # Show sample of added words
    print(f"\n📋 Sample of added words (last 10):")
    sample = complete_df.tail(10)[['rank', 'word', 'clue', 'connection_strength']]
    print(sample.to_string(index=False))
    
    print(f"\n✅ Cat CSV completed successfully!")
    print(f"💡 All {len(all_ranked_words):,} words are now included")
    
    return True

if __name__ == "__main__":
    complete_cat_csv()
