#!/usr/bin/env python3
"""
Fix the dog CSV by properly implementing the weak connection queue
This script will:
1. Read the current dog CSV
2. Read the weak queue file
3. Remove weak connections from their original positions
4. Resequence ranks to fill gaps
5. Insert weak connections at the end (before rank 10K)
"""

import os
import json
import pandas as pd

def fix_dog_csv_with_queue():
    """Fix the dog CSV by properly applying the weak connection queue"""
    
    print("=== Fixing Dog CSV with Weak Connection Queue ===")
    
    # File paths
    csv_file = "secretword/secretword-easy-animals-dog.csv"
    queue_file = "secretword/secretword-easy-animals-dog_weak_queue.json"
    fixed_file = "secretword/secretword-easy-animals-dog_fixed.csv"
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
        
    if not os.path.exists(queue_file):
        print(f"❌ Queue file not found: {queue_file}")
        return False
    
    # Load CSV
    print(f"📂 Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df):,} rows")
    
    # Load queue
    print(f"📂 Loading weak connection queue: {queue_file}")
    with open(queue_file, 'r', encoding='utf-8') as f:
        queue_data = json.load(f)
    
    weak_connections = queue_data.get('weak_connections', [])
    print(f"✅ Found {len(weak_connections)} weak connections to process")
    
    if not weak_connections:
        print("ℹ️  No weak connections to process")
        return True
    
    # Get list of words to remove
    words_to_remove = [entry['word'] for entry in weak_connections]
    print(f"🗑️  Words to remove from original positions: {words_to_remove[:10]}..." if len(words_to_remove) > 10 else f"🗑️  Words to remove: {words_to_remove}")
    
    # Remove weak connection words from DataFrame
    print(f"🔄 Removing {len(words_to_remove)} weak connections from original positions...")
    df_filtered = df[~df['word'].isin(words_to_remove)].copy()
    print(f"✅ Removed {len(df) - len(df_filtered)} rows")
    
    # Resequence ranks to fill gaps
    print(f"📊 Resequencing ranks to fill gaps...")
    df_filtered = df_filtered.sort_values('rank')
    df_filtered['rank'] = range(1, len(df_filtered) + 1)
    print(f"✅ Resequenced ranks from 1 to {len(df_filtered)}")
    
    # Create DataFrame for weak connections to insert at the end
    print(f"📥 Preparing to insert {len(weak_connections)} weak connections at rank 10,000...")
    
    # Find insertion point (rank 10,000 or end of current data)
    insertion_rank = min(10000, len(df_filtered) + 1)
    
    # Adjust ranks of existing entries that would come after insertion point
    mask = df_filtered['rank'] >= insertion_rank
    if mask.any():
        df_filtered.loc[mask, 'rank'] = df_filtered.loc[mask, 'rank'] + len(weak_connections)
        print(f"✅ Adjusted {mask.sum()} existing ranks to make room for weak connections")
    
    # Create weak connections DataFrame
    weak_df_data = []
    for i, weak_entry in enumerate(weak_connections):
        weak_df_data.append({
            'rank': insertion_rank + i,
            'secret_word': 'dog',
            'word': weak_entry['word'],
            'clue': 'weak connection',
            'connection_strength': 'weak'
        })
    
    weak_df = pd.DataFrame(weak_df_data)
    
    # Combine DataFrames
    print(f"🔗 Combining main CSV with weak connections...")
    final_df = pd.concat([df_filtered, weak_df], ignore_index=True)
    final_df = final_df.sort_values('rank').reset_index(drop=True)
    
    # Add connection_strength column if missing for main entries
    if 'connection_strength' not in df_filtered.columns:
        # For existing entries, we don't know their strength, so leave empty
        mask = final_df['connection_strength'].isna()
        final_df.loc[mask, 'connection_strength'] = ''
    
    # Save fixed CSV
    print(f"💾 Saving fixed CSV: {fixed_file}")
    final_df.to_csv(fixed_file, index=False)
    
    # Report results
    file_size = os.path.getsize(fixed_file)
    weak_connection_clues = len([c for c in final_df['clue'] if c == 'weak connection'])
    
    print(f"\n📊 Fix Results:")
    print(f"   Original CSV rows: {len(df):,}")
    print(f"   Rows after removing weak connections: {len(df_filtered):,}")
    print(f"   Weak connections reinserted: {len(weak_connections):,}")
    print(f"   Final CSV rows: {len(final_df):,}")
    print(f"   Weak connection clues: {weak_connection_clues:,}")
    print(f"   Weak connections inserted at ranks: {insertion_rank}-{insertion_rank + len(weak_connections) - 1}")
    print(f"   File size: {file_size/1024/1024:.1f} MB")
    
    # Show some examples
    print(f"\n📋 Sample of resequenced ranks (first 10 rows):")
    print(final_df[['rank', 'word', 'clue']].head(10).to_string(index=False))
    
    print(f"\n📋 Sample of weak connections (ranks {insertion_rank}-{insertion_rank+4}):")
    weak_sample = final_df[(final_df['rank'] >= insertion_rank) & (final_df['rank'] < insertion_rank + 5)]
    print(weak_sample[['rank', 'word', 'clue', 'connection_strength']].to_string(index=False))
    
    print(f"\n✅ Dog CSV fixed successfully!")
    print(f"💡 Review the fixed file: {fixed_file}")
    print(f"💡 If it looks good, you can replace the original: mv {fixed_file} {csv_file}")
    
    return True

if __name__ == "__main__":
    fix_dog_csv_with_queue()
