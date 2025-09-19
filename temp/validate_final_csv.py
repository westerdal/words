#!/usr/bin/env python3
"""
Validate the final generated CSV file
"""

import pandas as pd

def validate_csv(filename="final_semantic_rank_planet.csv"):
    """Validate the final CSV file."""
    print(f"🔍 Validating {filename}...")
    
    # Load the CSV
    df = pd.read_csv(filename)
    
    print(f"✅ Rows: {len(df):,}")
    print(f"✅ Columns: {list(df.columns)}")
    
    # Check secret word
    secret_word_row = df[df['word'] == 'planet']
    if len(secret_word_row) > 0:
        print(f"✅ Secret word 'planet' has rank: {secret_word_row['rank'].iloc[0]}")
    else:
        print("❌ Secret word 'planet' not found!")
    
    # Check rank distribution
    print(f"\n📊 Rank Distribution:")
    print(f"  Rank 1: {len(df[df['rank'] == 1])}")
    print(f"  Ranks 2-1000: {len(df[(df['rank'] >= 2) & (df['rank'] <= 1000)])}")
    print(f"  Ranks 1001-5000: {len(df[(df['rank'] >= 1001) & (df['rank'] <= 5000)])}")
    print(f"  Ranks 5001-50000: {len(df[(df['rank'] >= 5001) & (df['rank'] <= 50000)])}")
    print(f"  Ranks 50001+: {len(df[df['rank'] >= 50001])}")
    
    print(f"\n🎯 Top 10 closest to 'planet':")
    top_10 = df.head(10)
    for _, row in top_10.iterrows():
        print(f"  {row['rank']:3d}. {row['word']:15s} - {row['clue']}")
    
    print(f"\n🎲 Sample distant words (50001+):")
    if len(df[df['rank'] >= 50001]) > 0:
        distant = df[df['rank'] >= 50001].sample(5)
        for _, row in distant.iterrows():
            print(f"  {row['rank']:6d}. {row['word']:15s} - {row['clue']}")
    
    # Check for proper "nothing like" clues
    nothing_like_clues = df[df['clue'].str.contains('nothing like', case=False, na=False)]
    print(f"\n📝 'Nothing like' clues: {len(nothing_like_clues):,}")
    
    # File size
    import os
    file_size = os.path.getsize(filename)
    print(f"\n💾 File size: {file_size / 1024 / 1024:.1f} MB")
    
    print(f"\n🎉 Validation Complete - CSV is ready for Semantic Rank game!")

if __name__ == "__main__":
    validate_csv()
