#!/usr/bin/env python3
"""
Analyze the generated semantic rank CSV file.
"""

import pandas as pd

def analyze_csv(filename="semantic_rank_planet.csv"):
    """Analyze the generated CSV file."""
    print(f"Analyzing {filename}...")
    
    # Load the CSV
    df = pd.read_csv(filename)
    
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nRank distribution:")
    print(f"Rank 1 (secret word): {len(df[df['rank'] == 1])}")
    print(f"Ranks 2-1000 (closest): {len(df[(df['rank'] >= 2) & (df['rank'] <= 1000)])}")
    print(f"Ranks 1001-5000 (medium): {len(df[(df['rank'] >= 1001) & (df['rank'] <= 5000)])}")
    print(f"Ranks 5001-50000 (weak): {len(df[(df['rank'] >= 5001) & (df['rank'] <= 50000)])}")
    print(f"Ranks 50001+ (distant): {len(df[df['rank'] >= 50001])}")
    
    print("\nTop 20 words closest to 'planet':")
    top_20 = df.head(20)
    for _, row in top_20.iterrows():
        print(f"{row['rank']:3d}. {row['word']:15s} - {row['clue']}")
    
    print(f"\nSample from different tiers:")
    
    # Tier 2 sample
    tier2 = df[(df['rank'] >= 2) & (df['rank'] <= 1000)].sample(5)
    print("Tier 2 (ranks 2-1000):")
    for _, row in tier2.iterrows():
        print(f"  {row['rank']:4d}. {row['word']:15s} - {row['clue']}")
    
    # Tier 3 sample
    tier3 = df[(df['rank'] >= 1001) & (df['rank'] <= 5000)].sample(5)
    print("Tier 3 (ranks 1001-5000):")
    for _, row in tier3.iterrows():
        print(f"  {row['rank']:4d}. {row['word']:15s} - {row['clue']}")
    
    # Tier 4 sample
    tier4 = df[(df['rank'] >= 5001) & (df['rank'] <= 50000)].sample(5)
    print("Tier 4 (ranks 5001-50000):")
    for _, row in tier4.iterrows():
        print(f"  {row['rank']:4d}. {row['word']:15s} - {row['clue']}")
    
    # Tier 5 sample
    if len(df[df['rank'] >= 50001]) > 0:
        tier5 = df[df['rank'] >= 50001].sample(5)
        print("Tier 5 (ranks 50001+):")
        for _, row in tier5.iterrows():
            print(f"  {row['rank']:4d}. {row['word']:15s} - {row['clue']}")
    
    print(f"\nFile size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB in memory")

if __name__ == "__main__":
    analyze_csv()
