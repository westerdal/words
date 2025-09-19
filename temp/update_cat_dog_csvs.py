#!/usr/bin/env python3
"""
Update cat and dog CSV files with new dynamic cutoff system
More efficient approach that processes in smaller batches with frequent saves
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
from datetime import datetime

def update_csv_with_dynamic_cutoff(secret_word, consecutive_weak_threshold=5, hard_cutoff_rank=10000):
    """Update a CSV file with dynamic cutoff system"""
    
    print(f"\n=== Updating {secret_word.upper()} CSV with Dynamic Cutoff ===")
    
    # File paths
    csv_file = f"secretword/secretword-easy-animals-{secret_word}.csv"
    backup_file = f"secretword/secretword-easy-animals-{secret_word}_backup.csv"
    temp_file = f"secretword/secretword-easy-animals-{secret_word}_temp.csv"
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Create backup
    print(f"💾 Creating backup: {backup_file}")
    import shutil
    shutil.copy2(csv_file, backup_file)
    
    # Load existing CSV
    print(f"📂 Loading existing CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df):,} rows")
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        use_ai = True
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI not available: {e}")
        return False
    
    # Dynamic cutoff tracking
    consecutive_weak_count = 0
    ai_cutoff_reached = False
    cutoff_rank = None
    total_ai_calls = 0
    weak_connections = []  # Track words with weak connections for rank adjustment
    
    def get_clue_and_strength(guess_word, current_rank):
        """Get clue and relationship strength"""
        nonlocal consecutive_weak_count, ai_cutoff_reached, cutoff_rank, total_ai_calls, weak_connections
        
        if ai_cutoff_reached or current_rank > hard_cutoff_rank:
            return {'clue': None, 'strength': 'cutoff', 'ai_used': False}
        
        try:
            total_ai_calls += 1
            
            prompt = f"""You must analyze the word '{guess_word}' and its relationship to '{secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature/thing' instead of '{secret_word}')
2. "strength": Rate the relationship strength as "strong", "medium", or "weak"

Relationship strength guide:
- "strong": Direct interaction, clear connection (leash, collar, bone, walk)
- "medium": Indirect connection, some logical link (tree, park, house)  
- "weak": Very distant, forced connection, opposite concepts (calculator, mathematics, philosophy)

Focus on HOW they connect, not what '{guess_word}' is.

Word to analyze: {guess_word}

Example format:
{{"clue": "connects to that animal for control", "strength": "strong"}}"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_content = response.choices[0].message.content
            clue_data = json.loads(response_content)
            
            if "clue" in clue_data and "strength" in clue_data:
                clue = clue_data["clue"]
                strength = clue_data["strength"]
                
                # Update weak relationship tracking
                if strength == "weak":
                    consecutive_weak_count += 1
                    # Track weak connections for rank adjustment
                    weak_connections.append(guess_word)
                    if consecutive_weak_count >= consecutive_weak_threshold:
                        ai_cutoff_reached = True
                        cutoff_rank = current_rank
                        print(f"\n🛑 AI Cutoff Reached at rank {cutoff_rank}!")
                        print(f"💰 {consecutive_weak_count} consecutive weak relationships detected.")
                else:
                    consecutive_weak_count = 0
                
                return {'clue': clue, 'strength': strength, 'ai_used': True}
            else:
                return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
                
        except Exception as e:
            print(f"⚠️  AI call failed for '{guess_word}': {e}")
            return {'clue': 'ERROR', 'strength': 'error', 'ai_used': True}
    
    # Add connection_strength column if it doesn't exist
    if 'connection_strength' not in df.columns:
        df['connection_strength'] = ''
        print("✅ Added connection_strength column")
    
    # Process words in batches, starting from rank 2 (skip secret word)
    print(f"🤖 Updating clues with dynamic cutoff system...")
    print(f"📊 Hard cutoff at rank {hard_cutoff_rank:,}")
    
    # Sort by rank and process
    df_sorted = df.sort_values('rank').copy()
    save_interval = 200  # Save every 200 words
    
    updated_count = 0
    last_save = 0
    weak_count = 0
    
    # Skip rank 1 (secret word)
    words_to_process = df_sorted[df_sorted['rank'] > 1].copy()
    
    for idx, row in tqdm(words_to_process.iterrows(), total=len(words_to_process), desc="Updating clues"):
        word = row['word']
        rank = row['rank']
        
        if ai_cutoff_reached or rank > hard_cutoff_rank:
            # Set remaining words to NULL
            df.loc[idx, 'clue'] = None
            df.loc[idx, 'connection_strength'] = 'cutoff' if ai_cutoff_reached else 'hard_cutoff'
        else:
            # Get new clue
            result = get_clue_and_strength(word, rank)
            if result:
                strength = result.get('strength', 'unknown')
                df.loc[idx, 'connection_strength'] = strength
                
                if strength == 'weak':
                    df.loc[idx, 'clue'] = 'weak connection'
                    weak_count += 1
                    print(f"    Rank {rank:>5}: {word:<15} → ⚠️  weak connection (consecutive: {consecutive_weak_count})")
                else:
                    df.loc[idx, 'clue'] = result['clue']
                
                updated_count += 1
        
        # Periodic save
        if (idx - last_save) >= save_interval:
            print(f"💾 Saving progress at rank {rank}...")
            df.to_csv(temp_file, index=False)
            last_save = idx
    
    # Adjust ranks for weak connections (bump down by 2000)
    if weak_connections:
        print(f"🔄 Adjusting ranks for {len(weak_connections)} weak connections...")
        
        for weak_word in weak_connections:
            # Find the word in the dataframe
            mask = df['word'] == weak_word
            if mask.any():
                current_rank = df.loc[mask, 'rank'].iloc[0]
                new_rank = current_rank + 2000
                
                # Update the rank
                df.loc[mask, 'rank'] = new_rank
                print(f"    {weak_word}: rank {current_rank} → {new_rank}")
        
        # Re-sort the dataframe by rank
        df = df.sort_values('rank').reset_index(drop=True)
        print(f"✅ Adjusted {len(weak_connections)} weak connection ranks")
    
    # Final save
    print(f"💾 Saving final CSV...")
    df.to_csv(csv_file, index=False)
    
    # Clean up temp file if it exists
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Report results
    file_size = os.path.getsize(csv_file)
    ai_clues = len([c for c in df['clue'] if c and c not in ['This is the *.', 'ERROR', 'weak connection'] and pd.notna(c)])
    weak_connection_clues = len([c for c in df['clue'] if c == 'weak connection'])
    null_clues = len([c for c in df['clue'] if c is None or pd.isna(c)])
    error_clues = len([c for c in df['clue'] if c == 'ERROR'])
    
    # Connection strength breakdown
    strong_connections = len([s for s in df['connection_strength'] if s == 'strong'])
    medium_connections = len([s for s in df['connection_strength'] if s == 'medium'])
    weak_connections_count = len([s for s in df['connection_strength'] if s == 'weak'])
    cutoff_connections = len([s for s in df['connection_strength'] if s in ['cutoff', 'hard_cutoff']])
    
    print(f"\n📊 Update Results for {secret_word.upper()}:")
    print(f"   Total AI calls: {total_ai_calls:,}")
    print(f"   Words updated: {updated_count:,}")
    print(f"   AI clues: {ai_clues:,}")
    print(f"   Weak connection clues: {weak_connection_clues:,}")
    print(f"   NULL clues: {null_clues:,}")
    print(f"   Error clues: {error_clues:,}")
    print(f"   File size: {file_size/1024/1024:.1f} MB")
    print(f"\n🔗 Connection Strength Breakdown:")
    print(f"   Strong: {strong_connections:,}")
    print(f"   Medium: {medium_connections:,}")
    print(f"   Weak: {weak_connections_count:,} (ranks adjusted +2000)")
    print(f"   Cutoff: {cutoff_connections:,}")
    
    if cutoff_rank:
        total_words = len(df)
        saved_calls = total_words - cutoff_rank
        savings_percent = (saved_calls / total_words) * 100
        print(f"💰 Dynamic cutoff at rank {cutoff_rank:,} saved ~{saved_calls:,} API calls ({savings_percent:.1f}%)")
    
    print(f"✅ {secret_word.upper()} CSV updated successfully!")
    return True

def main():
    """Update both cat and dog CSV files with enhanced features"""
    print("=== Updating Cat and Dog CSVs with Enhanced Dynamic Cutoff ===")
    print("Features:")
    print("• Replace old 5-word clues with new 7-word relationship-focused clues")
    print("• Add connection_strength column to track relationship quality")
    print("• Weak connections get 'weak connection' clue and rank bumped down 2000")
    print("• Hard cutoff at rank 10,000 - beyond that, all clues are NULL")
    print("• Dynamic cutoff after 5 consecutive weak relationships")
    print("• Significant API cost savings through intelligent cutoffs")
    print()
    
    # Check if dog CSV exists, if not, we need to create it first
    if not os.path.exists("secretword/secretword-easy-animals-dog.csv"):
        print("❌ Dog CSV not found. Please generate it first.")
        return False
    
    # Update dog CSV
    try:
        success_dog = update_csv_with_dynamic_cutoff('dog', consecutive_weak_threshold=5, hard_cutoff_rank=10000)
        if success_dog:
            print("✅ Dog CSV updated successfully!")
        else:
            print("❌ Dog CSV update failed!")
            return False
    except KeyboardInterrupt:
        print("\n⏸️  Process interrupted for dog. Progress saved.")
        return False
    
    # Update cat CSV (or create if it doesn't exist)
    try:
        if os.path.exists("secretword/secretword-easy-animals-cat.csv"):
            success_cat = update_csv_with_dynamic_cutoff('cat', consecutive_weak_threshold=5, hard_cutoff_rank=10000)
        else:
            print("ℹ️  Cat CSV doesn't exist. Need to create it first using the dynamic generator.")
            print("💡 Run: python process_semantic_rank_word_dynamic.py")
            return False
        
        if success_cat:
            print("✅ Cat CSV updated successfully!")
        else:
            print("❌ Cat CSV update failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n⏸️  Process interrupted for cat. Progress saved.")
        return False
    
    print("\n🎉 Both CSV files updated with dynamic cutoff system!")
    print("📁 Backup files created with '_backup' suffix")
    print("💰 Significant API cost savings achieved through dynamic cutoff")
    
    return True

if __name__ == "__main__":
    main()
