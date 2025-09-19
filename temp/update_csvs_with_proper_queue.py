#!/usr/bin/env python3
"""
Update cat and dog CSV files with proper weak connection queue system
This version correctly removes weak connections immediately and resequences ranks
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
from datetime import datetime

def update_csv_with_proper_queue(secret_word, consecutive_weak_threshold=5, hard_cutoff_rank=10000):
    """Update a CSV file with proper weak connection queue system"""
    
    print(f"\n=== Updating {secret_word.upper()} CSV with Proper Weak Queue ===")
    
    # File paths
    csv_file = f"secretword/secretword-easy-animals-{secret_word}.csv"
    backup_file = f"secretword/secretword-easy-animals-{secret_word}_backup.csv"
    temp_file = f"secretword/secretword-easy-animals-{secret_word}_temp.csv"
    queue_file = f"secretword/secretword-easy-animals-{secret_word}_weak_queue.json"
    
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
    original_count = len(df)
    print(f"✅ Loaded {original_count:,} rows")
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        use_ai = True
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI not available: {e}")
        return False
    
    # Initialize weak connection tracking
    weak_queue = []
    consecutive_weak_count = 0
    ai_cutoff_reached = False
    cutoff_rank = None
    total_ai_calls = 0
    words_processed = 0
    
    def get_clue_and_strength(guess_word, current_rank):
        """Get clue and relationship strength"""
        nonlocal consecutive_weak_count, ai_cutoff_reached, cutoff_rank, total_ai_calls
        
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
    
    def save_progress():
        """Save current progress"""
        df.to_csv(temp_file, index=False)
        
        # Save weak queue
        queue_data = {
            'secret_word': secret_word,
            'weak_connections': weak_queue,
            'metadata': {
                'total_queued': len(weak_queue),
                'words_processed': words_processed,
                'last_updated': datetime.now().isoformat()
            }
        }
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump(queue_data, f, indent=2)
    
    # Add connection_strength column if it doesn't exist
    if 'connection_strength' not in df.columns:
        df['connection_strength'] = ''
        print("✅ Added connection_strength column")
    
    print(f"🤖 Processing with proper weak connection queue system...")
    print(f"📊 Hard cutoff at rank {hard_cutoff_rank:,}")
    
    # Sort by rank and process - but work with a copy for iteration
    df = df.sort_values('rank').reset_index(drop=True)
    save_interval = 200
    
    # Process words starting from rank 2 (skip secret word)
    current_rank = 2
    i = 1  # Start from second row (skip secret word at index 0)
    
    while i < len(df) and current_rank <= hard_cutoff_rank and not ai_cutoff_reached:
        row = df.iloc[i]
        word = row['word']
        original_rank = row['rank']
        
        # Get AI assessment
        result = get_clue_and_strength(word, current_rank)
        words_processed += 1
        
        if result and result['strength'] == 'weak':
            # Queue this weak connection
            weak_entry = {
                'word': word,
                'original_rank': original_rank,
                'current_rank': current_rank,
                'clue': result['clue'],
                'strength': 'weak',
                'queued_at': datetime.now().isoformat()
            }
            weak_queue.append(weak_entry)
            
            print(f"    Rank {current_rank:>5}: {word:<15} → 🗂️  QUEUED (weak #{len(weak_queue)}) - REMOVING")
            
            # Remove this row from the DataFrame
            df = df.drop(df.index[i]).reset_index(drop=True)
            # Don't increment i since we removed a row
            # Don't increment current_rank since this position is now free
            
        else:
            # Keep this word, update its clue and rank
            df.at[i, 'rank'] = current_rank
            df.at[i, 'connection_strength'] = result.get('strength', 'unknown')
            
            if result['ai_used'] and result['clue']:
                df.at[i, 'clue'] = result['clue']
                print(f"    Rank {current_rank:>5}: {word:<15} → ✅ {result.get('strength', 'unknown')}")
            else:
                df.at[i, 'clue'] = None
                df.at[i, 'connection_strength'] = 'cutoff'
            
            i += 1
            current_rank += 1
        
        # Periodic save
        if words_processed % save_interval == 0:
            print(f"💾 Saving progress... Processed {words_processed} words, current rank {current_rank}")
            save_progress()
    
    # Handle remaining words (beyond cutoff)
    while i < len(df):
        df.at[i, 'rank'] = current_rank
        df.at[i, 'clue'] = None
        df.at[i, 'connection_strength'] = 'hard_cutoff'
        i += 1
        current_rank += 1
    
    # Insert weak connections at rank 10,000 or at the end
    if weak_queue:
        insertion_rank = min(hard_cutoff_rank, current_rank)
        print(f"\n📥 Inserting {len(weak_queue)} weak connections at rank {insertion_rank}")
        
        # Adjust existing ranks if needed
        mask = df['rank'] >= insertion_rank
        if mask.any():
            df.loc[mask, 'rank'] = df.loc[mask, 'rank'] + len(weak_queue)
        
        # Create weak connection rows
        weak_rows = []
        for i, weak_entry in enumerate(weak_queue):
            weak_rows.append({
                'rank': insertion_rank + i,
                'secret_word': secret_word,
                'word': weak_entry['word'],
                'clue': 'weak connection',
                'connection_strength': 'weak'
            })
        
        weak_df = pd.DataFrame(weak_rows)
        df = pd.concat([df, weak_df], ignore_index=True)
        df = df.sort_values('rank').reset_index(drop=True)
        
        print(f"✅ Inserted weak connections at ranks {insertion_rank}-{insertion_rank + len(weak_queue) - 1}")
    
    # Final save
    print(f"💾 Saving final CSV...")
    df.to_csv(csv_file, index=False)
    save_progress()
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Report results
    file_size = os.path.getsize(csv_file)
    final_count = len(df)
    ai_clues = len([c for c in df['clue'] if c and c not in ['This is the *.', 'ERROR', 'weak connection'] and pd.notna(c)])
    weak_connection_clues = len([c for c in df['clue'] if c == 'weak connection'])
    null_clues = len([c for c in df['clue'] if c is None or pd.isna(c)])
    
    print(f"\n📊 Update Results for {secret_word.upper()}:")
    print(f"   Original rows: {original_count:,}")
    print(f"   Final rows: {final_count:,}")
    print(f"   Words processed: {words_processed:,}")
    print(f"   Total AI calls: {total_ai_calls:,}")
    print(f"   AI clues: {ai_clues:,}")
    print(f"   Weak connections queued and reinserted: {len(weak_queue):,}")
    print(f"   NULL clues: {null_clues:,}")
    print(f"   File size: {file_size/1024/1024:.1f} MB")
    
    if cutoff_rank:
        print(f"💰 Dynamic cutoff at rank {cutoff_rank:,}")
    
    print(f"✅ {secret_word.upper()} CSV updated successfully with proper weak queue!")
    return True

def main():
    """Update CSV files with proper weak connection queue"""
    print("=== Updating CSVs with Proper Weak Connection Queue ===")
    print("This version correctly removes weak connections during processing")
    print("and resequences ranks in real-time.")
    print()
    
    # Process cat CSV (create if doesn't exist)
    if os.path.exists("secretword/secretword-easy-animals-cat.csv"):
        print("🐱 Processing cat CSV...")
        try:
            success_cat = update_csv_with_proper_queue('cat')
            if success_cat:
                print("✅ Cat CSV updated successfully!")
            else:
                print("❌ Cat CSV update failed!")
        except KeyboardInterrupt:
            print("\n⏸️  Cat processing interrupted. Progress saved.")
    else:
        print("ℹ️  Cat CSV doesn't exist yet. Skipping for now.")
    
    print("\n🎉 CSV processing complete!")

if __name__ == "__main__":
    main()
