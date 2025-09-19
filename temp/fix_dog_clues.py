#!/usr/bin/env python3
"""
Fix the existing dog CSV file by adding AI-generated clues for ranks 1-10,000
"""

import pandas as pd
import json
import time
from openai import OpenAI
from tqdm import tqdm

def fix_dog_clues():
    """Add AI clues to the existing dog CSV file"""
    
    print("=== Fixing Dog CSV with AI Clues ===")
    
    # Load the existing CSV
    csv_file = "secretword/secretword-easy-animals-dog.csv"
    print(f"📄 Loading existing CSV: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df):,} rows")
        print(f"📊 Current structure: {list(df.columns)}")
        
        # Show current state
        print(f"\nCurrent clue status:")
        total_rows = len(df)
        null_clues = df['clue'].isnull().sum()
        non_null_clues = total_rows - null_clues
        print(f"  Non-null clues: {non_null_clues:,}")
        print(f"  NULL clues: {null_clues:,}")
        
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file}")
        return False
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        return False
    
    # Get words that need AI clues (ranks 2-10,000, excluding rank 1 which should be "This is the *.")
    words_needing_clues = df[(df['rank'] >= 2) & (df['rank'] <= 10000)].copy()
    words_needing_clues = words_needing_clues.sort_values('rank')
    
    print(f"\n🤖 Generating AI clues for ranks 2-10,000...")
    print(f"📝 Need to generate clues for {len(words_needing_clues):,} words")
    
    # Estimate API calls
    batch_size = 50
    estimated_batches = len(words_needing_clues) // batch_size + (1 if len(words_needing_clues) % batch_size > 0 else 0)
    print(f"📞 Estimated API calls: ~{estimated_batches:,} batches")
    
    # Generate clues in batches
    successful_clues = 0
    failed_clues = 0
    
    for i in tqdm(range(0, len(words_needing_clues), batch_size), desc="Generating AI clues"):
        batch_df = words_needing_clues.iloc[i:i + batch_size]
        batch_words = batch_df['word'].tolist()
        
        try:
            # Create prompt for batch
            word_list = ", ".join(batch_words)
            prompt = f"""For each word below, describe the SPECIFIC relationship between the word and 'dog' in 7 words or less. Focus on HOW they connect, not what the word is. Don't mention 'dog' directly - use 'that animal/creature/thing' instead. Return JSON format.

Words: {word_list}

Example format: {{"puppy": "young offspring of that animal", "collar": "restrains and guides that animal", "bone": "that animal buries and chews it"}}"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            batch_clues = json.loads(response.choices[0].message.content)
            
            # Update the DataFrame with generated clues
            for word in batch_words:
                if word in batch_clues:
                    # Find the row and update the clue
                    mask = (df['word'] == word) & (df['secret_word'] == 'dog')
                    if mask.any():
                        df.loc[mask, 'clue'] = batch_clues[word]
                        successful_clues += 1
                else:
                    # Fallback clue
                    mask = (df['word'] == word) & (df['secret_word'] == 'dog')
                    if mask.any():
                        df.loc[mask, 'clue'] = "ERROR"
                        failed_clues += 1
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n⚠️  Batch failed: {e}")
            # Fallback clues for this batch
            for word in batch_words:
                mask = (df['word'] == word) & (df['secret_word'] == 'dog')
                if mask.any():
                    df.loc[mask, 'clue'] = "ERROR"
                    failed_clues += 1
    
    print(f"\n✅ AI clue generation complete!")
    print(f"   Successful AI clues: {successful_clues:,}")
    print(f"   Fallback clues: {failed_clues:,}")
    
    # Ensure rank 1 has the correct clue
    rank_1_mask = (df['rank'] == 1) & (df['secret_word'] == 'dog')
    df.loc[rank_1_mask, 'clue'] = "This is the *."
    
    # Ensure ranks > 10,000 have NULL clues
    high_rank_mask = df['rank'] > 10000
    df.loc[high_rank_mask, 'clue'] = None
    
    # Save the updated CSV
    print(f"\n💾 Saving updated CSV...")
    try:
        df.to_csv(csv_file, index=False)
        
        # Report final statistics
        final_null_clues = df['clue'].isnull().sum()
        final_non_null_clues = len(df) - final_null_clues
        
        print(f"✅ Updated CSV saved: {csv_file}")
        print(f"📊 Final clue statistics:")
        print(f"   Total rows: {len(df):,}")
        print(f"   Non-null clues: {final_non_null_clues:,} (ranks 1-10,000)")
        print(f"   NULL clues: {final_null_clues:,} (ranks 10,001+)")
        
        # Show sample of updated clues
        print(f"\nSample of updated clues (ranks 2-11):")
        sample = df[(df['rank'] >= 2) & (df['rank'] <= 11)][['rank', 'word', 'clue']]
        print(sample.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving updated CSV: {e}")
        return False

def main():
    """Main function"""
    success = fix_dog_clues()
    
    if success:
        print("\n🎉 Dog CSV successfully updated with AI clues!")
    else:
        print("\n❌ Failed to update dog CSV!")
    
    return success

if __name__ == "__main__":
    main()
