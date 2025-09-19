#!/usr/bin/env python3
"""
Process semantic rank words following the complete standalone prompt
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import math
from tqdm import tqdm

def is_plural(word):
    """
    Determines if a given word is likely plural based on a comprehensive
    set of English pluralization rules.

    Args:
      word: A string representing the word to check.

    Returns:
      True if the word is likely plural, False otherwise.
    """
    # Convert word to lowercase for case-insensitive checks
    w_lower = word.lower()

    # Rule 5: Check against a set of common irregular plurals
    irregular_plurals = {"men", "women", "children", "feet", "teeth", "mice", "people"}
    if w_lower in irregular_plurals:
        return True

    # Rule 6: Check against nouns where singular and plural forms are the same
    no_change_nouns = {"sheep", "deer", "fish", "series", "species"}
    if w_lower in no_change_nouns:
        return True

    # Rule 4: Check for words ending in "-ves" (e.g., wives, leaves)
    if w_lower.endswith("ves"):
        return True
        
    # Rule 3: Check for words ending in "-ies" (e.g., babies, parties)
    if w_lower.endswith("ies"):
        return True
        
    # Rule 2: Check for words ending in "-es" (e.g., boxes, buses)
    if w_lower.endswith("es"):
        return True
        
    # Rule 1: Check for the most common "-s" ending, excluding common singulars
    # to reduce false positives (e.g., "status", "bus", "class").
    if w_lower.endswith("s") and not w_lower.endswith(("ss", "us", "is")):
        return True

    # If none of the plural rules match, assume the word is singular.
    return False

def process_first_unprocessed_word():
    """Process the first unprocessed word from master-list.txt"""
    
    print("=== Semantic Rank Word Processing ===")
    
    # Step 1: Read master-list.txt - Get first unprocessed word
    master_list_path = "secretword/master-list.txt"
    
    try:
        with open(master_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("❌ Master list is empty")
            return False
            
        print(f"📋 Found {len(lines)} words in master list")
        
    except FileNotFoundError:
        print(f"❌ Master list not found: {master_list_path}")
        return False
    
    # Find first unprocessed word
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Step 2: Parse format - Extract difficulty, category, word
        try:
            parts = line.split('-')
            if len(parts) != 3:
                print(f"⚠️  Skipping invalid format: {line}")
                continue
                
            difficulty, category, word = parts
            print(f"📝 Processing word {i+1}: {word} ({difficulty}, {category})")
            
        except Exception as e:
            print(f"⚠️  Error parsing line '{line}': {e}")
            continue
        
        # Step 3: Check if exists - Skip if CSV already exists
        csv_file = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
        if os.path.exists(csv_file):
            print(f"✅ Already exists: {csv_file} - skipping")
            continue
        
        # Found first unprocessed word - process it
        print(f"🎯 Processing: {word}")
        success = process_single_word(difficulty, category, word)
        
        if success:
            print(f"✅ Successfully processed '{word}'")
            return True
        else:
            print(f"❌ Failed to process '{word}'")
            return False
    
    print("✅ All words in master list have been processed")
    return True

def process_single_word(difficulty, category, word):
    """Process a single word to create its CSV file"""
    
    print(f"\n--- Processing {word} ---")
    
    # Step 4: Load word list - Read all 172,823 words from data/enable1.txt
    print("📚 Loading ENABLE word list...")
    try:
        with open("data/enable1.txt", 'r', encoding='utf-8') as f:
            all_words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(all_words):,} words from ENABLE list")
        
        # Filter out plural words
        print("🔍 Filtering out plural words...")
        enable_words = []
        plural_count = 0
        
        for enable_word in all_words:
            if is_plural(enable_word):
                plural_count += 1
            else:
                enable_words.append(enable_word)
        
        print(f"✅ Filtered to {len(enable_words):,} singular words ({plural_count:,} plurals removed)")
        
        if word not in enable_words:
            print(f"⚠️  Warning: '{word}' not found in filtered word list")
            if word in all_words:
                if is_plural(word):
                    print(f"❌ Error: Secret word '{word}' is detected as plural!")
                    return False
            
    except FileNotFoundError:
        print("❌ ENABLE word list not found at data/enable1.txt")
        return False
    
    # Step 5: Compute rankings - Use .env/embeddings.json
    print("🧮 Loading embeddings and computing semantic similarity rankings...")
    
    try:
        # Load embeddings
        with open(".env/embeddings.json", 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        print(f"✅ Loaded embeddings for {len(embeddings):,} words")
        
    except FileNotFoundError:
        print("❌ Embeddings file not found at .env/embeddings.json")
        return False
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Check if secret word is in embeddings
    if word not in embeddings:
        print(f"❌ '{word}' not found in embeddings")
        return False
    
    # Get secret word embedding and normalize
    secret_embedding = np.array(embeddings[word])
    secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
    
    # Compute similarities for all ENABLE words
    print(f"🔍 Computing similarities for {len(enable_words):,} words...")
    word_similarities = []
    
    for enable_word in tqdm(enable_words, desc="Computing similarities"):
        if enable_word in embeddings:
            # Get cached embedding
            word_embedding = np.array(embeddings[enable_word])
            word_embedding = word_embedding / np.linalg.norm(word_embedding)
            
            # Compute cosine similarity
            similarity = np.dot(word_embedding, secret_embedding)
            word_similarities.append((enable_word, similarity))
        else:
            # Word not in cache - assign very low similarity
            word_similarities.append((enable_word, -1.0))
    
    # Sort by similarity (descending), then alphabetically for ties
    word_similarities.sort(key=lambda x: (-x[1], x[0]))
    
    # Create rankings dictionary
    rankings = {}
    for rank, (ranked_word, similarity) in enumerate(word_similarities, 1):
        rankings[ranked_word] = {'rank': rank, 'similarity': similarity}
    
    print(f"✅ Computed rankings for {len(rankings):,} words")
    print(f"🎯 '{word}' has rank: {rankings[word]['rank']}")
    
    # Show top 10 for verification
    top_10 = word_similarities[:10]
    print(f"\nTop 10 most similar words to '{word}':")
    for i, (w, sim) in enumerate(top_10, 1):
        print(f"  {i:2d}. {w:<15} (similarity: {sim:.6f})")
    
    # Step 6: Generate clues - Create AI clues for ranks 1-10,000
    print(f"\n🤖 Generating AI clues for ranks 1-10,000...")
    
    # Get words that need AI clues
    ai_words = [w for w, data in rankings.items() if data['rank'] <= 10000]
    ai_words.sort(key=lambda x: rankings[x]['rank'])  # Sort by rank
    
    print(f"📝 Need to generate {len(ai_words):,} AI clues")
    
    # Generate clues (simplified for this implementation)
    clues = {}
    
    # Special case for secret word (rank 1)
    clues[word] = "This is the *."
    
    # For other words, use OpenAI if available
    try:
        client = OpenAI()
        print("✅ OpenAI client initialized")
        
        # Generate clues in batches
        batch_size = 50
        successful_clues = 1  # Count the secret word clue
        
        for i in tqdm(range(1, len(ai_words), batch_size), desc="Generating AI clues"):
            batch_words = ai_words[i:i + batch_size]  # Skip secret word (index 0)
            
            try:
                # Create prompt for batch
                word_list = ", ".join(batch_words)
                prompt = f"""For each word below, describe the SPECIFIC relationship between the word and '{word}' in 7 words or less. Focus on HOW they connect, not what the word is. Don't mention '{word}' directly - use 'that animal/creature/thing' instead. Return JSON format.

Words: {word_list}

Example format: {{"puppy": "young offspring of that animal", "collar": "restrains and guides that animal", "bone": "that animal buries and chews it"}}"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                batch_clues = json.loads(response.choices[0].message.content)
                
                for batch_word in batch_words:
                    if batch_word in batch_clues:
                        clues[batch_word] = batch_clues[batch_word]
                        successful_clues += 1
                    else:
                        clues[batch_word] = "ERROR"
                        successful_clues += 1
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️  Batch clue generation failed: {e}")
                # Fallback clues for this batch
                for batch_word in batch_words:
                    clues[batch_word] = "ERROR"
                    successful_clues += 1
        
        print(f"✅ Generated {successful_clues:,} clues")
        
    except Exception as e:
        print(f"⚠️  OpenAI not available: {e}")
        print("📝 Using fallback clues...")
        
        # Fallback: error clues for all words
        for ai_word in ai_words[1:]:  # Skip secret word
            clues[ai_word] = "ERROR"
    
    # Step 7: Create CSV - Save with proper filename and format
    print(f"\n📄 Creating CSV file...")
    
    csv_data = []
    for enable_word in enable_words:
        rank_info = rankings.get(enable_word, {'rank': len(enable_words) + 1})
        rank = rank_info['rank']
        
        # Determine clue based on rank
        if rank == 1:
            clue = "This is the *."
        elif rank <= 10000:
            clue = clues.get(enable_word, "ERROR")
        else:
            clue = None  # NULL for ranks 10,001+
        
        csv_data.append({
            'rank': rank,
            'secret_word': word,
            'word': enable_word,
            'clue': clue
        })
    
    # Create DataFrame and sort by rank
    df = pd.DataFrame(csv_data)
    df.sort_values(by='rank', inplace=True)
    
    # Save CSV
    csv_filename = f"secretword/secretword-{difficulty}-{category}-{word}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    
    # Report results
    file_size = os.path.getsize(csv_filename)
    print(f"✅ CSV created: {csv_filename}")
    print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"📊 Total rows: {len(df):,}")
    
    # Show sample rows
    print(f"\nSample CSV content:")
    print(df.head(10).to_string(index=False))
    
    return True

def main():
    """Main function"""
    print("Starting Semantic Rank word processing...")
    success = process_first_unprocessed_word()
    
    if success:
        print("\n🎉 Processing completed successfully!")
    else:
        print("\n❌ Processing failed!")
    
    return success

if __name__ == "__main__":
    main()
