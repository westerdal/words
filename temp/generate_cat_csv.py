#!/usr/bin/env python3
"""
Generate semantic rank CSV for the word "cat"
"""

import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm

def generate_cat_csv():
    """Generate the CSV file for secret word 'cat'"""
    
    print("=== Generating CSV for Secret Word: CAT ===")
    
    secret_word = "cat"
    difficulty = "easy"
    category = "animals"
    
    # Check if CSV already exists
    csv_filename = f"secretword/secretword-{difficulty}-{category}-{secret_word}.csv"
    if os.path.exists(csv_filename):
        print(f"✅ CSV already exists: {csv_filename}")
        return True
    
    # Load ENABLE2 word list
    print("📚 Loading ENABLE2 word list...")
    try:
        with open("data/enable2.txt", 'r', encoding='utf-8') as f:
            enable2_words = [w.strip().lower() for w in f.readlines()]
        print(f"✅ Loaded {len(enable2_words):,} singular words from ENABLE2")
        
        if secret_word not in enable2_words:
            print(f"❌ Secret word '{secret_word}' not found in ENABLE2 word list")
            return False
            
    except FileNotFoundError:
        print("❌ ENABLE2 word list not found at data/enable2.txt")
        return False
    
    # Load embeddings
    print("🧮 Loading embeddings and computing semantic similarity rankings...")
    try:
        with open(".env/embeddings2.json", 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        print(f"✅ Loaded embeddings for {len(embeddings):,} words")
    except FileNotFoundError:
        print("❌ Embeddings file not found at .env/embeddings2.json")
        return False
    
    # Check if secret word has embedding
    if secret_word not in embeddings:
        print(f"❌ '{secret_word}' not found in embeddings")
        return False
    
    # Get secret word embedding and normalize
    secret_embedding = np.array(embeddings[secret_word])
    secret_embedding = secret_embedding / np.linalg.norm(secret_embedding)
    
    print(f"🔍 Computing similarities for {len(enable2_words):,} words...")
    word_similarities = []
    
    for word in tqdm(enable2_words, desc="Computing similarities"):
        if word in embeddings:
            word_embedding = np.array(embeddings[word])
            word_embedding = word_embedding / np.linalg.norm(word_embedding)
            similarity = np.dot(word_embedding, secret_embedding)
            word_similarities.append((word, similarity))
        else:
            word_similarities.append((word, -1.0))
    
    # Sort by similarity (descending), then alphabetically for ties
    word_similarities.sort(key=lambda x: (-x[1], x[0]))
    
    # Create rankings dictionary
    rankings = {}
    for rank, (word, similarity) in enumerate(word_similarities, 1):
        rankings[word] = {'rank': rank, 'similarity': similarity}
    
    print(f"✅ Computed rankings for {len(rankings):,} words")
    print(f"🎯 '{secret_word}' has rank: {rankings[secret_word]['rank']}")
    
    # Show top 10 for verification
    top_10 = word_similarities[:10]
    print(f"\nTop 10 most similar words to '{secret_word}':")
    for i, (w, sim) in enumerate(top_10, 1):
        print(f"  {i:2d}. {w:<15} (similarity: {sim:.6f})")
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        print("✅ OpenAI client initialized")
        use_ai = True
    except Exception as e:
        print(f"⚠️  OpenAI initialization failed: {e}")
        use_ai = False
    
    # Generate clues for ranks 2-10,000
    clues = {}
    clues[secret_word] = "This is the *."  # Rank 1
    
    if use_ai:
        print(f"\n🤖 Generating AI clues for ranks 2-10,000...")
        
        # Get words that need AI clues
        ai_words = [w for w, data in rankings.items() if 2 <= data['rank'] <= 10000]
        ai_words.sort(key=lambda x: rankings[x]['rank'])
        
        print(f"📝 Need to generate clues for {len(ai_words):,} words")
        
        batch_size = 50
        successful_clues = 1  # Count the secret word clue
        failed_clues = 0
        
        for i in tqdm(range(0, len(ai_words), batch_size), desc="Generating AI clues"):
            batch_words = ai_words[i:i + batch_size]
            
            try:
                word_list = ", ".join(batch_words)
                prompt = f"""For each word below, describe the SPECIFIC relationship between the word and '{secret_word}' in 7 words or less. Focus on HOW they connect, not what the word is. Don't mention '{secret_word}' directly - use 'that animal/creature/thing' instead. Return JSON format.

Words: {word_list}

Example format: {{"kitten": "young offspring of that animal", "collar": "restrains and guides that animal", "litter": "box used by that animal"}}"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                batch_clues = json.loads(response.choices[0].message.content)
                
                for word in batch_words:
                    if word in batch_clues:
                        clues[word] = batch_clues[word]
                        successful_clues += 1
                    else:
                        clues[word] = "ERROR"
                        failed_clues += 1
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"\n⚠️  Batch failed: {e}")
                for word in batch_words:
                    clues[word] = "ERROR"
                    failed_clues += 1
        
        print(f"✅ Generated {successful_clues:,} successful clues, {failed_clues:,} errors")
    
    else:
        print("⚠️  Using ERROR clues for all words (no AI available)")
        for word in enable2_words:
            if rankings[word]['rank'] == 1:
                clues[word] = "This is the *."
            elif rankings[word]['rank'] <= 10000:
                clues[word] = "ERROR"
    
    # Create CSV data
    print(f"\n📄 Creating CSV file...")
    csv_data = []
    
    for word in enable2_words:
        rank_info = rankings.get(word, {'rank': len(enable2_words) + 1})
        rank = rank_info['rank']
        
        # Determine clue based on rank
        if rank == 1:
            clue = "This is the *."
        elif rank <= 10000:
            clue = clues.get(word, "ERROR")
        else:
            clue = None  # NULL for ranks 10,001+
        
        csv_data.append({
            'rank': rank,
            'secret_word': secret_word,
            'word': word,
            'clue': clue
        })
    
    # Create DataFrame and sort by rank
    df = pd.DataFrame(csv_data)
    df.sort_values(by='rank', inplace=True)
    
    # Save CSV
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    
    # Report results
    file_size = os.path.getsize(csv_filename)
    print(f"✅ CSV created: {csv_filename}")
    print(f"📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"📊 Total rows: {len(df):,}")
    
    # Show sample rows
    print(f"\nSample CSV content (ranks 1-10):")
    sample = df.head(10)[['rank', 'word', 'clue']]
    print(sample.to_string(index=False))
    
    return True

def main():
    """Main function"""
    success = generate_cat_csv()
    
    if success:
        print("\n🎉 Cat CSV generated successfully!")
    else:
        print("\n❌ Cat CSV generation failed!")
    
    return success

if __name__ == "__main__":
    main()
