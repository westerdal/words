#!/usr/bin/env python3
"""
Analyze fish embeddings to identify words that contain the secret word
"""

def analyze_embeddings_for_secret_word_contamination(secret_word, embeddings_file):
    """Analyze embeddings file for words containing the secret word"""
    print(f"🔍 Analyzing {embeddings_file} for '{secret_word}' contamination")
    
    contaminated_words = []
    total_words = 0
    
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()
            
            for line_num, line in enumerate(f, 2):  # Start from line 2
                if not line.strip():
                    continue
                
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                
                rank = parts[0]
                word = parts[1]
                similarity = parts[2]
                total_words += 1
                
                # Check if word contains secret word
                if secret_word.lower() in word.lower():
                    contaminated_words.append({
                        'rank': rank,
                        'word': word,
                        'similarity': similarity
                    })
                
                # Stop after checking first 1000 words (where contamination is most problematic)
                if total_words >= 1000:
                    break
    
    except FileNotFoundError:
        print(f"❌ File not found: {embeddings_file}")
        return
    
    print(f"📊 Analysis Results:")
    print(f"   Total words checked: {total_words:,}")
    print(f"   Contaminated words: {len(contaminated_words)}")
    
    if contaminated_words:
        print(f"\n⚠️ CONTAMINATED WORDS (top 20):")
        for i, word_data in enumerate(contaminated_words[:20]):
            print(f"   {word_data['rank']:>3}: {word_data['word']:<15} (similarity: {word_data['similarity']})")
        
        if len(contaminated_words) > 20:
            print(f"   ... and {len(contaminated_words) - 20} more")
    else:
        print("✅ No contaminated words found in top 1000!")
    
    return contaminated_words

if __name__ == "__main__":
    secret_word = "fish"
    
    print("=" * 60)
    print("🐟 FISH EMBEDDINGS CONTAMINATION ANALYSIS")
    print("=" * 60)
    
    # Check both standard and enhanced embeddings
    files_to_check = [
        "secretword/embeddings-fish.txt",
        "secretword/embeddings-fish2.txt"
    ]
    
    for file_path in files_to_check:
        print(f"\n{'-' * 40}")
        contaminated = analyze_embeddings_for_secret_word_contamination(secret_word, file_path)
        
        if contaminated:
            print(f"\n💡 RECOMMENDATION: Filter out these {len(contaminated)} contaminated words")
            print(f"   They will likely generate clues containing '{secret_word}'")
