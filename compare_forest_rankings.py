#!/usr/bin/env python3
"""
Compare OpenAI's suggested forest-similar words with our semantic rankings
"""

import csv
from pathlib import Path

def load_our_rankings():
    """Load our semantic rankings for forest"""
    rankings = {}
    csv_file = "secretword/secretword-easy-animals-forest.csv"
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['word'].lower()
                rank = int(row['rank'])
                rankings[word] = rank
        
        print(f"✅ Loaded {len(rankings):,} words from our semantic rankings")
        return rankings
        
    except Exception as e:
        print(f"❌ Error loading our rankings: {e}")
        return {}

def parse_openai_words():
    """Parse OpenAI's suggested words"""
    openai_words = []
    
    # OpenAI's top suggestions (from the output above)
    openai_suggestions = [
        "woods", "woodland", "trees", "jungle", "rainforest", "wilderness", "timberland", 
        "bush", "grove", "thicket", "greenery", "copse", "undergrowth", "foliage", 
        "vegetation", "flora", "fauna", "wildlife", "ecosystem", "habitat", "biodiversity", 
        "nature", "environment", "canopy", "deciduous", "coniferous", "evergreen", "pine", 
        "oak", "maple", "birch", "cedar", "sequoia", "redwood", "fir", "spruce", "ash", 
        "elm", "beech", "poplar", "willow", "cypress", "sycamore", "alder", "chestnut", 
        "larch", "teak", "mahogany", "ebony", "hickory", "mangrove", "bamboo", "ferns", 
        "moss", "lichen", "mushrooms", "fungi", "ivy", "vines", "understory", "shrubbery", 
        "brush", "bramble", "thorns", "flowers", "blossoms", "berries", "nuts", "seeds", 
        "cones", "leaves", "bark", "roots", "branches", "trunks", "stumps", "logs", 
        "lumber", "timber", "firewood", "charcoal", "sawdust", "pulp", "paper", "furniture", 
        "woodwork", "carpentry", "forestry", "logging", "conservation", "preservation", 
        "reforestation", "deforestation", "clearcutting", "planting", "growth", "regeneration", 
        "sustainability", "climate", "rainfall", "seasons", "sunlight", "shade", "soil", 
        "compost", "mulch", "nutrients", "water", "streams", "rivers", "ponds", "lakes", 
        "swamps", "marshes", "wetlands", "bogs", "fens", "peat", "mud", "silt", "rocks", 
        "stones", "boulders", "cliffs", "hills", "mountains", "valleys", "ravines", "gorges", 
        "caves", "paths", "trails", "hiking", "camping", "picnicking", "birdwatching", 
        "hunting", "fishing", "foraging", "survival", "adventure", "exploration", "solitude", 
        "peace", "serenity", "beauty", "scenery", "landscape", "view", "sunset", "sunrise", 
        "dusk", "dawn", "twilight", "night", "day", "shadows", "echoes", "silence", "sounds", 
        "birdsong", "howls", "roars", "rustling", "wind", "rain", "snow", "fog", "mist", 
        "dew", "frost", "ice", "fire", "smoke", "ashes", "char", "burn", "heat", "cold", 
        "warmth", "chill", "humidity", "dryness", "wetness", "dampness", "freshness", 
        "aroma", "scent", "fragrance", "smell", "taste", "touch", "feel", "sight", "vision", 
        "perception", "experience", "memory", "dream", "fantasy"
    ]
    
    return openai_suggestions

def compare_rankings():
    """Compare OpenAI suggestions with our semantic rankings"""
    
    our_rankings = load_our_rankings()
    openai_words = parse_openai_words()
    
    if not our_rankings:
        return
    
    print(f"\n🔍 COMPARISON: OpenAI Top 200 vs Our Semantic Rankings")
    print("=" * 80)
    
    found_in_our_data = 0
    not_found = []
    comparison_data = []
    
    for i, word in enumerate(openai_words, 1):
        word_lower = word.lower()
        
        if word_lower in our_rankings:
            our_rank = our_rankings[word_lower]
            found_in_our_data += 1
            comparison_data.append((i, word, our_rank))
            
            # Show significant discrepancies
            if our_rank > 1000:  # OpenAI thinks it's top 200, but we ranked it >1000
                print(f"⚠️  DISCREPANCY: '{word}' - OpenAI #{i}, Our rank #{our_rank:,}")
        else:
            not_found.append((i, word))
    
    print(f"\n📊 SUMMARY:")
    print(f"   OpenAI suggestions found in our data: {found_in_our_data}/200 ({found_in_our_data/200*100:.1f}%)")
    print(f"   Not found in our ENABLE2 word list: {len(not_found)}")
    
    # Show top matches
    print(f"\n✅ TOP 20 MATCHES (OpenAI rank vs Our rank):")
    comparison_data.sort(key=lambda x: x[0])  # Sort by OpenAI rank
    for i, (openai_rank, word, our_rank) in enumerate(comparison_data[:20]):
        status = "✅" if our_rank <= 100 else "⚠️" if our_rank <= 1000 else "❌"
        print(f"   {status} {openai_rank:>3}. {word:<15} → Our rank: {our_rank:>5,}")
    
    # Show words not found
    if not_found:
        print(f"\n❌ WORDS NOT IN ENABLE2 LIST (first 10):")
        for openai_rank, word in not_found[:10]:
            print(f"   {openai_rank:>3}. {word}")
        if len(not_found) > 10:
            print(f"   ... and {len(not_found) - 10} more")
    
    # Show our top 20 for comparison
    print(f"\n🏆 OUR TOP 20 SEMANTIC RANKINGS:")
    our_top_20 = [(word, rank) for word, rank in our_rankings.items() if rank <= 20]
    our_top_20.sort(key=lambda x: x[1])
    
    for word, rank in our_top_20:
        # Check if this word was in OpenAI's top 200
        word_in_openai = word.lower() in [w.lower() for w in openai_words]
        status = "🤝" if word_in_openai else "🔍"
        print(f"   {status} {rank:>2}. {word}")
    
    print(f"\n🤝 = Also in OpenAI top 200")
    print(f"🔍 = Not in OpenAI top 200")

if __name__ == "__main__":
    compare_rankings()
