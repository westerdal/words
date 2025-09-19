#!/usr/bin/env python3
"""
Ask OpenAI for the top 200 words most similar in meaning to "forest"
"""

import os
import openai
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_forest_similar_words():
    """Get top 200 words similar to forest from OpenAI"""
    
    prompt = """
    Please provide the top 200 words that are most similar in meaning to "forest". 
    
    Include words that are:
    - Synonyms and near-synonyms
    - Related natural environments and ecosystems
    - Types of forests and woodlands
    - Trees and plant life associated with forests
    - Animals commonly found in forests
    - Forest-related activities and concepts
    - Geographic features found in or near forests
    
    Return the words as a simple numbered list from 1-200, ordered by semantic similarity to "forest" (most similar first).
    
    Format:
    1. woods
    2. woodland
    3. trees
    ... etc
    """
    
    try:
        print("🤖 Asking OpenAI for top 200 words similar to 'forest'...")
        
        response = openai.chat.completions.create(
            model="gpt-4",  # Use GPT-4 for better quality
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        content = response.choices[0].message.content.strip()
        
        print("✅ Received response from OpenAI")
        print("\n" + "="*60)
        print("TOP 200 WORDS SIMILAR TO 'FOREST' (according to OpenAI)")
        print("="*60)
        print(content)
        print("="*60)
        
        # Save to file
        with open("openai_forest_similar_words.txt", "w", encoding="utf-8") as f:
            f.write("TOP 200 WORDS SIMILAR TO 'FOREST' (according to OpenAI)\n")
            f.write("="*60 + "\n\n")
            f.write(content)
        
        print(f"\n💾 Results saved to: openai_forest_similar_words.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    get_forest_similar_words()
