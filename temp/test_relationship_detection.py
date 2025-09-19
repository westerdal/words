#!/usr/bin/env python3
"""
Test relationship strength detection across different word distances
"""

import json
from openai import OpenAI

def test_word_relationship(guess, secret_word):
    """Test AI relationship strength detection"""
    
    try:
        client = OpenAI()
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
        return None
    
    prompt = f"""You must analyze the word '{guess}' and its relationship to '{secret_word}'. 

Return JSON with TWO fields:
1. "clue": A 7-word-or-less description of how they relate (use 'that animal/creature' instead of '{secret_word}')
2. "strength": Rate the relationship strength as "strong", "medium", or "weak"

Relationship strength guide:
- "strong": Direct interaction, clear connection (leash, collar, bone, walk)
- "medium": Indirect connection, some logical link (tree, park, house)  
- "weak": Very distant, forced connection, opposite concepts (calculator, mathematics, philosophy)

Word to analyze: {guess}

Example format:
{{"clue": "connects to that animal for control", "strength": "strong"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        clue_data = json.loads(response_content)
        
        if "clue" in clue_data and "strength" in clue_data:
            return {
                "word": guess,
                "clue": clue_data["clue"],
                "strength": clue_data["strength"],
                "word_count": len(clue_data["clue"].split())
            }
        else:
            return None
            
    except Exception as e:
        print(f"❌ API call failed for '{guess}': {e}")
        return None

def main():
    """Test multiple words across relationship strengths"""
    
    secret_word = "dog"
    
    # Test words across different expected relationship strengths
    test_words = [
        # Expected strong relationships
        ("leash", "strong"),
        ("bone", "strong"), 
        ("collar", "strong"),
        
        # Expected medium relationships  
        ("tree", "medium"),
        ("park", "medium"),
        ("camp", "medium"),
        
        # Expected weak relationships
        ("calculator", "weak"),
        ("mathematics", "weak"),
        ("philosophy", "weak")
    ]
    
    print("=== Relationship Strength Detection Test ===")
    print(f"Secret word: '{secret_word}'")
    print()
    
    results = []
    
    for word, expected_strength in test_words:
        print(f"Testing: '{word}' (expected: {expected_strength})")
        
        result = test_word_relationship(word, secret_word)
        
        if result:
            actual_strength = result["strength"]
            clue = result["clue"]
            word_count = result["word_count"]
            
            # Check if prediction matches expectation
            match = "✅" if actual_strength == expected_strength else "⚠️"
            
            print(f"  {match} Strength: {actual_strength} (expected: {expected_strength})")
            print(f"  📝 Clue: '{clue}' ({word_count} words)")
            
            results.append({
                "word": word,
                "expected": expected_strength,
                "actual": actual_strength,
                "clue": clue,
                "match": actual_strength == expected_strength
            })
        else:
            print(f"  ❌ Failed to get result")
            
        print()
    
    # Summary
    print("=== Summary ===")
    correct_predictions = sum(1 for r in results if r["match"])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
    
    # Show strength distribution
    strong_count = sum(1 for r in results if r["actual"] == "strong")
    medium_count = sum(1 for r in results if r["actual"] == "medium")
    weak_count = sum(1 for r in results if r["actual"] == "weak")
    
    print(f"Actual distribution:")
    print(f"  Strong: {strong_count}")
    print(f"  Medium: {medium_count}")
    print(f"  Weak: {weak_count}")
    
    # Dynamic cutoff suggestion
    print()
    print("=== Dynamic AI Cutoff Suggestion ===")
    print("When we see 3+ consecutive 'weak' relationships:")
    print("  → Stop making AI calls")
    print("  → Switch to NULL clues")
    print("  → Save API costs and processing time")

if __name__ == "__main__":
    main()
