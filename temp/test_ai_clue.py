#!/usr/bin/env python3
"""
Test the AI clue system with a specific guess and secret word
"""

import json
from openai import OpenAI

def test_ai_clue():
    """Test AI clue generation for leash + dog"""
    
    print("=== Testing AI Clue System ===")
    print("Guess: 'camp'")
    print("Secret word: 'dog'")
    print()
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
        return False
    
    # Test with our exact prompt format
    guess = "camp"
    secret_word = "dog"
    word_list = guess  # Single word for this test
    
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

    print("📝 Sending prompt to OpenAI:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    print()
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_content = response.choices[0].message.content
        print("🤖 OpenAI Response:")
        print(response_content)
        print()
        
        # Parse JSON
        try:
            clue_data = json.loads(response_content)
            print("✅ Parsed JSON successfully:")
            
            # Handle new format with clue and strength
            if "clue" in clue_data and "strength" in clue_data:
                clue = clue_data["clue"]
                strength = clue_data["strength"]
                
                print(f"  Word: '{guess}'")
                print(f"  Clue: '{clue}'")
                print(f"  Relationship strength: '{strength}'")
                
                # Count words in clue
                word_count = len(clue.split())
                print(f"  Word count: {word_count}")
                
                if word_count <= 7:
                    print(f"  ✅ Within 7-word limit")
                else:
                    print(f"  ⚠️  Exceeds 7-word limit")
                
                # Check if secret word is mentioned
                if secret_word.lower() in clue.lower():
                    print(f"  ❌ Contains secret word '{secret_word}'")
                else:
                    print(f"  ✅ Does not contain secret word")
                
                # Analyze relationship strength
                if strength == "weak":
                    print(f"  ⚠️  Weak relationship - consider stopping AI calls")
                elif strength == "medium":
                    print(f"  📊 Medium relationship - continue with caution")
                else:
                    print(f"  💪 Strong relationship - continue AI calls")
                
                print()
            else:
                # Fallback for old format
                for word, clue in clue_data.items():
                    print(f"  Word: '{word}'")
                    print(f"  Clue: '{clue}'")
                    print(f"  ⚠️  Old format - no relationship strength")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            return False
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def main():
    """Main function"""
    print("Testing AI clue generation system...")
    print("This will test how the AI generates a clue for 'camp' relating to 'dog'")
    print()
    
    success = test_ai_clue()
    
    if success:
        print("🎉 AI clue test completed successfully!")
    else:
        print("❌ AI clue test failed!")

if __name__ == "__main__":
    main()
