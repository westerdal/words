#!/usr/bin/env python3
"""
Test the actual production prompt with fish-related words
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_production_prompt():
    """Test the production prompt with actual fish-related words that are failing"""
    secret_word = "fish"
    
    # Test with the exact words that were failing in production
    failing_words = ["salmon", "tuna", "cod", "trout", "bass", "goldfish", "angelfish"]
    
    print(f"🧪 Testing PRODUCTION prompt with fish-related words")
    print(f"🐟 Secret word: '{secret_word}'")
    print(f"📝 Failing words: {failing_words}")
    print("=" * 70)
    
    # Use the EXACT production prompt from generate_csv.py
    prompt = (
        f"CRITICAL: You are a '{secret_word}' speaking about guess words in a word guessing game. "
        f"For each guess word, write a riddle from YOUR PERSPECTIVE (as '{secret_word}') describing your RELATIONSHIP to that word in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: You cannot say your own name '{secret_word}' in any clue. Use 'I', 'me', 'my' instead.\n\n"
        
        f"✅ GOOD EXAMPLES (YOU are the {secret_word} speaking about guess words):\n"
        f"• Guess: 'salmon' → Clue: 'I swim in the same waters as this' (shared habitat)\n"
        f"• Guess: 'tuna' → Clue: 'I am smaller than this swimmer' (comparison)\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold than this' (comparison)\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me' (describes relationship)\n"
        f"• Guess: 'telescope' → Clue: 'I have nothing to do with this' (no relationship)\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'I am a different type of fish' ← Contains forbidden word '{secret_word}'\n"
        f"• 'I am a popular fish species' ← Contains forbidden word '{secret_word}'\n"
        f"• 'Sea creature like a fish' ← Contains forbidden word '{secret_word}'\n"
        f"• 'Type of fish that swims' ← Contains forbidden word '{secret_word}'\n\n"
        
        f"📝 MANDATORY REQUIREMENTS (You are the {secret_word}):\n"
        f"• Describe YOUR relationship to each guess word, not what the guess word is\n"
        f"• Use 'I', 'me', 'my' to refer to yourself (the {secret_word})\n"
        f"• Use 'this', 'these', 'that' to refer to the guess word\n"
        f"• Focus on connections: shared category, usage, comparison, or lack thereof\n"
        f"• For unrelated words: 'I have nothing to do with this'\n"
        f"• Connection strength: 'strong', 'medium', 'weak'\n\n"
        
        f"REMEMBER: You are the {secret_word} talking about how you relate to guess words!\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {failing_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI with PRODUCTION prompt...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Response:\n{content}\n")
        
        try:
            results = json.loads(content)
            print("✅ RESULTS:")
            print("-" * 70)
            
            violations = []
            good_clues = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                status = "❌ VIOLATION" if violation else "✅ GOOD"
                
                print(f"{status} | {word:12} | {clue:35} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                else:
                    good_clues.append(f"'{word}' → '{clue}'")
            
            print("-" * 70)
            print(f"📊 RESULTS:")
            print(f"   Total words: {len(results)}")
            print(f"   ❌ Violations: {len(violations)}")
            print(f"   ✅ Good clues: {len(good_clues)}")
            
            if violations:
                print(f"\n🚨 VIOLATIONS:")
                for violation in violations:
                    print(f"   • {violation}")
            
            if good_clues:
                print(f"\n🎉 GOOD CLUES:")
                for good in good_clues:
                    print(f"   • {good}")
                    
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error: {e}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_production_prompt()
