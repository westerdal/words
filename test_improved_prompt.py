#!/usr/bin/env python3
"""
Test the improved AI prompt with fish-related words
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_fish_prompt():
    """Test the improved prompt with challenging fish-related words"""
    secret_word = "fish"
    
    # Test with words that contain "fish" - these should NOT have "fish" in their clues
    test_words = ["starfish", "goldfish", "catfish", "fishing", "fisherman", "swordfish"]
    
    print(f"🧪 Testing improved prompt for secret word: '{secret_word}'")
    print(f"📝 Test words: {test_words}")
    print("=" * 60)
    
    # Create the improved prompt
    prompt = (
        f"You are generating clues for a word guessing game. The secret word is '{secret_word}'. "
        f"For each guess word below, write a clue describing the relationship in 7 words or less.\n\n"
        
        f"🚫 CRITICAL RULE: NEVER use '{secret_word}' in any clue, even if the guess word contains it.\n\n"
        
        f"✅ GOOD EXAMPLES (riddles from secret word's perspective):\n"
        f"• Guess: 'starfish' → Clue: 'I am a type of these'\n"
        f"• Guess: 'catfish' → Clue: 'I have whiskers but I am also one of these'\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me'\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold'\n\n"
        
        f"❌ BAD EXAMPLES:\n"
        f"• 'Sea creature like a fish' ← Contains '{secret_word}'\n"
        f"• 'Type of fish that swims' ← Contains '{secret_word}'\n"
        f"• 'Five-armed sea creature' ← Just defines the word, no relationship\n"
        f"• 'Small orange pet in bowls' ← Just defines the word, no relationship\n\n"
        
        f"📝 GUIDELINES:\n"
        f"• Write riddles from the SECRET WORD'S perspective (use 'I', 'me', 'my')\n"
        f"• Think: How does the secret word relate to the guess word?\n"
        f"• Use 'these', 'those', 'this thing', 'that activity' instead of the secret word\n"
        f"• Relationship patterns: 'I am a type of these', 'Used for catching me', 'I have more/less X'\n"
        f"• Assess connection strength: 'strong' (very related), 'medium' (somewhat related), 'weak' (barely related)\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {test_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Raw response:\n{content}\n")
        
        # Try to parse JSON
        try:
            results = json.loads(content)
            print("✅ RESULTS:")
            print("-" * 40)
            
            violations = []
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                status = "❌ VIOLATION" if violation else "✅ GOOD"
                
                print(f"{status} | {word:12} | {clue:40} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
            
            print("-" * 40)
            if violations:
                print(f"🚨 Found {len(violations)} violations:")
                for violation in violations:
                    print(f"   • {violation}")
                print("\n💡 The prompt needs further refinement!")
            else:
                print(f"🎉 SUCCESS! No violations found in {len(results)} clues!")
                print("✅ The improved prompt is working!")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print("🔄 Raw content might need cleaning...")
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        exit(1)
    
    test_fish_prompt()
