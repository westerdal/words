#!/usr/bin/env python3
"""
Test the improved riddle prompt with stronger language and examples
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_improved_riddle_prompt():
    """Test the improved riddle prompt with challenging words"""
    secret_word = "fish"
    
    # Test with 10 challenging words
    test_words = ["starfish", "goldfish", "fishing", "telescope", "democracy", "pizza", "dancing", "computer", "hammer", "freedom"]
    
    print(f"🧪 Testing IMPROVED riddle prompt for secret word: '{secret_word}'")
    print(f"📝 Test words: {test_words}")
    print("=" * 80)
    
    # Create the MUCH MORE FORCEFUL prompt
    prompt = (
        f"CRITICAL: You are generating riddle clues for a word guessing game. The secret word is '{secret_word}'. "
        f"You MUST write each clue from the SECRET WORD'S PERSPECTIVE as a riddle in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: The word '{secret_word}' is FORBIDDEN in any clue. ANY clue containing '{secret_word}' will be rejected.\n\n"
        
        f"✅ GOOD EXAMPLES (riddles from secret word's perspective):\n"
        f"• Guess: 'starfish' → Clue: 'I am a type of these'\n"
        f"• Guess: 'catfish' → Clue: 'I have whiskers but I am also one of these'\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me'\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold'\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'Sea creature like a fish' ← Contains FORBIDDEN word '{secret_word}'\n"
        f"• 'Type of fish that swims' ← Contains FORBIDDEN word '{secret_word}'\n"
        f"• 'Fish move gracefully' ← Contains FORBIDDEN word '{secret_word}'\n"
        f"• 'Used to spot fish' ← Contains FORBIDDEN word '{secret_word}'\n"
        f"• 'Five-armed sea creature' ← Not a riddle, just a definition\n\n"
        
        f"📝 MANDATORY REQUIREMENTS:\n"
        f"• MUST write riddles from the SECRET WORD'S perspective using 'I', 'me', 'my'\n"
        f"• MUST use pronouns: 'these', 'those', 'this thing', 'that activity' - NEVER '{secret_word}'\n"
        f"• MUST follow patterns: 'I am a type of these', 'Used for catching me', 'I have more/less X'\n"
        f"• For distant words, be creative: 'I have nothing to do with this', 'This is completely unrelated to me'\n"
        f"• Connection strength: 'strong' (very related), 'medium' (somewhat related), 'weak' (barely related)\n\n"
        
        f"REMINDER: The word '{secret_word}' is ABSOLUTELY FORBIDDEN. Use pronouns instead!\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {test_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI with IMPROVED prompt...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1  # Lower temperature for more consistent following of instructions
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Raw response:\n{content}\n")
        
        # Try to parse JSON
        try:
            results = json.loads(content)
            print("✅ RESULTS:")
            print("-" * 80)
            
            violations = []
            good_riddles = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                status = "❌ VIOLATION" if violation else "✅ GOOD"
                
                # Check if it's a proper riddle (uses I/me/my)
                is_riddle = any(pronoun in clue.lower() for pronoun in ['i ', 'me', 'my'])
                riddle_status = "🎭 RIDDLE" if is_riddle else "📝 NOT RIDDLE"
                
                print(f"{status} {riddle_status} | {word:12} | {clue:40} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                else:
                    good_riddles.append(f"'{word}' → '{clue}'")
            
            print("-" * 80)
            print(f"📊 IMPROVED RESULTS:")
            print(f"   Total words: {len(results)}")
            print(f"   ❌ Violations: {len(violations)}")
            print(f"   ✅ Good clues: {len(good_riddles)}")
            
            success_rate = (len(results) - len(violations)) / len(results) * 100 if results else 0
            print(f"   🎯 Success rate: {success_rate:.1f}%")
            
            if violations:
                print(f"\n🚨 REMAINING VIOLATIONS:")
                for violation in violations:
                    print(f"   • {violation}")
            
            if success_rate >= 90:
                print("\n🏆 EXCELLENT! The improved prompt is working!")
            elif success_rate >= 70:
                print("\n👍 GOOD! Much better than before!")
            else:
                print("\n⚠️ Still needs work, but improvement noted.")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")

if __name__ == "__main__":
    test_improved_riddle_prompt()


