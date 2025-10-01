#!/usr/bin/env python3
"""
Test the improved riddle prompt on 15 challenging words to verify consistency
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_riddle_prompt_medium_batch():
    """Test the riddle prompt with 15 challenging words"""
    secret_word = "fish"
    
    # 15 challenging words from different categories
    test_words = [
        "democracy", "telescope", "pizza", "dancing", "computer",
        "hammer", "freedom", "mathematics", "library", "thunder",
        "chocolate", "jealousy", "bicycle", "painting", "mirror"
    ]
    
    print(f"🧪 Testing riddle prompt on {len(test_words)} challenging words")
    print(f"📝 Secret word: '{secret_word}'")
    print(f"🎯 Words: {test_words}")
    print("=" * 70)
    
    # Use the same improved prompt that worked for 10 words
    prompt = (
        f"CRITICAL: You are generating riddle clues for a word guessing game. The secret word is '{secret_word}'. "
        f"You MUST write each clue from the SECRET WORD'S PERSPECTIVE as a riddle in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: The word '{secret_word}' is FORBIDDEN in any clue. ANY clue containing '{secret_word}' will be rejected.\n\n"
        
        f"✅ GOOD EXAMPLES (riddles from secret word's perspective):\n"
        f"• Guess: 'starfish' → Clue: 'I am a type of these'\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me'\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold'\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'Sea creature like a fish' ← Contains FORBIDDEN word\n"
        f"• 'Type of fish that swims' ← Contains FORBIDDEN word\n\n"
        
        f"📝 MANDATORY REQUIREMENTS:\n"
        f"• MUST write riddles from the SECRET WORD'S perspective using 'I', 'me', 'my'\n"
        f"• MUST use pronouns: 'these', 'those', 'this thing' - NEVER '{secret_word}'\n"
        f"• For distant words: 'I have nothing to do with this', 'This is completely unrelated to me'\n"
        f"• Connection strength: 'strong', 'medium', or 'weak'\n\n"
        
        f"REMINDER: The word '{secret_word}' is ABSOLUTELY FORBIDDEN. Use pronouns instead!\n\n"
        
        f"Return JSON format:\n"
        f'{{"word": {{"clue": "relationship description", "strength": "strong/medium/weak"}}}}\n\n'
        f"Words to process: {test_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Response length: {len(content)} characters")
        
        try:
            results = json.loads(content)
            print(f"\n✅ RESULTS ({len(results)} words processed):")
            print("-" * 70)
            
            violations = []
            good_riddles = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                status = "❌ VIOLATION" if violation else "✅ GOOD"
                
                # Check riddle format
                is_riddle = any(pronoun in clue.lower() for pronoun in ['i ', 'me', 'my', 'this is'])
                riddle_status = "🎭" if is_riddle else "📝"
                
                print(f"{status} {riddle_status} | {word:12} | {clue:35} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                else:
                    good_riddles.append(f"'{word}' → '{clue}'")
            
            print("-" * 70)
            success_rate = (len(results) - len(violations)) / len(results) * 100 if results else 0
            
            print(f"📊 RESULTS:")
            print(f"   Words processed: {len(results)}/{len(test_words)}")
            print(f"   ❌ Violations: {len(violations)}")
            print(f"   ✅ Good riddles: {len(good_riddles)}")
            print(f"   🎯 Success rate: {success_rate:.1f}%")
            
            if violations:
                print(f"\n🚨 VIOLATIONS:")
                for violation in violations:
                    print(f"   • {violation}")
            
            if success_rate >= 95:
                print(f"\n🏆 EXCELLENT! Ready for full CSV generation!")
            elif success_rate >= 80:
                print(f"\n👍 GOOD! Should work well for most cases!")
            else:
                print(f"\n⚠️ NEEDS MORE WORK!")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw response: {content}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_riddle_prompt_medium_batch()


