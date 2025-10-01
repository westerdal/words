#!/usr/bin/env python3
"""
Test the improved riddle-based prompt on 50 challenging words distant from 'fish'
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_riddle_prompt_hard_words():
    """Test the riddle prompt with 50 challenging words"""
    secret_word = "fish"
    
    # 50 challenging words - mix of very distant, abstract, and tricky ones
    hard_words = [
        # Abstract concepts
        "democracy", "philosophy", "mathematics", "justice", "freedom",
        # Distant objects  
        "telescope", "keyboard", "umbrella", "bicycle", "pencil",
        # Actions unrelated to fish
        "dancing", "singing", "writing", "painting", "running",
        # Places
        "library", "hospital", "airport", "theater", "museum",
        # Technology
        "computer", "internet", "software", "algorithm", "database",
        # Body parts
        "elbow", "shoulder", "eyebrow", "nostril", "ankle",
        # Weather/Nature (but not water-related)
        "thunder", "lightning", "tornado", "earthquake", "volcano",
        # Food (non-fish)
        "pizza", "chocolate", "coffee", "sandwich", "salad",
        # Emotions
        "jealousy", "curiosity", "excitement", "nervousness", "confidence",
        # Tools/Objects
        "hammer", "screwdriver", "ladder", "mirror", "clock"
    ]
    
    print(f"🧪 Testing IMPROVED riddle prompt for secret word: '{secret_word}'")
    print(f"📝 Testing {len(hard_words)} challenging words")
    print("=" * 80)
    
    # Create the IMPROVED riddle-based prompt (same as the working 10-word test)
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
        f"Words to process: {hard_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1  # Lower temperature for more consistent following of instructions
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Raw response length: {len(content)} characters\n")
        
        # Try to parse JSON
        try:
            results = json.loads(content)
            print("✅ RESULTS:")
            print("=" * 80)
            
            violations = []
            weak_connections = []
            good_riddles = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                status = "❌ VIOLATION" if violation else "✅"
                
                # Check if it's a proper riddle (uses I/me/my)
                is_riddle = any(pronoun in clue.lower() for pronoun in ['i ', 'me', 'my', 'this is'])
                riddle_icon = "🎭" if is_riddle else "📝"
                
                # Color code by strength
                strength_icon = {
                    'strong': '🟢',
                    'medium': '🟡', 
                    'weak': '🔴'
                }.get(strength.lower(), '⚪')
                
                print(f"{status} {riddle_icon} {strength_icon} | {word:12} | {clue:40} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                elif strength.lower() == 'weak':
                    weak_connections.append(word)
                else:
                    good_riddles.append(f"'{word}' → '{clue}'")
            
            print("=" * 80)
            print(f"📊 SUMMARY:")
            print(f"   Total words processed: {len(results)}")
            print(f"   ❌ Secret word violations: {len(violations)}")
            print(f"   🔴 Weak connections: {len(weak_connections)}")
            print(f"   ✅ Good riddles: {len(good_riddles)}")
            
            if violations:
                print(f"\n🚨 SECRET WORD VIOLATIONS:")
                for violation in violations[:10]:  # Show first 10
                    print(f"   • {violation}")
                if len(violations) > 10:
                    print(f"   ... and {len(violations) - 10} more")
            
            if good_riddles:
                print(f"\n🎉 EXCELLENT RIDDLES (sample):")
                for riddle in good_riddles[:10]:  # Show first 10
                    print(f"   • {riddle}")
                if len(good_riddles) > 10:
                    print(f"   ... and {len(good_riddles) - 10} more")
            
            # Overall assessment
            success_rate = (len(results) - len(violations)) / len(results) * 100 if results else 0
            print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}% (no secret word violations)")
            
            if success_rate >= 95:
                print("🏆 EXCELLENT! The riddle prompt is working very well!")
                print("✅ Ready to generate the full fish CSV!")
            elif success_rate >= 85:
                print("👍 GOOD! Minor improvements needed but mostly working!")
            elif success_rate >= 70:
                print("⚠️ OKAY! Needs some refinement but shows promise!")
            else:
                print("❌ NEEDS WORK! Too many violations, prompt needs major revision!")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print("🔄 Raw content (first 1000 chars):")
            print(content[:1000])
            print("..." if len(content) > 1000 else "")
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        exit(1)
    
    test_riddle_prompt_hard_words()


