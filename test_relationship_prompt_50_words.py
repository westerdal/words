#!/usr/bin/env python3
"""
Test the relationship-focused prompt on 50 challenging words
"""

import openai
import os
import json

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_relationship_prompt_50_words():
    """Test the relationship-focused prompt with 50 challenging words"""
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
    
    print(f"🧪 Testing RELATIONSHIP-FOCUSED prompt on 50 challenging words")
    print(f"🐟 You are a '{secret_word}' describing your relationships")
    print(f"📝 Testing {len(hard_words)} words")
    print("=" * 80)
    
    # Use the relationship-focused prompt
    prompt = (
        f"CRITICAL: You are a '{secret_word}' speaking about guess words in a word guessing game. "
        f"For each guess word, write a riddle from YOUR PERSPECTIVE (as '{secret_word}') describing your RELATIONSHIP to that word in 7 words or less.\n\n"
        
        f"🚫 ABSOLUTE RULE: You cannot say your own name '{secret_word}' in any clue. Use 'I', 'me', 'my' instead.\n\n"
        
        f"✅ GOOD EXAMPLES (YOU are the {secret_word} speaking about guess words):\n"
        f"• Guess: 'starfish' → Clue: 'I am a type of these' (you share category)\n"
        f"• Guess: 'fishing' → Clue: 'Used for catching me' (describes relationship)\n"
        f"• Guess: 'goldfish' → Clue: 'I have less gold' (comparison)\n"
        f"• Guess: 'telescope' → Clue: 'I have nothing to do with this' (no relationship)\n\n"
        
        f"❌ FORBIDDEN EXAMPLES (will be REJECTED):\n"
        f"• 'I see far into the sky' ← Wrong perspective (telescope describing itself)\n"
        f"• 'People choose me to govern' ← Wrong perspective (democracy describing itself)\n"
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
        f"Words to process: {hard_words}"
    )
    
    try:
        print("🤖 Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        print(f"📥 Response length: {len(content)} characters")
        
        try:
            results = json.loads(content)
            print(f"\n✅ PROCESSING {len(results)} WORDS:")
            print("=" * 80)
            
            violations = []
            wrong_perspective = []
            good_relationships = []
            
            for word, data in results.items():
                clue = data.get('clue', '')
                strength = data.get('strength', '')
                
                # Check for secret word violation
                violation = secret_word.lower() in clue.lower()
                
                # Check if it's describing the relationship (fish's perspective)
                is_relationship = any(phrase in clue.lower() for phrase in [
                    'i am', 'i have', 'i live', 'i swim', 'i breathe', 'i eat', 'i need',
                    'used for', 'nothing to do', 'unrelated to me', 'completely different',
                    'similar to me', 'different from me', 'part of me', 'related to me',
                    'this has nothing', 'this is unrelated', 'this is completely'
                ])
                
                # Status indicators
                if violation:
                    status = "❌ VIOLATION"
                elif not is_relationship:
                    status = "⚠️ WRONG PERSPECTIVE"
                else:
                    status = "✅ GOOD"
                
                # Strength indicator
                strength_icon = {'strong': '🟢', 'medium': '🟡', 'weak': '🔴'}.get(strength.lower(), '⚪')
                
                print(f"{status} {strength_icon} | {word:12} | {clue:35} | {strength}")
                
                if violation:
                    violations.append(f"'{word}' → '{clue}'")
                elif not is_relationship:
                    wrong_perspective.append(f"'{word}' → '{clue}'")
                else:
                    good_relationships.append(f"'{word}' → '{clue}'")
            
            print("=" * 80)
            print(f"📊 DETAILED ANALYSIS:")
            print(f"   Total words processed: {len(results)}/{len(hard_words)}")
            print(f"   ❌ Secret word violations: {len(violations)}")
            print(f"   ⚠️ Wrong perspective: {len(wrong_perspective)}")
            print(f"   ✅ Good relationships: {len(good_relationships)}")
            
            if len(results) > 0:
                violation_rate = len(violations) / len(results) * 100
                relationship_rate = len(good_relationships) / len(results) * 100
                print(f"   🎯 Violation rate: {violation_rate:.1f}%")
                print(f"   🎯 Relationship success: {relationship_rate:.1f}%")
            
            if violations:
                print(f"\n🚨 SECRET WORD VIOLATIONS:")
                for violation in violations[:10]:
                    print(f"   • {violation}")
                if len(violations) > 10:
                    print(f"   ... and {len(violations) - 10} more")
            
            if wrong_perspective:
                print(f"\n⚠️ WRONG PERSPECTIVE (first 10):")
                for wrong in wrong_perspective[:10]:
                    print(f"   • {wrong}")
                if len(wrong_perspective) > 10:
                    print(f"   ... and {len(wrong_perspective) - 10} more")
            
            if good_relationships:
                print(f"\n🎉 GOOD RELATIONSHIPS (sample):")
                for good in good_relationships[:10]:
                    print(f"   • {good}")
                if len(good_relationships) > 10:
                    print(f"   ... and {len(good_relationships) - 10} more")
            
            # Overall assessment
            if len(results) > 0:
                if violation_rate == 0 and relationship_rate >= 85:
                    print(f"\n🏆 EXCELLENT! Ready for production!")
                    print(f"   ✅ No secret word violations")
                    print(f"   ✅ {relationship_rate:.1f}% proper relationships")
                elif violation_rate == 0 and relationship_rate >= 70:
                    print(f"\n👍 GOOD! Should work well for production!")
                    print(f"   ✅ No secret word violations") 
                    print(f"   ⚠️ {relationship_rate:.1f}% proper relationships (could be better)")
                elif violation_rate <= 5:
                    print(f"\n⚠️ NEEDS MINOR TWEAKS but close to ready!")
                    print(f"   ⚠️ {violation_rate:.1f}% secret word violations")
                    print(f"   ⚠️ {relationship_rate:.1f}% proper relationships")
                else:
                    print(f"\n❌ NEEDS MAJOR WORK before production!")
                    print(f"   ❌ {violation_rate:.1f}% secret word violations")
                    print(f"   ❌ {relationship_rate:.1f}% proper relationships")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw response (first 1000 chars):")
            print(content[:1000])
            print("..." if len(content) > 1000 else "")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_relationship_prompt_50_words()


